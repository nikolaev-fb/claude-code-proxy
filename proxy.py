"""
Anthropic-to-OpenAI Proxy Server

FastAPI application that converts Anthropic-style API requests to OpenRouter
(OpenAI-compatible) format, handles responses, and returns them in Anthropic format.
"""
# Load environment variables from .env file FIRST, before importing config
from dotenv import load_dotenv
load_dotenv()

import logging
import json
from typing import AsyncGenerator, Dict, Any, Optional
import asyncio
import os
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse
import uvicorn

from config import settings
from transformers import (
    RequestTransformer,
    ResponseTransformer,
    ModelsResponseTransformer,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


def _setup_messages_logger() -> Optional[logging.Logger]:
    """
    Configure dedicated logger for /messages endpoint (if enabled).

    Returns:
        Configured logger instance that writes to logs/messages.log, or None if disabled
    """
    if not settings.FILE_LOGGING_ENABLED:
        logger.info("File logging disabled (FILE_LOGGING_ENABLED=false)")
        return None

    messages_log_dir = "logs"
    if not os.path.exists(messages_log_dir):
        os.makedirs(messages_log_dir)

    messages_log_file = os.path.join(messages_log_dir, "messages.log")
    msg_logger = logging.getLogger("messages")
    msg_logger.setLevel(logging.DEBUG)
    msg_logger.propagate = False
    msg_logger.handlers.clear()  # Avoid duplicates

    # File handler
    file_handler = logging.FileHandler(messages_log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    msg_logger.addHandler(file_handler)

    logger.info(f"Messages endpoint file logging enabled: {os.path.abspath(messages_log_file)}")
    return msg_logger


# Setup messages logger (optional)
messages_logger = _setup_messages_logger()


def _log_message(level: str, message: str) -> None:
    """
    Log message to messages.log if file logging is enabled.

    Args:
        level: Log level ('info', 'debug', 'error', 'warning')
        message: Message to log
    """
    if not messages_logger:
        return
    log_func = getattr(messages_logger, level.lower(), messages_logger.info)
    log_func(message)


def _update_trace_safe(trace: Optional[Any], output: Optional[Dict] = None, metadata: Optional[Dict] = None) -> None:
    """
    Safely update LangFuse trace with error handling.

    Args:
        trace: LangFuse trace object (or None)
        output: Output data to log
        metadata: Metadata to attach to trace
    """
    if not trace:
        return
    try:
        trace.update(output=output, metadata=metadata)
    except Exception as e:
        logger.warning(f"Failed to update LangFuse trace: {e}")


def _log_stream_event(anthropic_event: str) -> None:
    """
    Log streaming event to messages.log with structured parsing (if enabled).

    Args:
        anthropic_event: SSE event string in Anthropic format (may contain multiple events)
    """
    if not messages_logger:
        return  # File logging disabled

    try:
        # Split by double newlines to separate multiple SSE events
        # Example: "event: xxx\ndata: {...}\n\nevent: yyy\ndata: {...}\n\n"
        events = anthropic_event.strip().split('\n\n')

        for event_block in events:
            if not event_block.strip():
                continue

            # Parse each SSE event block
            lines = event_block.strip().split('\n')
            data_line = None

            for line in lines:
                if line.startswith('data: '):
                    data_line = line[6:]  # Remove "data: " prefix
                    break

            if not data_line:
                continue

            event_data = json.loads(data_line)
            event_type = event_data.get('type', 'unknown')

            msg_id = ''
            if event_type == 'message_start':
                msg_id = event_data.get('message', {}).get('id')
                usage = event_data.get('message', {}).get('usage', {})
                messages_logger.info(f"[STREAM] message_start - ID: {msg_id}")
                messages_logger.debug(f"  Initial usage: input_tokens={usage.get('input_tokens', 0)}, output_tokens={usage.get('output_tokens', 0)}")

            elif event_type == 'content_block_start':
                messages_logger.debug(f"[STREAM {msg_id}] content_block_start")

            elif event_type == 'content_block_delta':
                delta = event_data.get('delta', {})
                delta_type = delta.get('type', '')

                if delta_type == 'text_delta':
                    text = delta.get('text', '')
                    if text:
                        messages_logger.info(f"[STREAM {msg_id}] content_block_delta (text) - Text: {text}")
                    else:
                        messages_logger.debug(f"[STREAM {msg_id}] content_block_delta (text) - empty")
                elif delta_type == 'input_json_delta':
                    partial_json = delta.get('partial_json', '')
                    if partial_json:
                        messages_logger.info(f"[STREAM {msg_id}] content_block_delta (input_json) - JSON: {partial_json}")
                    else:
                        messages_logger.debug(f"[STREAM {msg_id}] content_block_delta (input_json) - empty")
                else:
                    messages_logger.debug(f"[STREAM {msg_id}] content_block_delta (unknown type: {delta_type})")
                    messages_logger.debug(event_data)

            elif event_type == 'content_block_stop':
                messages_logger.debug(f"[STREAM {msg_id}] content_block_stop")

            elif event_type == 'message_delta':
                stop_reason = event_data.get('delta', {}).get('stop_reason')
                usage = event_data.get('usage', {})
                messages_logger.info(f"[STREAM {msg_id}] message_delta - Stop Reason: {stop_reason}")
                messages_logger.debug(f"  Final usage: input_tokens={usage.get('input_tokens', 0)}, output_tokens={usage.get('output_tokens', 0)}, cache_read={usage.get('cache_read_input_tokens', 0)}, cache_creation={usage.get('cache_creation_input_tokens', 0)}")

            elif event_type == 'message_stop':
                messages_logger.debug(f"[STREAM {msg_id}] message_stop")

            else:
                messages_logger.info(f"[STREAM {msg_id}] {event_type}")

    except Exception as log_error:
        logger.debug(f"Error logging stream event: {log_error}")
        messages_logger.debug(f"[ERROR] logging stream event: {log_error}")
        messages_logger.debug(f"[STREAM {msg_id}] Event (raw): {anthropic_event}")


# LangFuse integration (optional)
langfuse_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    global langfuse_client

    # Initialize LangFuse if enabled
    if settings.LANGFUSE_ENABLED:
        try:
            from langfuse import Langfuse
            langfuse_client = Langfuse(
                public_key=settings.LANGFUSE_API_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            logger.info("LangFuse logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize LangFuse: {e}")
            langfuse_client = None

    # Validate configuration
    try:
        settings.validate()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise

    # Log startup info
    logger.info(f"Starting proxy on {settings.PROXY_HOST}:{settings.PROXY_PORT}")
    logger.info(f"OpenRouter base URL: {settings.OPENROUTER_BASE_URL}")
    if settings.LANGFUSE_ENABLED:
        logger.info(f"LangFuse host: {settings.LANGFUSE_HOST}")

    yield  # Application runs here

    # Shutdown (cleanup if needed)
    logger.info("Shutting down proxy server")


# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="Anthropic-to-OpenRouter Proxy",
    description="Proxy that converts Anthropic API requests to OpenRouter format",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "anthropic-to-openrouter-proxy"}


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    """
    Main messages endpoint - converts Anthropic format to OpenRouter and back.
    Supports both streaming and non-streaming responses.
    """
    trace = None
    try:
        # Parse incoming request
        request_body = await request.json()
        logger.debug(f"Incoming request: {json.dumps(request_body, indent=2)}")

        # Log to messages file (simplified - no full request body)
        _log_message("info", "=" * 80)
        _log_message("info", "RAW INCOMING REQUEST")
        _log_message("info", json.dumps(request_body, indent=2))
        _log_message("info", f"Timestamp: {datetime.now().isoformat()}")

        # Extract model and convert to OpenRouter format
        anthropic_model = request_body.get("model", "")
        openrouter_model = settings.get_openrouter_model(anthropic_model)
        logger.info(f"Converting model {anthropic_model} -> {openrouter_model}")
        _log_message("info", f"Model Conversion: {anthropic_model} -> {openrouter_model}")

        # IMPORTANT: Detect and handle priming assistant message
        # Claude Code sometimes sends a last assistant message with priming text (like "{")
        # to guide JSON responses. We need to:
        # 1. Remove it from the request to OpenRouter (so model generates full response)
        # 2. Strip the priming text from the start of our response (to avoid duplication in client)
        modified_request = request_body.copy()
        messages = modified_request.get("messages", [])
        priming_text_to_strip = ""  # Will hold the priming text we need to strip from response

        if messages and messages[-1].get("role") == "assistant":
            last_msg_content = messages[-1].get("content", "")
            # Check if it's a short priming text (< 10 chars) that looks like JSON start
            is_priming = False
            priming_text = ""

            if isinstance(last_msg_content, str) and len(last_msg_content) < 10:
                is_priming = True
                priming_text = last_msg_content
            elif isinstance(last_msg_content, list):
                # Array format - check if total text is short
                total_text = ""
                for item in last_msg_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_text += item.get("text", "")
                if len(total_text) < 10:
                    is_priming = True
                    priming_text = total_text

            if is_priming:
                logger.info(f"Detected priming assistant message: {repr(priming_text)}")
                _log_message("info", f"Detected priming assistant message: {repr(priming_text)}")
                modified_request["messages"] = messages[:-1]
                priming_text_to_strip = priming_text

        # Transform request to OpenRouter format
        openrouter_request = RequestTransformer.transform_messages_request(
            modified_request,
            openrouter_model,
        )
        max_tokens = max(openrouter_request.get('max_tokens', 32000), 32000)
        openrouter_request["max_tokens"] = max_tokens
        openrouter_request["max_output_tokens"] = max_tokens
        logger.debug(f"Transformed request: {json.dumps(openrouter_request, indent=2)}")

        # Log transformed OpenRouter request to messages.log
        _log_message("info", "-" * 80)
        if "tools" in openrouter_request:
            _log_message("info", f"Tools count: {len(openrouter_request['tools'])}")
            for idx, tool in enumerate(openrouter_request['tools']):
                _log_message("info", f"  Tool {idx + 1}: {tool.get('function', {}).get('name', 'unknown')}")
        _log_message("info", "-" * 80)

        # Create LangFuse trace AFTER transformation (captures the ACTUAL request sent to OpenRouter)
        if langfuse_client:
            try:
                trace = langfuse_client.trace(
                    name="anthropic_proxy_request",
                    input={
                        "original_request": {
                            "model": anthropic_model,
                            "messages": request_body.get("messages"),
                            "system": request_body.get("system"),
                            "tools": request_body.get("tools"),
                            "max_tokens": request_body.get("max_tokens"),
                            "temperature": request_body.get("temperature"),
                            "stream": request_body.get("stream", False),
                            "metadata": request_body.get("metadata"),
                        },
                        "openrouter_request": openrouter_request,
                    }
                )
                logger.debug("LangFuse trace created with full OpenRouter request")
            except Exception as e:
                logger.warning(f"Failed to create LangFuse trace: {e}")
                trace = None

        # Check if streaming is requested
        is_streaming = request_body.get("stream", False)

        _log_message("info", f"Streaming: {is_streaming}")

        # DISABLED: Extract initial assistant content (priming text like "{" for JSON)
        # This was causing duplicate symbols (e.g., "{{") when OpenRouter models
        # already generate the priming text. Commenting out to prevent duplication.
        initial_assistant_content = ""
        # if is_streaming:
        #     messages = request_body.get("messages", [])
        #     if messages and messages[-1].get("role") == "assistant":
        #         # Last message is from assistant - extract its content for prepending
        #         last_msg = messages[-1].get("content", [])
        #         if isinstance(last_msg, list):
        #             # Array format
        #             for item in last_msg:
        #                 if isinstance(item, dict) and item.get("type") == "text":
        #                     initial_assistant_content += item.get("text", "")
        #         elif isinstance(last_msg, str):
        #             # String format
        #             initial_assistant_content = last_msg

        if is_streaming:
            return StreamingResponse(
                stream_messages(openrouter_request, anthropic_model, trace, initial_assistant_content, priming_text_to_strip),
                media_type="text/event-stream",
            )
        else:
            return await handle_non_streaming_messages(
                openrouter_request,
                anthropic_model,
                request_body,
                trace,
            )

    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        _log_message("error", "ERROR: Invalid JSON in request body")
        _update_trace_safe(trace, output=None, metadata={"status": "error", "error": "Invalid JSON"})
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        _log_message("error", f"ERROR: {str(e)}")
        _update_trace_safe(trace, output=None, metadata={"status": "error", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


async def handle_non_streaming_messages(
    openrouter_request: Dict[str, Any],
    anthropic_model: str,
    anthropic_request: Optional[Dict[str, Any]] = None,
    trace: Optional[Any] = None,
) -> Response:
    """
    Handle non-streaming message requests.

    Args:
        openrouter_request: Transformed request for OpenRouter
        anthropic_model: Original Anthropic model name
        anthropic_request: Original Anthropic request (for logging system prompt)
        trace: LangFuse trace object (for logging)
    """
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Call OpenRouter API
            response = await client.post(
                f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                json=openrouter_request,
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
            )

            if response.status_code != 200:
                logger.error(f"OpenRouter error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenRouter API error: {response.text}",
                )

            openrouter_response = response.json()
            logger.debug(f"OpenRouter response: {json.dumps(openrouter_response, indent=2)}")

            # Transform response back to Anthropic format
            anthropic_response = ResponseTransformer.transform_messages_response(
                openrouter_response,
                anthropic_model,
            )
            logger.debug(f"Transformed response: {json.dumps(anthropic_response, indent=2)}")

            # Update LangFuse trace with output if it was created
            _update_trace_safe(
                trace,
                output={
                    "model": anthropic_response.get("model"),
                    "content": anthropic_response.get("content", []),
                    "stop_reason": anthropic_response.get("stop_reason"),
                    "usage": anthropic_response.get("usage", {}),
                },
                metadata={
                    "status": "success",
                    "openrouter_model": settings.get_openrouter_model(anthropic_model),
                }
            )

            return Response(
                content=json.dumps(anthropic_response),
                media_type="application/json",
            )

        except httpx.TimeoutException:
            logger.error("Request to OpenRouter timed out")
            _update_trace_safe(trace, output=None, metadata={"status": "timeout"})
            raise HTTPException(status_code=504, detail="Request to OpenRouter timed out")
        except httpx.RequestError as e:
            logger.error(f"Error connecting to OpenRouter: {e}")
            _update_trace_safe(trace, output=None, metadata={"status": "error", "error": str(e)})
            raise HTTPException(status_code=502, detail="Error connecting to OpenRouter")


async def stream_messages(
    openrouter_request: Dict[str, Any],
    anthropic_model: str,
    trace: Optional[Any] = None,
    initial_assistant_content: str = "",
    priming_text_to_strip: str = "",
) -> AsyncGenerator[str, None]:
    """
    Stream messages from OpenRouter and convert to Anthropic format.

    Args:
        openrouter_request: Transformed request for OpenRouter
        anthropic_model: Original Anthropic model name
        trace: LangFuse trace object (for logging)
        initial_assistant_content: Priming text from assistant message (DEPRECATED - not used)
        priming_text_to_strip: Text to strip from beginning of response (e.g., "{" if client sent assistant priming)
    """
    # Collect streaming metadata
    events_received = 0
    final_usage = None
    stream_error = None
    message_started = False  # Track if we've already sent message_start
    content_block_started = False  # Track if we've already sent content_block_start
    first_delta_received = False  # Track if first delta has been sent
    first_content_received = False  # Track if we've received first content (for stripping priming)
    accumulated_text = ""  # Start with empty text (no priming injection)
    tool_blocks_started = set()  # Track which tool call blocks have already been started (by index)

    if priming_text_to_strip:
        logger.info(f"Will strip priming text from response start: {repr(priming_text_to_strip)}")
        _log_message("info", f"Will strip priming text from response: {repr(priming_text_to_strip)}")

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            async with client.stream(
                "POST",
                f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                json=openrouter_request,
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"OpenRouter error: {response.status_code} - {error_text}")
                    yield f"event: error\ndata: {json.dumps({'error': error_text.decode()})}\n\n"
                    return

                # Peek at first chunk (priming text injection is now DISABLED)
                first_chunk = None
                # first_delta_content = None  # No longer needed (priming disabled)
                # should_inject_priming = False  # No longer needed (priming disabled)

                # Read lines from the stream
                line_iterator = response.aiter_lines()
                async for line in line_iterator:
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        chunk_str = line[6:]  # Remove "data: " prefix
                        if chunk_str == "[DONE]":
                            break

                        try:
                            first_chunk = json.loads(chunk_str)
                            # DISABLED: Extract first delta content (no longer needed - priming disabled)
                            # if "choices" in first_chunk:
                            #     choices = first_chunk.get("choices", [])
                            #     if choices:
                            #         delta = choices[0].get("delta", {})
                            #         first_delta_content = delta.get("content", "")

                            # DISABLED: Decide whether to inject priming text
                            # This causes duplicate symbols (e.g., "{{") because OpenRouter
                            # models already generate the priming text from the assistant message.
                            # Commenting out to prevent duplication issues.
                            # if initial_assistant_content and first_delta_content:
                            #     if not first_delta_content.startswith(initial_assistant_content):
                            #         should_inject_priming = True
                            #         _log_message("info", f"[STREAM] Model's first delta doesn't start with priming text, will inject: {repr(initial_assistant_content)}")
                            #     else:
                            #         _log_message("info", f"[STREAM] Model's first delta already starts with priming text, skipping injection")
                            # elif initial_assistant_content and not first_delta_content:
                            #     # No content in first delta, inject priming
                            #     should_inject_priming = True
                            #     _log_message("info", f"[STREAM] First delta has no content, will inject priming text")

                            # DISABLED: Inject synthetic events (was causing duplicate priming text)
                            # if should_inject_priming:
                            #     # Send message_start first
                            #     message_start_chunk = {
                            #         "choices": [
                            #             {
                            #                 "delta": {"role": "assistant"},
                            #                 "finish_reason": None
                            #             }
                            #         ]
                            #     }
                            #     message_start_event = ResponseTransformer.transform_streaming_chunk(
                            #         message_start_chunk,
                            #         anthropic_model,
                            #         skip_message_start=message_started,
                            #         skip_content_block_start=content_block_started,
                            #     )
                            #     if message_start_event:
                            #         yield message_start_event
                            #         events_received += 1
                            #         if "message_start" in message_start_event:
                            #             message_started = True
                            #         _log_message("debug", "[STREAM] Injected synthetic message_start")

                            #     # Send priming text as content_block_delta
                            #     priming_chunk = {
                            #         "choices": [
                            #             {
                            #                 "delta": {"content": initial_assistant_content},
                            #                 "finish_reason": None
                            #             }
                            #         ]
                            #     }
                            #     priming_event = ResponseTransformer.transform_streaming_chunk(
                            #         priming_chunk,
                            #         anthropic_model,
                            #         skip_message_start=True,
                            #         skip_content_block_start=content_block_started,
                            #     )
                            #     if priming_event:
                            #         yield priming_event
                            #         events_received += 1
                            #         # if not accumulated_text.startswith('{'):
                            #         # accumulated_text += initial_assistant_content

                            #         if "content_block_start" in priming_event:
                            #             content_block_started = True
                            #         if "content_block_delta" in priming_event:
                            #             first_delta_received = True
                            #         _log_message("info", f"[STREAM] Injected priming text: {repr(initial_assistant_content)}")

                            # Now process the first chunk we peeked at
                            break
                        except json.JSONDecodeError as e:
                            logger.debug(f"Failed to parse first chunk: {e}")
                            continue

                # Process the first chunk we peeked at
                if first_chunk:
                    logger.debug(f"Processing peeked first chunk: {json.dumps(first_chunk)}")

                    # Collect usage from chunks
                    if "usage" in first_chunk:
                        final_usage = first_chunk["usage"]

                    # Collect text content from delta chunks
                    # IMPORTANT: Strip priming text from first content if needed
                    if "choices" in first_chunk:
                        choices = first_chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                content = delta["content"]

                                # Strip priming text from first content chunk
                                if priming_text_to_strip and not first_content_received:
                                    if content.startswith(priming_text_to_strip):
                                        content = content[len(priming_text_to_strip):]
                                        logger.info(f"Stripped priming text from first chunk: {repr(priming_text_to_strip)}")
                                        _log_message("info", f"Stripped priming text from first chunk: {repr(priming_text_to_strip)}")
                                        # Modify the chunk before transformation
                                        first_chunk["choices"][0]["delta"]["content"] = content
                                    first_content_received = True

                                accumulated_text += content

                    # Transform chunk to Anthropic format
                    anthropic_event = ResponseTransformer.transform_streaming_chunk(
                        first_chunk,
                        anthropic_model,
                        skip_message_start=message_started,
                        skip_content_block_start=content_block_started,
                        tool_blocks_started=tool_blocks_started,
                    )

                    if anthropic_event:
                        # Track if we've now sent message_start
                        if "message_start" in anthropic_event:
                            message_started = True

                        # Track if we've now sent content_block_start
                        if "content_block_start" in anthropic_event:
                            content_block_started = True

                        # Track if we've sent first delta
                        if "content_block_delta" in anthropic_event and not first_delta_received:
                            first_delta_received = True

                        # Log streaming event to messages.log
                        _log_stream_event(anthropic_event)

                        yield anthropic_event
                        events_received += 1

                # Stream and convert remaining chunks
                # Buffer to hold finish chunk until we check next chunk for usage
                pending_finish_chunk = None

                async for line in line_iterator:
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        chunk_str = line[6:]  # Remove "data: " prefix
                        if chunk_str == "[DONE]":
                            # If we have a pending finish chunk, send it now
                            if pending_finish_chunk:
                                anthropic_event = ResponseTransformer.transform_streaming_chunk(
                                    pending_finish_chunk,
                                    anthropic_model,
                                    skip_message_start=message_started,
                                    skip_content_block_start=content_block_started,
                                    accumulated_usage=final_usage,
                                    tool_blocks_started=tool_blocks_started,
                                )
                                if anthropic_event:
                                    yield anthropic_event
                                    events_received += 1
                            break

                        try:
                            openrouter_chunk = json.loads(chunk_str)
                            logger.debug(f"OpenRouter chunk: {json.dumps(openrouter_chunk)}")

                            # Collect usage from chunks
                            if "usage" in openrouter_chunk:
                                final_usage = openrouter_chunk["usage"]
                                logger.debug(f"Collected usage from chunk: {final_usage}")

                            # Collect text content from delta chunks
                            # IMPORTANT: Strip priming text from first content if needed
                            if "choices" in openrouter_chunk:
                                choices = openrouter_chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        content = delta["content"]

                                        # Strip priming text from first content chunk
                                        if priming_text_to_strip and not first_content_received:
                                            if content.startswith(priming_text_to_strip):
                                                content = content[len(priming_text_to_strip):]
                                                logger.info(f"Stripped priming text from chunk: {repr(priming_text_to_strip)}")
                                                _log_message("info", f"Stripped priming text from chunk: {repr(priming_text_to_strip)}")
                                                # Modify the chunk before transformation
                                                openrouter_chunk["choices"][0]["delta"]["content"] = content
                                            first_content_received = True

                                        accumulated_text += content

                            # Check if this chunk has finish_reason (indicates end of stream)
                            has_finish_reason = False
                            if "choices" in openrouter_chunk:
                                choices = openrouter_chunk.get("choices", [])
                                if choices and choices[0].get("finish_reason") is not None:
                                    has_finish_reason = True

                            # If we have a pending finish chunk and current chunk has usage, merge them
                            if pending_finish_chunk and "usage" in openrouter_chunk:
                                logger.debug("Found usage chunk after finish chunk - merging usage data")
                                # Transform the pending finish chunk with the accumulated usage
                                anthropic_event = ResponseTransformer.transform_streaming_chunk(
                                    pending_finish_chunk,
                                    anthropic_model,
                                    skip_message_start=message_started,
                                    skip_content_block_start=content_block_started,
                                    accumulated_usage=final_usage,
                                    tool_blocks_started=tool_blocks_started,
                                )
                                if anthropic_event:
                                    yield anthropic_event
                                    events_received += 1
                                pending_finish_chunk = None
                                # Skip processing this usage-only chunk as event
                                continue

                            # If we had a pending finish chunk but this isn't usage, send the pending one first
                            if pending_finish_chunk:
                                logger.debug("Sending pending finish chunk (no usage in next chunk)")
                                anthropic_event = ResponseTransformer.transform_streaming_chunk(
                                    pending_finish_chunk,
                                    anthropic_model,
                                    skip_message_start=message_started,
                                    skip_content_block_start=content_block_started,
                                    accumulated_usage=final_usage,
                                    tool_blocks_started=tool_blocks_started,
                                )
                                if anthropic_event:
                                    yield anthropic_event
                                    events_received += 1
                                pending_finish_chunk = None

                            # If this chunk has finish_reason, buffer it to check next chunk for usage
                            if has_finish_reason:
                                logger.debug("Buffering finish chunk to check next chunk for usage")
                                pending_finish_chunk = openrouter_chunk
                                continue

                            # Transform chunk to Anthropic format (only send message_start and content_block_start once)
                            anthropic_event = ResponseTransformer.transform_streaming_chunk(
                                openrouter_chunk,
                                anthropic_model,
                                skip_message_start=message_started,
                                skip_content_block_start=content_block_started,
                            )

                            if anthropic_event:
                                # Track if we've now sent message_start
                                if "message_start" in anthropic_event:
                                    message_started = True

                                # Track if we've now sent content_block_start
                                if "content_block_start" in anthropic_event:
                                    content_block_started = True

                                # Track if we've sent first delta
                                if "content_block_delta" in anthropic_event and not first_delta_received:
                                    first_delta_received = True

                                # Log streaming event to messages.log
                                _log_stream_event(anthropic_event)

                                yield anthropic_event
                                events_received += 1
                                logger.debug(f"Streamed event: {anthropic_event[:100]}...")

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse chunk: {e}")
                            stream_error = str(e)
                            continue

                # Stream completed successfully
                logger.info(f"Stream completed: {events_received} events, trace: {trace is not None}")

                # Log stream summary to messages.log
                _log_message("info", "=" * 80)
                _log_message("info", "STREAMING RESPONSE COMPLETED")
                _log_message("info", f"Total Events: {events_received}")
                _log_message("info", f"Total Text Characters: {len(accumulated_text)}")
                if final_usage:
                    _log_message("info", f"Final Usage: {final_usage}")
                _log_message("info", "=" * 80)

                logger.debug(f"Updating trace with output: events={events_received}, text length={len(accumulated_text)}, usage={final_usage}")
                _update_trace_safe(
                    trace,
                    output={
                        "model": anthropic_model,
                        "content": [{"type": "text", "text": accumulated_text}],  # Match non-streaming format
                        "stop_reason": "end_turn",  # Streaming completed successfully
                        "usage": final_usage if final_usage else {},
                    },
                    metadata={
                        "status": "streaming_completed",
                        "events_count": events_received,
                        "text_length": len(accumulated_text),
                    }
                )
                if trace:
                    logger.info(f"Trace updated successfully for streaming request with {len(accumulated_text)} chars")

        except httpx.TimeoutException:
            logger.error("Request to OpenRouter timed out during streaming")
            _update_trace_safe(
                trace,
                output={
                    "model": anthropic_model,
                    "content": [{"type": "text", "text": accumulated_text}] if accumulated_text else [],
                    "stop_reason": "timeout",
                    "usage": final_usage if final_usage else {},
                },
                metadata={"status": "timeout", "text_partial": len(accumulated_text) > 0}
            )
            yield f"event: error\ndata: {json.dumps({'error': 'Request timed out'})}\n\n"
        except httpx.RequestError as e:
            logger.error(f"Error connecting to OpenRouter: {e}")
            _update_trace_safe(
                trace,
                output={
                    "model": anthropic_model,
                    "content": [{"type": "text", "text": accumulated_text}] if accumulated_text else [],
                    "stop_reason": "error",
                    "usage": final_usage if final_usage else {},
                },
                metadata={"status": "connection_error", "error": str(e), "text_partial": len(accumulated_text) > 0}
            )
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"Unexpected error during streaming: {e}", exc_info=True)
            _update_trace_safe(
                trace,
                output={
                    "model": anthropic_model,
                    "content": [{"type": "text", "text": accumulated_text}] if accumulated_text else [],
                    "stop_reason": "error",
                    "usage": final_usage if final_usage else {},
                },
                metadata={"status": "error", "error": str(e), "text_partial": len(accumulated_text) > 0}
            )
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@app.get("/v1/models")
async def list_models() -> Response:
    """
    List available models - fetches Claude models from OpenRouter.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.OPENROUTER_BASE_URL}/models",
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                },
            )

            if response.status_code != 200:
                logger.error(f"OpenRouter error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Error fetching models from OpenRouter",
                )

            openrouter_models = response.json().get("data", [])
            logger.debug(f"Fetched {len(openrouter_models)} models from OpenRouter")

            # Transform to Anthropic format
            anthropic_response = ModelsResponseTransformer.transform_models_response(
                openrouter_models
            )
            logger.debug(f"Transformed {len(anthropic_response['data'])} models to Anthropic format")

            return Response(
                content=json.dumps(anthropic_response),
                media_type="application/json",
            )

    except httpx.TimeoutException:
        logger.error("Request to OpenRouter timed out")
        raise HTTPException(status_code=504, detail="Request to OpenRouter timed out")
    except httpx.RequestError as e:
        logger.error(f"Error connecting to OpenRouter: {e}")
        raise HTTPException(status_code=502, detail="Error connecting to OpenRouter")
    except Exception as e:
        logger.error(f"Error fetching models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> Response:
    """
    Get details for a specific model.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.OPENROUTER_BASE_URL}/models",
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                },
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Error fetching models from OpenRouter",
                )

            # Search for the model in OpenRouter format
            openrouter_model_id = settings.get_openrouter_model(model_id)
            all_models = response.json().get("data", [])

            # Find matching model
            matching_model = None
            for model in all_models:
                if model.get("id") == openrouter_model_id or model.get("canonical_slug") == openrouter_model_id:
                    matching_model = model
                    break

            if not matching_model:
                logger.warning(f"Model not found: {model_id}")
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

            # Transform to Anthropic format
            anthropic_model = ModelsResponseTransformer.transform_single_model_response(
                matching_model
            )

            return Response(
                content=json.dumps(anthropic_model),
                media_type="application/json",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint - provides API information."""
    return {
        "name": "Anthropic-to-OpenRouter Proxy",
        "version": "1.0.0",
        "description": "Converts Anthropic API requests to OpenRouter format",
        "endpoints": {
            "health": "GET /health",
            "messages": "POST /v1/messages",
            "models": "GET /v1/models",
            "model_detail": "GET /v1/models/{model_id}",
        },
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return Response(
        content=json.dumps({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": exc.detail,
            }
        }),
        status_code=exc.status_code,
        media_type="application/json",
    )


def main():
    """Run the proxy server."""
    logger.info("Starting Anthropic-to-OpenRouter Proxy")
    logger.info(f"Configuration:")
    logger.info(f"  Host: {settings.PROXY_HOST}")
    logger.info(f"  Port: {settings.PROXY_PORT}")
    logger.info(f"  Log Level: {settings.LOG_LEVEL}")
    logger.info(f"  LangFuse: {'Enabled' if settings.LANGFUSE_ENABLED else 'Disabled'}")

    uvicorn.run(
        "proxy:app",
        host=settings.PROXY_HOST,
        port=settings.PROXY_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
