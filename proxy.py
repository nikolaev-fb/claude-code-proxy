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


def _setup_endpoint_logger(endpoint_name: str) -> Optional[logging.Logger]:
    """
    Configure dedicated logger for a specific endpoint (if enabled).

    Args:
        endpoint_name: Name of the endpoint (e.g., 'messages', 'models', 'count_tokens')

    Returns:
        Configured logger instance that writes to logs/{endpoint_name}.log, or None if disabled
    """
    if not settings.FILE_LOGGING_ENABLED:
        return None

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"{endpoint_name}.log")
    endpoint_logger = logging.getLogger(endpoint_name)
    endpoint_logger.setLevel(logging.DEBUG)
    endpoint_logger.propagate = False
    endpoint_logger.handlers.clear()  # Avoid duplicates

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    endpoint_logger.addHandler(file_handler)

    logger.info(f"{endpoint_name} endpoint file logging enabled: {os.path.abspath(log_file)}")
    return endpoint_logger


def _setup_messages_logger() -> Optional[logging.Logger]:
    """
    Configure dedicated logger for /messages endpoint (if enabled).

    Returns:
        Configured logger instance that writes to logs/messages.log, or None if disabled
    """
    return _setup_endpoint_logger("messages")


def _setup_streaming_logger() -> Optional[logging.Logger]:
    """
    Configure dedicated logger for streaming responses (if enabled).

    Returns:
        Configured logger instance that writes to logs/streaming.log, or None if disabled
    """
    return _setup_endpoint_logger("streaming")


def _log_endpoint_request(endpoint_logger: Optional[logging.Logger], endpoint_name: str, request_data: Dict[str, Any]) -> None:
    """
    Log request to endpoint-specific log file in JSON format with pretty print.

    Args:
        endpoint_logger: Logger instance for the endpoint
        endpoint_name: Name of the endpoint
        request_data: Request data to log
    """
    if not endpoint_logger:
        return

    endpoint_logger.info("=" * 80)
    endpoint_logger.info(f"{endpoint_name.upper()} REQUEST")
    endpoint_logger.info(json.dumps(request_data, indent=2))
    endpoint_logger.info(f"Timestamp: {datetime.now().isoformat()}")
    endpoint_logger.info("=" * 80)


# Setup messages logger (optional)
messages_logger = _setup_messages_logger()

# Setup streaming logger (optional)
streaming_logger = _setup_streaming_logger()


def _setup_raw_openrouter_logger() -> Optional[logging.Logger]:
    """
    Configure dedicated logger for raw OpenRouter streaming chunks (if enabled).

    Returns:
        Configured logger that writes raw JSON chunks to logs/openrouter-streaming.log, or None if disabled
    """
    if not settings.FILE_LOGGING_ENABLED:
        return None

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "openrouter-streaming.log")
    raw_logger = logging.getLogger("openrouter_streaming")
    raw_logger.setLevel(logging.DEBUG)
    raw_logger.propagate = False
    raw_logger.handlers.clear()

    # File handler - raw JSON only, no timestamp
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(message)s'))  # Raw content only
    raw_logger.addHandler(file_handler)

    logger.info(f"OpenRouter streaming logging enabled: {os.path.abspath(log_file)}")
    return raw_logger


def _setup_raw_anthropic_logger() -> Optional[logging.Logger]:
    """
    Configure dedicated logger for raw Anthropic SSE events (if enabled).

    Returns:
        Configured logger that writes raw SSE events to logs/anthropic-streaming.log, or None if disabled
    """
    if not settings.FILE_LOGGING_ENABLED:
        return None

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "anthropic-streaming.log")
    raw_logger = logging.getLogger("anthropic_streaming")
    raw_logger.setLevel(logging.DEBUG)
    raw_logger.propagate = False
    raw_logger.handlers.clear()

    # File handler - raw SSE only, no timestamp
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(message)s'))  # Raw content only
    raw_logger.addHandler(file_handler)

    logger.info(f"Anthropic streaming logging enabled: {os.path.abspath(log_file)}")
    return raw_logger


# Setup raw streaming loggers (optional)
openrouter_streaming_logger = _setup_raw_openrouter_logger()
anthropic_streaming_logger = _setup_raw_anthropic_logger()


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


def _log_streaming(level: str, message: str) -> None:
    """
    Log message to streaming.log if file logging is enabled.

    Args:
        level: Log level ('info', 'debug', 'error', 'warning')
        message: Message to log
    """
    if not streaming_logger:
        return
    log_func = getattr(streaming_logger, level.lower(), streaming_logger.info)
    log_func(message)


def _update_generation_safe(
    generation: Optional[Any],
    output: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    usage: Optional[Dict] = None
) -> None:
    """
    Safely update LangFuse generation with error handling.

    Updates both generation-level data (output, usage_details) and trace-level data (metadata).

    Args:
        generation: LangFuse generation object (or None)
        output: Output data to log (generation-level)
        metadata: Metadata to attach to trace (trace-level)
        usage: Usage data in Anthropic format (input_tokens, output_tokens, cache_read_input_tokens)
    """
    if not generation:
        return
    try:
        # Update generation with output and usage_details
        # Call methods directly on the generation object
        update_params = {}
        if output:
            update_params["output"] = output
        if usage:
            # Convert Anthropic usage format to LangFuse usage_details format
            update_params["usage_details"] = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            }

        if update_params:
            generation.update(**update_params)
            logger.debug(f"Updated generation with: {list(update_params.keys())}")

        # Update trace-level metadata separately
        if metadata:
            generation.update_trace(metadata=metadata)
            logger.debug(f"Updated trace metadata")
    except Exception as e:
        logger.warning(f"Failed to update LangFuse generation: {e}")


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


# Suppress OpenTelemetry context detach errors (they're harmless for our use case)
class SuppressOTelContextErrors(logging.Filter):
    """Filter to suppress OpenTelemetry context detach errors in async generators."""
    def filter(self, record):
        if record.name == "opentelemetry.context" and "Failed to detach context" in record.getMessage():
            return False
        if record.name == "opentelemetry.sdk.trace" and "Calling end() on an ended span" in record.getMessage():
            return False
        return True


# Apply the filter to OpenTelemetry loggers
logging.getLogger("opentelemetry.context").addFilter(SuppressOTelContextErrors())
logging.getLogger("opentelemetry.sdk.trace").addFilter(SuppressOTelContextErrors())


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
            # Use Langfuse() constructor with parameters, then get_client() can be used elsewhere
            langfuse_client = Langfuse(
                public_key=settings.LANGFUSE_API_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                base_url=settings.LANGFUSE_HOST,
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

        # Log thinking/reasoning conversion if present
        if "extra_body" in openrouter_request and "reasoning" in openrouter_request.get("extra_body", {}):
            logger.info(f"Converted thinking parameter to extra_body.reasoning for model: {openrouter_model}")
            _log_message("info", f"Converted thinking parameter to extra_body.reasoning for model: {openrouter_model}")

        # Log transformed OpenRouter request to messages.log
        _log_message("info", "-" * 80)
        if "tools" in openrouter_request:
            _log_message("info", f"Tools count: {len(openrouter_request['tools'])}")
            for idx, tool in enumerate(openrouter_request['tools']):
                _log_message("info", f"  Tool {idx + 1}: {tool.get('function', {}).get('name', 'unknown')}")
        _log_message("info", "-" * 80)

        # Check if streaming is requested (need to know before creating LangFuse generation)
        is_streaming = request_body.get("stream", False)

        # Create LangFuse generation AFTER transformation (captures the ACTUAL request sent to OpenRouter)
        generation = None
        generation_ctx = None
        if langfuse_client:
            try:
                if is_streaming:
                    # For streaming: Don't use context manager to avoid async context issues
                    # Just create generation and we'll update it manually
                    generation_ctx = langfuse_client.start_as_current_generation(
                        name="anthropic-proxy-request-stream",
                        model=anthropic_model,
                        input={
                            "model": anthropic_model,
                            "messages": request_body.get("messages"),
                            "system": request_body.get("system"),
                            "tools": request_body.get("tools"),
                            "max_tokens": request_body.get("max_tokens"),
                            "temperature": request_body.get("temperature"),
                            "stream": True,
                        }
                    )
                    # Enter context manually - we'll handle exit in wrapper
                    generation = generation_ctx.__enter__()
                    logger.debug("LangFuse generation created for streaming")
                else:
                    # For non-streaming: Use context manager normally
                    generation_ctx = langfuse_client.start_as_current_generation(
                        name="anthropic-proxy-request",
                        model=anthropic_model,
                        input={
                            "model": anthropic_model,
                            "messages": request_body.get("messages"),
                            "system": request_body.get("system"),
                            "tools": request_body.get("tools"),
                            "max_tokens": request_body.get("max_tokens"),
                            "temperature": request_body.get("temperature"),
                            "stream": False,
                        }
                    )
                    generation = generation_ctx.__enter__()
                    logger.debug("LangFuse generation created for non-streaming")
            except Exception as e:
                logger.warning(f"Failed to create LangFuse generation: {e}")
                generation = None
                generation_ctx = None

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
            # Choose streaming implementation based on config
            # For streaming, we need to keep the generation context active
            if settings.STREAMING_V2_ENABLED:
                logger.info("Using V2 streaming (with thinking/reasoning support)")
                async def streaming_wrapper():
                    """Wrapper to maintain LangFuse context during streaming."""
                    try:
                        async for chunk in stream_messages_v2(openrouter_request, anthropic_model, generation, priming_text_to_strip):
                            yield chunk
                    finally:
                        # Exit the context manager properly (don't call generation.end() separately)
                        if generation_ctx:
                            try:
                                generation_ctx.__exit__(None, None, None)
                            except ValueError as e:
                                # OpenTelemetry context detach error - safe to ignore
                                # This happens because async generators create different contexts
                                if "was created in a different Context" not in str(e):
                                    logger.debug(f"Context exit warning: {e}")
                            except Exception as e:
                                logger.debug(f"Context exit warning: {e}")
                        if langfuse_client:
                            try:
                                langfuse_client.flush()
                            except Exception as e:
                                logger.debug(f"Flush warning: {e}")

                return StreamingResponse(
                    streaming_wrapper(),
                    media_type="text/event-stream",
                )
            else:
                logger.info("Using V1 streaming (legacy)")
                async def streaming_wrapper_v1():
                    """Wrapper to maintain LangFuse context during streaming."""
                    try:
                        async for chunk in stream_messages(openrouter_request, anthropic_model, generation, initial_assistant_content, priming_text_to_strip):
                            yield chunk
                    finally:
                        # Exit the context manager properly (don't call generation.end() separately)
                        if generation_ctx:
                            try:
                                generation_ctx.__exit__(None, None, None)
                            except ValueError as e:
                                # OpenTelemetry context detach error - safe to ignore
                                # This happens because async generators create different contexts
                                if "was created in a different Context" not in str(e):
                                    logger.debug(f"Context exit warning: {e}")
                            except Exception as e:
                                logger.debug(f"Context exit warning: {e}")
                        if langfuse_client:
                            try:
                                langfuse_client.flush()
                            except Exception as e:
                                logger.debug(f"Flush warning: {e}")

                return StreamingResponse(
                    streaming_wrapper_v1(),
                    media_type="text/event-stream",
                )
        else:
            return await handle_non_streaming_messages(
                openrouter_request,
                anthropic_model,
                request_body,
                generation,
            )

    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        _log_message("error", "ERROR: Invalid JSON in request body")
        _update_generation_safe(generation, output=None, metadata={"status": "error", "error": "Invalid JSON"})
        if generation:
            generation.end()
        if langfuse_client:
            langfuse_client.flush()
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        _log_message("error", f"ERROR: {str(e)}")
        _update_generation_safe(generation, output=None, metadata={"status": "error", "error": str(e)})
        if generation:
            generation.end()
        if langfuse_client:
            langfuse_client.flush()
        raise HTTPException(status_code=500, detail=str(e))


async def handle_non_streaming_messages(
    openrouter_request: Dict[str, Any],
    anthropic_model: str,
    anthropic_request: Optional[Dict[str, Any]] = None,
    generation: Optional[Any] = None,
) -> Response:
    """
    Handle non-streaming message requests.

    Args:
        openrouter_request: Transformed request for OpenRouter
        anthropic_model: Original Anthropic model name
        anthropic_request: Original Anthropic request (for logging system prompt)
        generation: LangFuse generation object (for logging)
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

            # Update LangFuse generation with output and usage if it was created
            _update_generation_safe(
                generation,
                output={
                    "model": anthropic_response.get("model"),
                    "content": anthropic_response.get("content", []),
                    "stop_reason": anthropic_response.get("stop_reason"),
                },
                usage=anthropic_response.get("usage", {}),  # Pass usage separately for token tracking
                metadata={
                    "status": "success",
                    "openrouter_model": settings.get_openrouter_model(anthropic_model),
                }
            )

            # Finalize generation
            if generation:
                generation.end()
            if langfuse_client:
                langfuse_client.flush()

            return Response(
                content=json.dumps(anthropic_response),
                media_type="application/json",
            )

        except httpx.TimeoutException:
            logger.error("Request to OpenRouter timed out")
            _update_generation_safe(generation, output=None, metadata={"status": "timeout"})
            if generation:
                generation.end()
            if langfuse_client:
                langfuse_client.flush()
            raise HTTPException(status_code=504, detail="Request to OpenRouter timed out")
        except httpx.RequestError as e:
            logger.error(f"Error connecting to OpenRouter: {e}")
            _update_generation_safe(generation, output=None, metadata={"status": "error", "error": str(e)})
            if generation:
                generation.end()
            if langfuse_client:
                langfuse_client.flush()
            raise HTTPException(status_code=502, detail="Error connecting to OpenRouter")


async def stream_messages(
    openrouter_request: Dict[str, Any],
    anthropic_model: str,
    generation: Optional[Any] = None,
    initial_assistant_content: str = "",
    priming_text_to_strip: str = "",
) -> AsyncGenerator[str, None]:
    """
    Stream messages from OpenRouter and convert to Anthropic format.

    Args:
        openrouter_request: Transformed request for OpenRouter
        anthropic_model: Original Anthropic model name
        generation: LangFuse generation object (for logging)
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

    # Accumulate complete response for LangFuse (text + tool calls)
    accumulated_tool_calls = {}  # {index: {"id": "...", "name": "...", "arguments": "..."}}

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
                            break
                        except json.JSONDecodeError as e:
                            logger.debug(f"Failed to parse first chunk: {e}")
                            continue

                # Process the first chunk we peeked at
                if first_chunk:
                    logger.debug(f"Processing peeked first chunk: {json.dumps(first_chunk)}")

                    # Log raw OpenRouter chunk to openrouter-streaming.log (one line, no formatting)
                    if openrouter_streaming_logger:
                        openrouter_streaming_logger.info(json.dumps(first_chunk))

                    # Log raw OpenRouter chunk to streaming.log
                    _log_streaming("info", "=" * 80)
                    _log_streaming("info", "OPENROUTER CHUNK (FIRST)")
                    _log_streaming("info", json.dumps(first_chunk, indent=2))
                    _log_streaming("info", "=" * 80)

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

                        # Log raw Anthropic SSE event to anthropic-streaming.log
                        if anthropic_streaming_logger:
                            anthropic_streaming_logger.info(anthropic_event.rstrip())

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
                                    # Log raw Anthropic SSE event to anthropic-streaming.log
                                    if anthropic_streaming_logger:
                                        anthropic_streaming_logger.info(anthropic_event.rstrip())

                                    yield anthropic_event
                                    events_received += 1
                            break

                        try:
                            openrouter_chunk = json.loads(chunk_str)
                            logger.debug(f"OpenRouter chunk: {json.dumps(openrouter_chunk)}")

                            # Log raw OpenRouter chunk to openrouter-streaming.log (one line, no formatting)
                            if openrouter_streaming_logger:
                                openrouter_streaming_logger.info(json.dumps(openrouter_chunk))

                            # Log raw OpenRouter chunk to streaming.log
                            _log_streaming("info", "-" * 80)
                            _log_streaming("info", "OPENROUTER CHUNK")
                            _log_streaming("info", json.dumps(openrouter_chunk, indent=2))
                            _log_streaming("info", "-" * 80)

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
                                    # Log raw Anthropic SSE event to anthropic-streaming.log
                                    if anthropic_streaming_logger:
                                        anthropic_streaming_logger.info(anthropic_event.rstrip())

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
                                    # Log raw Anthropic SSE event to anthropic-streaming.log
                                    if anthropic_streaming_logger:
                                        anthropic_streaming_logger.info(anthropic_event.rstrip())

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

                                # Log raw Anthropic SSE event to anthropic-streaming.log
                                if anthropic_streaming_logger:
                                    anthropic_streaming_logger.info(anthropic_event.rstrip())

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

                # Log stream summary to streaming.log
                _log_streaming("info", "=" * 80)
                _log_streaming("info", "STREAMING RESPONSE SUMMARY")
                _log_streaming("info", f"Total Events Received: {events_received}")
                _log_streaming("info", f"Accumulated Text Length: {len(accumulated_text)}")
                _log_streaming("info", f"Accumulated Tool Calls: {len(accumulated_tool_calls)}")
                if accumulated_tool_calls:
                    _log_streaming("info", "Tool Calls Details:")
                    for idx, tool_call in accumulated_tool_calls.items():
                        _log_streaming("info", f"  [{idx}] {tool_call.get('name', 'unknown')}")
                        _log_streaming("info", f"      ID: {tool_call.get('id', 'N/A')}")
                        _log_streaming("info", f"      Args: {tool_call.get('arguments', '')[:200]}...")
                if final_usage:
                    _log_streaming("info", f"Final Usage: {json.dumps(final_usage, indent=2)}")
                _log_streaming("info", "=" * 80)

                logger.debug(f"Updating generation with output: events={events_received}, text length={len(accumulated_text)}, usage={final_usage}")

                # Transform usage from OpenRouter format to Anthropic format
                anthropic_usage = {}
                if final_usage:
                    prompt_tokens_details = final_usage.get("prompt_tokens_details") or {}
                    anthropic_usage = {
                        "input_tokens": final_usage.get("prompt_tokens", 0),
                        "output_tokens": final_usage.get("completion_tokens", 0),
                        "cache_read_input_tokens": prompt_tokens_details.get("cached_tokens", 0),
                        "cache_creation_input_tokens": prompt_tokens_details.get("cache_creation_input_tokens", 0),
                    }

                _update_generation_safe(
                    generation,
                    output={
                        "model": anthropic_model,
                        "content": [{"type": "text", "text": accumulated_text}],  # Match non-streaming format
                        "stop_reason": "end_turn",  # Streaming completed successfully
                    },
                    usage=anthropic_usage,
                    metadata={
                        "status": "streaming_completed",
                        "events_count": events_received,
                        "text_length": len(accumulated_text),
                    }
                )
                # Generation will be finalized by the wrapper
                # Don't call end() here to avoid "ended span" warning
                logger.info(f"Streaming V1 completed with {len(accumulated_text)} chars")

        except httpx.TimeoutException:
            logger.error("Request to OpenRouter timed out during streaming")

            # Transform usage for error case
            error_usage = {}
            if final_usage:
                prompt_tokens_details = final_usage.get("prompt_tokens_details") or {}
                error_usage = {
                    "input_tokens": final_usage.get("prompt_tokens", 0),
                    "output_tokens": final_usage.get("completion_tokens", 0),
                    "cache_read_input_tokens": prompt_tokens_details.get("cached_tokens", 0),
                    "cache_creation_input_tokens": prompt_tokens_details.get("cache_creation_input_tokens", 0),
                }

            _update_generation_safe(
                generation,
                output={
                    "model": anthropic_model,
                    "content": [{"type": "text", "text": accumulated_text}] if accumulated_text else [],
                    "stop_reason": "timeout",
                },
                usage=error_usage,
                metadata={"status": "timeout", "text_partial": len(accumulated_text) > 0}
            )
        except httpx.RequestError as e:
            logger.error(f"Error connecting to OpenRouter: {e}")

            # Transform usage for error case
            error_usage = {}
            if final_usage:
                prompt_tokens_details = final_usage.get("prompt_tokens_details") or {}
                error_usage = {
                    "input_tokens": final_usage.get("prompt_tokens", 0),
                    "output_tokens": final_usage.get("completion_tokens", 0),
                    "cache_read_input_tokens": prompt_tokens_details.get("cached_tokens", 0),
                    "cache_creation_input_tokens": prompt_tokens_details.get("cache_creation_input_tokens", 0),
                }

            _update_generation_safe(
                generation,
                output={
                    "model": anthropic_model,
                    "content": [{"type": "text", "text": accumulated_text}] if accumulated_text else [],
                    "stop_reason": "error",
                },
                usage=error_usage,
                metadata={"status": "connection_error", "error": str(e), "text_partial": len(accumulated_text) > 0}
            )
        except Exception as e:
            logger.error(f"Unexpected error during streaming: {e}", exc_info=True)

            # Transform usage for error case
            error_usage = {}
            if final_usage:
                prompt_tokens_details = final_usage.get("prompt_tokens_details") or {}
                error_usage = {
                    "input_tokens": final_usage.get("prompt_tokens", 0),
                    "output_tokens": final_usage.get("completion_tokens", 0),
                    "cache_read_input_tokens": prompt_tokens_details.get("cached_tokens", 0),
                    "cache_creation_input_tokens": prompt_tokens_details.get("cache_creation_input_tokens", 0),
                }

            _update_generation_safe(
                generation,
                output={
                    "model": anthropic_model,
                    "content": [{"type": "text", "text": accumulated_text}] if accumulated_text else [],
                    "stop_reason": "error",
                },
                usage=error_usage,
                metadata={"status": "error", "error": str(e), "text_partial": len(accumulated_text) > 0}
            )


async def stream_messages_v2(
    openrouter_request: Dict[str, Any],
    anthropic_model: str,
    generation: Optional[Any] = None,
    priming_text_to_strip: str = "",
) -> AsyncGenerator[str, None]:
    """
    V2: Stream messages with proper Anthropic SSE event structure.

    Key improvements over V1:
    - Always sends message_start first (before any content)
    - Handles Kimi reasoning → thinking blocks
    - Better tool call handling
    - Cleaner state management

    Args:
        openrouter_request: Transformed request for OpenRouter
        anthropic_model: Original Anthropic model name
        generation: LangFuse generation object (for logging)
        priming_text_to_strip: Text to strip from beginning of response
    """
    # Stream metadata
    events_received = 0
    final_usage = None
    stream_error = None
    first_content_received = False
    accumulated_text = ""
    accumulated_reasoning = ""
    actual_stop_reason = "end_turn"  # Will be updated from finish chunk

    # Usage waiting state (OpenRouter sends usage AFTER finish_reason)
    finish_reason_seen = None
    waiting_for_usage = False

    # Initialize content blocks state for v2 transformer
    content_blocks_state = {
        "next_index": 0,  # Next available Anthropic content block index
        "thinking": {},   # Will be populated when reasoning appears
        "text": {},       # Will be populated when content appears
        "tool_calls": {}  # Will be populated when tool calls appear
    }

    if priming_text_to_strip:
        logger.info(f"Will strip priming text from response: {repr(priming_text_to_strip)}")
        _log_message("info", f"Will strip priming text: {repr(priming_text_to_strip)}")

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

                # STEP 1: Send message_start IMMEDIATELY (before processing any chunks)
                message_id = ResponseTransformer._generate_message_id()
                message_start_event = {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": anthropic_model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                }
                message_start_sse = f"event: message_start\ndata: {json.dumps(message_start_event)}\n\n"

                # Log raw Anthropic SSE event to anthropic-streaming.log
                if anthropic_streaming_logger:
                    anthropic_streaming_logger.info(message_start_sse.rstrip())

                yield message_start_sse
                _log_stream_event(message_start_sse)
                events_received += 1

                # STEP 2: Stream and transform chunks
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        chunk_str = line[6:]  # Remove "data: " prefix
                        if chunk_str == "[DONE]":
                            break

                        try:
                            openrouter_chunk = json.loads(chunk_str)
                            logger.debug(f"OpenRouter chunk: {json.dumps(openrouter_chunk)}")

                            # Log raw OpenRouter chunk to openrouter-streaming.log (one line, no formatting)
                            if openrouter_streaming_logger:
                                openrouter_streaming_logger.info(json.dumps(openrouter_chunk))

                            # Log raw chunk to streaming.log
                            _log_streaming("info", "-" * 80)
                            _log_streaming("info", "OPENROUTER CHUNK")
                            _log_streaming("info", json.dumps(openrouter_chunk, indent=2))
                            _log_streaming("info", "-" * 80)

                            # Collect usage from chunks
                            if "usage" in openrouter_chunk:
                                final_usage = openrouter_chunk["usage"]
                                logger.debug(f"Collected usage: {final_usage}")

                            # USAGE FIX: Handle finish_reason arriving before usage
                            # OpenRouter sends: chunk1={finish_reason=stop}, chunk2={usage=...}
                            # We need to wait for usage before sending finish events
                            if "choices" in openrouter_chunk:
                                choices = openrouter_chunk.get("choices", [])
                                if choices and choices[0].get("finish_reason"):
                                    finish_reason = choices[0]["finish_reason"]
                                    actual_stop_reason = ResponseTransformer._map_finish_reason(finish_reason)

                                    # Save finish_reason and wait for usage
                                    finish_reason_seen = finish_reason
                                    waiting_for_usage = True
                                    logger.debug(f"Finish reason detected: {finish_reason} -> {actual_stop_reason}. Waiting for usage...")

                                    # Remove finish_reason from chunk so transformer doesn't process it yet
                                    openrouter_chunk["choices"][0].pop("finish_reason", None)

                            # If we're waiting for usage and just received it, add finish_reason back
                            if waiting_for_usage and final_usage:
                                logger.debug(f"Usage received while waiting. Adding finish_reason back: {finish_reason_seen}")
                                if "choices" in openrouter_chunk and openrouter_chunk["choices"]:
                                    openrouter_chunk["choices"][0]["finish_reason"] = finish_reason_seen
                                    waiting_for_usage = False

                            # Strip priming text from first content chunk
                            if "choices" in openrouter_chunk:
                                choices = openrouter_chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})

                                    # Handle text content stripping
                                    if "content" in delta and delta["content"]:
                                        content = delta["content"]
                                        if priming_text_to_strip and not first_content_received:
                                            if content.startswith(priming_text_to_strip):
                                                content = content[len(priming_text_to_strip):]
                                                logger.info(f"Stripped priming text: {repr(priming_text_to_strip)}")
                                                _log_message("info", f"Stripped priming: {repr(priming_text_to_strip)}")
                                                # Modify chunk before transformation
                                                openrouter_chunk["choices"][0]["delta"]["content"] = content
                                            first_content_received = True

                                        accumulated_text += content

                                    # Accumulate reasoning for logging
                                    if "reasoning" in delta and delta["reasoning"]:
                                        accumulated_reasoning += delta["reasoning"]

                            # Transform chunk using V2 transformer
                            anthropic_event = ResponseTransformer.transform_streaming_chunk_v2(
                                openrouter_chunk,
                                anthropic_model,
                                content_blocks_state,
                                accumulated_usage=final_usage,
                            )

                            if anthropic_event:
                                # Log raw Anthropic SSE event to anthropic-streaming.log
                                if anthropic_streaming_logger:
                                    # Log each event (may contain multiple events in one string)
                                    anthropic_streaming_logger.info(anthropic_event.rstrip())

                                # Log event to messages.log
                                _log_stream_event(anthropic_event)

                                yield anthropic_event
                                events_received += 1

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse chunk: {e}")
                            stream_error = str(e)
                            continue

                # EDGE CASE: Stream ended while still waiting for usage
                # Send finish events with whatever usage we have
                if waiting_for_usage:
                    logger.warning(f"Stream ended while waiting for usage. Sending finish events with available usage.")

                    # Create synthetic finish chunk with finish_reason and available usage
                    synthetic_chunk = {
                        "choices": [{"finish_reason": finish_reason_seen, "delta": {}}],
                        "usage": final_usage if final_usage else {}
                    }

                    # Transform and send finish events
                    anthropic_event = ResponseTransformer.transform_streaming_chunk_v2(
                        synthetic_chunk,
                        anthropic_model,
                        content_blocks_state,
                        accumulated_usage=final_usage,
                    )

                    if anthropic_event:
                        if anthropic_streaming_logger:
                            anthropic_streaming_logger.info(anthropic_event.rstrip())
                        _log_stream_event(anthropic_event)
                        yield anthropic_event
                        events_received += 1

                    waiting_for_usage = False

                # Stream completed successfully
                logger.info(f"Stream completed: {events_received} events")

                # Log summary to messages.log
                _log_message("info", "=" * 80)
                _log_message("info", "STREAMING V2 RESPONSE COMPLETED")
                _log_message("info", f"Total Events: {events_received}")
                _log_message("info", f"Text Length: {len(accumulated_text)}")
                _log_message("info", f"Reasoning Length: {len(accumulated_reasoning)}")
                if final_usage:
                    _log_message("info", f"Final Usage: {final_usage}")
                _log_message("info", "=" * 80)

                # Log summary to streaming.log
                _log_streaming("info", "=" * 80)
                _log_streaming("info", "STREAMING V2 SUMMARY")
                _log_streaming("info", f"Total Events: {events_received}")
                _log_streaming("info", f"Text: {len(accumulated_text)} chars")
                _log_streaming("info", f"Reasoning: {len(accumulated_reasoning)} chars")
                _log_streaming("info", f"Tool Calls: {len(content_blocks_state.get('tool_calls', {}))}")
                if final_usage:
                    _log_streaming("info", f"Usage: {json.dumps(final_usage, indent=2)}")
                _log_streaming("info", "=" * 80)

                # Build complete content array for LangFuse (matches Anthropic API format)
                content = []

                # Add thinking block if reasoning was present
                if accumulated_reasoning:
                    content.append({
                        "type": "thinking",
                        "thinking": accumulated_reasoning
                    })

                # Add text block if content was present
                if accumulated_text:
                    content.append({
                        "type": "text",
                        "text": accumulated_text
                    })

                # Add tool_use blocks from content_blocks_state
                tool_calls_state = content_blocks_state.get("tool_calls", {})
                for tool_idx in sorted(tool_calls_state.keys()):
                    tool_state = tool_calls_state[tool_idx]
                    if tool_state.get("started"):
                        # Parse accumulated JSON arguments
                        try:
                            input_data = json.loads(tool_state.get("accumulated_arguments", "{}"))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse tool arguments: {e}")
                            input_data = {}

                        content.append({
                            "type": "tool_use",
                            "id": tool_state.get("id", ""),
                            "name": tool_state.get("name", ""),
                            "input": input_data
                        })

                # Transform usage from OpenRouter format to Anthropic format for LangFuse
                anthropic_usage = {}
                if final_usage:
                    prompt_tokens_details = final_usage.get("prompt_tokens_details") or {}
                    anthropic_usage = {
                        "input_tokens": final_usage.get("prompt_tokens", 0),
                        "output_tokens": final_usage.get("completion_tokens", 0),
                        "cache_read_input_tokens": prompt_tokens_details.get("cached_tokens", 0),
                        "cache_creation_input_tokens": prompt_tokens_details.get("cache_creation_input_tokens", 0),
                    }

                # Update LangFuse generation with complete content structure
                _update_generation_safe(
                    generation,
                    output={
                        "model": anthropic_model,
                        "content": content,  # Complete content array with all block types
                        "stop_reason": actual_stop_reason,  # Actual stop reason from finish chunk
                    },
                    usage=anthropic_usage,  # Anthropic-formatted usage for token tracking
                    metadata={
                        "status": "streaming_completed_v2",
                        "events_count": events_received,
                        "text_length": len(accumulated_text),
                        "reasoning_length": len(accumulated_reasoning),
                        "tool_calls_count": len(tool_calls_state),
                    }
                )

                # Generation will be finalized by the wrapper
                # Don't call end() here to avoid "ended span" warning

        except httpx.TimeoutException:
            logger.error("Request to OpenRouter timed out")

            # Build partial content array for error case
            error_content = []
            if accumulated_reasoning:
                error_content.append({"type": "thinking", "thinking": accumulated_reasoning})
            if accumulated_text:
                error_content.append({"type": "text", "text": accumulated_text})

            # Transform usage for LangFuse
            error_usage = {}
            if final_usage:
                prompt_tokens_details = final_usage.get("prompt_tokens_details") or {}
                error_usage = {
                    "input_tokens": final_usage.get("prompt_tokens", 0),
                    "output_tokens": final_usage.get("completion_tokens", 0),
                    "cache_read_input_tokens": prompt_tokens_details.get("cached_tokens", 0),
                    "cache_creation_input_tokens": prompt_tokens_details.get("cache_creation_input_tokens", 0),
                }

            _update_generation_safe(
                generation,
                output={
                    "model": anthropic_model,
                    "content": error_content,
                    "stop_reason": "timeout",
                },
                usage=error_usage,
                metadata={"status": "timeout", "version": "v2"}
            )
            # Generation will be finalized by the wrapper
            yield f"event: error\ndata: {json.dumps({'error': 'Request timed out'})}\n\n"
        except httpx.RequestError as e:
            logger.error(f"Error connecting to OpenRouter: {e}")

            # Build partial content array for error case
            error_content = []
            if accumulated_reasoning:
                error_content.append({"type": "thinking", "thinking": accumulated_reasoning})
            if accumulated_text:
                error_content.append({"type": "text", "text": accumulated_text})

            # Transform usage for LangFuse
            error_usage = {}
            if final_usage:
                prompt_tokens_details = final_usage.get("prompt_tokens_details") or {}
                error_usage = {
                    "input_tokens": final_usage.get("prompt_tokens", 0),
                    "output_tokens": final_usage.get("completion_tokens", 0),
                    "cache_read_input_tokens": prompt_tokens_details.get("cached_tokens", 0),
                    "cache_creation_input_tokens": prompt_tokens_details.get("cache_creation_input_tokens", 0),
                }

            _update_generation_safe(
                generation,
                output={
                    "model": anthropic_model,
                    "content": error_content,
                    "stop_reason": "error",
                },
                usage=error_usage,
                metadata={"status": "connection_error", "error": str(e), "version": "v2"}
            )
            # Generation will be finalized by the wrapper
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"Unexpected error during streaming: {e}", exc_info=True)

            # Build partial content array for error case
            error_content = []
            if accumulated_reasoning:
                error_content.append({"type": "thinking", "thinking": accumulated_reasoning})
            if accumulated_text:
                error_content.append({"type": "text", "text": accumulated_text})

            # Transform usage for LangFuse
            error_usage = {}
            if final_usage:
                prompt_tokens_details = final_usage.get("prompt_tokens_details") or {}
                error_usage = {
                    "input_tokens": final_usage.get("prompt_tokens", 0),
                    "output_tokens": final_usage.get("completion_tokens", 0),
                    "cache_read_input_tokens": prompt_tokens_details.get("cached_tokens", 0),
                    "cache_creation_input_tokens": prompt_tokens_details.get("cache_creation_input_tokens", 0),
                }

            _update_generation_safe(
                generation,
                output={
                    "model": anthropic_model,
                    "content": error_content,
                    "stop_reason": "error",
                },
                usage=error_usage,
                metadata={"status": "error", "error": str(e), "version": "v2"}
            )
            # Generation will be finalized by the wrapper
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
            "count_tokens": "POST /v1/messages/count_tokens",
            "complete": "POST /v1/complete (deprecated)",
            "files_upload": "POST /v1/files",
            "files_list": "GET /v1/files",
            "files_get": "GET /v1/files/{file_id}",
            "files_delete": "DELETE /v1/files/{file_id}",
            "files_content": "GET /v1/files/{file_id}/content",
            "batches_create": "POST /v1/messages/batches",
            "batches_list": "GET /v1/messages/batches",
            "batches_get": "GET /v1/messages/batches/{batch_id}",
            "batches_cancel": "POST /v1/messages/batches/{batch_id}/cancel",
            "batches_results": "GET /v1/messages/batches/{batch_id}/results",
        },
        "note": "Most endpoints are dummy implementations for compatibility. Only /v1/messages, /v1/models, and /v1/models/{model_id} are fully functional.",
    }


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> Response:
    """
    Dummy endpoint for counting tokens in a message.
    Returns estimated token count based on character count.
    """
    count_tokens_logger = _setup_endpoint_logger("count_tokens")

    try:
        request_body = await request.json()

        # Log request
        _log_endpoint_request(count_tokens_logger, "count_tokens", request_body)

        # Estimate token count (rough approximation: ~4 chars per token)
        messages = request_body.get("messages", [])
        system = request_body.get("system", "")

        total_chars = len(system)
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_chars += len(item.get("text", ""))

        estimated_tokens = total_chars // 4

        response_data = {
            "input_tokens": estimated_tokens
        }

        if count_tokens_logger:
            count_tokens_logger.info(f"Response: {json.dumps(response_data, indent=2)}")

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Error in count_tokens: {e}", exc_info=True)
        if count_tokens_logger:
            count_tokens_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/complete")
async def complete(request: Request) -> Response:
    """
    Dummy endpoint for legacy text completion (deprecated).
    Returns a placeholder response.
    """
    complete_logger = _setup_endpoint_logger("complete")

    try:
        request_body = await request.json()

        # Log request
        _log_endpoint_request(complete_logger, "complete", request_body)

        response_data = {
            "type": "completion",
            "id": f"compl_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "completion": "This is a dummy response. The /v1/complete endpoint is deprecated.",
            "stop_reason": "end_turn",
            "model": request_body.get("model", "claude-2"),
        }

        if complete_logger:
            complete_logger.info(f"Response: {json.dumps(response_data, indent=2)}")

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Error in complete: {e}", exc_info=True)
        if complete_logger:
            complete_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Files API endpoints
@app.post("/v1/files")
async def upload_file(request: Request) -> Response:
    """
    Dummy endpoint for file upload.
    Returns a placeholder file ID.
    """
    files_logger = _setup_endpoint_logger("files")

    try:
        # For multipart/form-data, we can't directly parse as JSON
        content_type = request.headers.get("content-type", "")

        request_info = {
            "content_type": content_type,
            "headers": dict(request.headers),
        }

        # Log request
        _log_endpoint_request(files_logger, "files_upload", request_info)

        file_id = f"file_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        response_data = {
            "type": "file",
            "id": file_id,
            "filename": "dummy_file.txt",
            "purpose": "user_data",
            "created_at": datetime.now().isoformat(),
            "status": "processed",
        }

        if files_logger:
            files_logger.info(f"Response: {json.dumps(response_data, indent=2)}")

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Error in upload_file: {e}", exc_info=True)
        if files_logger:
            files_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/files")
async def list_files(request: Request) -> Response:
    """
    Dummy endpoint for listing files.
    Returns empty list.
    """
    files_logger = _setup_endpoint_logger("files")

    try:
        request_info = {
            "method": "GET",
            "query_params": dict(request.query_params),
        }

        # Log request
        _log_endpoint_request(files_logger, "files_list", request_info)

        response_data = {
            "data": [],
            "has_more": False,
        }

        if files_logger:
            files_logger.info(f"Response: {json.dumps(response_data, indent=2)}")

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Error in list_files: {e}", exc_info=True)
        if files_logger:
            files_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/files/{file_id}")
async def get_file(file_id: str, request: Request) -> Response:
    """
    Dummy endpoint for getting file metadata.
    Returns placeholder file info.
    """
    files_logger = _setup_endpoint_logger("files")

    try:
        request_info = {
            "method": "GET",
            "file_id": file_id,
        }

        # Log request
        _log_endpoint_request(files_logger, "files_get", request_info)

        response_data = {
            "type": "file",
            "id": file_id,
            "filename": "dummy_file.txt",
            "purpose": "user_data",
            "created_at": datetime.now().isoformat(),
            "status": "processed",
        }

        if files_logger:
            files_logger.info(f"Response: {json.dumps(response_data, indent=2)}")

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Error in get_file: {e}", exc_info=True)
        if files_logger:
            files_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str, request: Request) -> Response:
    """
    Dummy endpoint for deleting a file.
    Returns success response.
    """
    files_logger = _setup_endpoint_logger("files")

    try:
        request_info = {
            "method": "DELETE",
            "file_id": file_id,
        }

        # Log request
        _log_endpoint_request(files_logger, "files_delete", request_info)

        response_data = {
            "type": "file",
            "id": file_id,
            "deleted": True,
        }

        if files_logger:
            files_logger.info(f"Response: {json.dumps(response_data, indent=2)}")

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Error in delete_file: {e}", exc_info=True)
        if files_logger:
            files_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str, request: Request) -> Response:
    """
    Dummy endpoint for downloading file content.
    Returns placeholder text.
    """
    files_logger = _setup_endpoint_logger("files")

    try:
        request_info = {
            "method": "GET",
            "file_id": file_id,
        }

        # Log request
        _log_endpoint_request(files_logger, "files_content", request_info)

        content = "This is dummy file content."

        if files_logger:
            files_logger.info(f"Response: Returning dummy file content ({len(content)} bytes)")

        return Response(
            content=content,
            media_type="text/plain",
        )

    except Exception as e:
        logger.error(f"Error in get_file_content: {e}", exc_info=True)
        if files_logger:
            files_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Batches API endpoints
@app.post("/v1/messages/batches")
async def create_batch(request: Request) -> Response:
    """
    Dummy endpoint for creating a message batch.
    Returns placeholder batch ID.
    """
    batches_logger = _setup_endpoint_logger("batches")

    try:
        request_body = await request.json()

        # Log request
        _log_endpoint_request(batches_logger, "batches_create", request_body)

        batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        response_data = {
            "type": "message_batch",
            "id": batch_id,
            "processing_status": "in_progress",
            "request_counts": {
                "processing": len(request_body.get("requests", [])),
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
            "created_at": datetime.now().isoformat(),
        }

        if batches_logger:
            batches_logger.info(f"Response: {json.dumps(response_data, indent=2)}")

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Error in create_batch: {e}", exc_info=True)
        if batches_logger:
            batches_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/messages/batches")
async def list_batches(request: Request) -> Response:
    """
    Dummy endpoint for listing batches.
    Returns empty list.
    """
    batches_logger = _setup_endpoint_logger("batches")

    try:
        request_info = {
            "method": "GET",
            "query_params": dict(request.query_params),
        }

        # Log request
        _log_endpoint_request(batches_logger, "batches_list", request_info)

        response_data = {
            "data": [],
            "has_more": False,
        }

        if batches_logger:
            batches_logger.info(f"Response: {json.dumps(response_data, indent=2)}")

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Error in list_batches: {e}", exc_info=True)
        if batches_logger:
            batches_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/messages/batches/{batch_id}")
async def get_batch(batch_id: str, request: Request) -> Response:
    """
    Dummy endpoint for getting batch status.
    Returns placeholder batch info.
    """
    batches_logger = _setup_endpoint_logger("batches")

    try:
        request_info = {
            "method": "GET",
            "batch_id": batch_id,
        }

        # Log request
        _log_endpoint_request(batches_logger, "batches_get", request_info)

        response_data = {
            "type": "message_batch",
            "id": batch_id,
            "processing_status": "ended",
            "request_counts": {
                "processing": 0,
                "succeeded": 10,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
            "created_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
        }

        if batches_logger:
            batches_logger.info(f"Response: {json.dumps(response_data, indent=2)}")

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Error in get_batch: {e}", exc_info=True)
        if batches_logger:
            batches_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str, request: Request) -> Response:
    """
    Dummy endpoint for canceling a batch.
    Returns success response.
    """
    batches_logger = _setup_endpoint_logger("batches")

    try:
        request_info = {
            "method": "POST",
            "batch_id": batch_id,
        }

        # Log request
        _log_endpoint_request(batches_logger, "batches_cancel", request_info)

        response_data = {
            "type": "message_batch",
            "id": batch_id,
            "processing_status": "canceling",
        }

        if batches_logger:
            batches_logger.info(f"Response: {json.dumps(response_data, indent=2)}")

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    except Exception as e:
        logger.error(f"Error in cancel_batch: {e}", exc_info=True)
        if batches_logger:
            batches_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/messages/batches/{batch_id}/results")
async def get_batch_results(batch_id: str, request: Request) -> Response:
    """
    Dummy endpoint for getting batch results.
    Returns empty JSONL stream.
    """
    batches_logger = _setup_endpoint_logger("batches")

    try:
        request_info = {
            "method": "GET",
            "batch_id": batch_id,
        }

        # Log request
        _log_endpoint_request(batches_logger, "batches_results", request_info)

        # Return empty JSONL content
        content = ""

        if batches_logger:
            batches_logger.info(f"Response: Empty JSONL results")

        return Response(
            content=content,
            media_type="application/jsonl",
        )

    except Exception as e:
        logger.error(f"Error in get_batch_results: {e}", exc_info=True)
        if batches_logger:
            batches_logger.error(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
