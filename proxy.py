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

# Setup file logging for /messages endpoint
messages_log_dir = "logs"
if not os.path.exists(messages_log_dir):
    os.makedirs(messages_log_dir)

messages_log_file = os.path.join(messages_log_dir, "messages.log")
messages_logger = logging.getLogger("messages")
messages_logger.setLevel(logging.DEBUG)
messages_logger.propagate = False  # Prevent propagation to parent loggers

# Clear any existing handlers to avoid duplicates
messages_logger.handlers.clear()

# File handler for messages
file_handler = logging.FileHandler(messages_log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
messages_logger.addHandler(file_handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('[MESSAGES] %(message)s'))
# messages_logger.addHandler(console_handler)

logger.info(f"Messages endpoint logging to: {os.path.abspath(messages_log_file)}")

# Initialize FastAPI app
app = FastAPI(
    title="Anthropic-to-OpenRouter Proxy",
    description="Proxy that converts Anthropic API requests to OpenRouter format",
    version="1.0.0",
)

# LangFuse integration (optional)
langfuse_client = None
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

# Validate configuration on startup
try:
    settings.validate()
    logger.info("Configuration validated successfully")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info(f"Starting proxy on {settings.PROXY_HOST}:{settings.PROXY_PORT}")
    logger.info(f"OpenRouter base URL: {settings.OPENROUTER_BASE_URL}")
    if settings.LANGFUSE_ENABLED:
        logger.info(f"LangFuse host: {settings.LANGFUSE_HOST}")


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
        messages_logger.info("=" * 80)
        messages_logger.info("INCOMING REQUEST")
        # messages_logger.info(f"{json.dumps(request_body, indent=2)}")
        messages_logger.info(f"Timestamp: {datetime.now().isoformat()}")

        # Extract model and convert to OpenRouter format
        anthropic_model = request_body.get("model", "")
        openrouter_model = settings.get_openrouter_model(anthropic_model)
        logger.info(f"Converting model {anthropic_model} -> {openrouter_model}")
        messages_logger.info(f"Model Conversion: {anthropic_model} -> {openrouter_model}")

        # Transform request to OpenRouter format
        openrouter_request = RequestTransformer.transform_messages_request(
            request_body,
            openrouter_model,
        )
        logger.debug(f"Transformed request: {json.dumps(openrouter_request, indent=2)}")

        # Log transformed OpenRouter request to messages.log
        messages_logger.info("-" * 80)
        messages_logger.info("TRANSFORMED OPENROUTER REQUEST")
        messages_logger.info(f"{json.dumps(openrouter_request, indent=2)}")
        if "tools" in openrouter_request:
            messages_logger.info(f"Tools count: {len(openrouter_request['tools'])}")
            for idx, tool in enumerate(openrouter_request['tools']):
                messages_logger.info(f"  Tool {idx + 1}: {tool.get('function', {}).get('name', 'unknown')}")
        messages_logger.info("-" * 80)

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

        messages_logger.info(f"Streaming: {is_streaming}")

        # Extract initial assistant content (priming text like "{" for JSON)
        initial_assistant_content = ""
        if is_streaming:
            messages = request_body.get("messages", [])
            if messages and messages[-1].get("role") == "assistant":
                # Last message is from assistant - extract its content for prepending
                last_msg = messages[-1].get("content", [])
                if isinstance(last_msg, list):
                    # Array format
                    for item in last_msg:
                        if isinstance(item, dict) and item.get("type") == "text":
                            initial_assistant_content += item.get("text", "")
                elif isinstance(last_msg, str):
                    # String format
                    initial_assistant_content = last_msg

        if is_streaming:
            return StreamingResponse(
                stream_messages(openrouter_request, anthropic_model, trace, initial_assistant_content),
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
        messages_logger.error("ERROR: Invalid JSON in request body")
        # Update trace with error
        if trace:
            try:
                trace.update(
                    output=None,
                    metadata={"status": "error", "error": "Invalid JSON"}
                )
            except Exception:
                pass
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        messages_logger.error(f"ERROR: {str(e)}", exc_info=True)
        # Update trace with error
        if trace:
            try:
                trace.update(
                    output=None,
                    metadata={"status": "error", "error": str(e)}
                )
            except Exception:
                pass
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
            if trace:
                try:
                    trace.update(
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
                    # Note: LangFuse client handles async batching, no need to flush here
                except Exception as e:
                    logger.warning(f"Failed to update LangFuse trace: {e}")

            return Response(
                content=json.dumps(anthropic_response),
                media_type="application/json",
            )

        except httpx.TimeoutException:
            logger.error("Request to OpenRouter timed out")
            if trace:
                try:
                    trace.update(
                        output=None,
                        metadata={"status": "timeout"}
                    )
                except Exception:
                    pass
            raise HTTPException(status_code=504, detail="Request to OpenRouter timed out")
        except httpx.RequestError as e:
            logger.error(f"Error connecting to OpenRouter: {e}")
            if trace:
                try:
                    trace.update(
                        output=None,
                        metadata={"status": "error", "error": str(e)}
                    )
                except Exception:
                    pass
            raise HTTPException(status_code=502, detail="Error connecting to OpenRouter")


async def stream_messages(
    openrouter_request: Dict[str, Any],
    anthropic_model: str,
    trace: Optional[Any] = None,
    initial_assistant_content: str = "",
) -> AsyncGenerator[str, None]:
    """
    Stream messages from OpenRouter and convert to Anthropic format.

    Args:
        openrouter_request: Transformed request for OpenRouter
        anthropic_model: Original Anthropic model name
        trace: LangFuse trace object (for logging)
        initial_assistant_content: Priming text from assistant message (e.g., "{" for JSON)
    """
    # Collect streaming metadata
    events_received = 0
    final_usage = None
    stream_error = None
    message_started = False  # Track if we've already sent message_start
    content_block_started = False  # Track if we've already sent content_block_start
    first_delta_received = False  # Track if first delta has been sent
    accumulated_text = initial_assistant_content  # Start with priming text
    if initial_assistant_content:
        messages_logger.info(f"[STREAM] Priming text: {repr(initial_assistant_content)}")

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

                # Peek at first chunk to decide whether to inject priming text
                first_chunk = None
                first_delta_content = None
                should_inject_priming = False

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
                            # Extract first delta content if available
                            if "choices" in first_chunk:
                                choices = first_chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    first_delta_content = delta.get("content", "")

                            # Decide whether to inject priming text
                            # Only inject if model's first delta doesn't start with the priming text
                            if initial_assistant_content and first_delta_content:
                                if not first_delta_content.startswith(initial_assistant_content):
                                    should_inject_priming = True
                                    messages_logger.info(f"[STREAM] Model's first delta doesn't start with priming text, will inject: {repr(initial_assistant_content)}")
                                else:
                                    messages_logger.info(f"[STREAM] Model's first delta already starts with priming text, skipping injection")
                            elif initial_assistant_content and not first_delta_content:
                                # No content in first delta, inject priming
                                should_inject_priming = True
                                messages_logger.info(f"[STREAM] First delta has no content, will inject priming text")

                            # Now that we've decided, inject synthetic events if needed
                            if should_inject_priming:
                                # Send message_start first
                                message_start_chunk = {
                                    "choices": [
                                        {
                                            "delta": {"role": "assistant"},
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                message_start_event = ResponseTransformer.transform_streaming_chunk(
                                    message_start_chunk,
                                    anthropic_model,
                                    skip_message_start=message_started,
                                    skip_content_block_start=content_block_started,
                                )
                                if message_start_event:
                                    yield message_start_event
                                    events_received += 1
                                    if "message_start" in message_start_event:
                                        message_started = True
                                    messages_logger.debug("[STREAM] Injected synthetic message_start")

                                # Send priming text as content_block_delta
                                priming_chunk = {
                                    "choices": [
                                        {
                                            "delta": {"content": initial_assistant_content},
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                priming_event = ResponseTransformer.transform_streaming_chunk(
                                    priming_chunk,
                                    anthropic_model,
                                    skip_message_start=True,
                                    skip_content_block_start=content_block_started,
                                )
                                if priming_event:
                                    yield priming_event
                                    events_received += 1
                                    accumulated_text += initial_assistant_content
                                    if "content_block_start" in priming_event:
                                        content_block_started = True
                                    if "content_block_delta" in priming_event:
                                        first_delta_received = True
                                    messages_logger.info(f"[STREAM] Injected priming text: {repr(initial_assistant_content)}")

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
                    if "choices" in first_chunk:
                        choices = first_chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                accumulated_text += delta["content"]

                    # Transform chunk to Anthropic format
                    anthropic_event = ResponseTransformer.transform_streaming_chunk(
                        first_chunk,
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
                        try:
                            data_match = anthropic_event.split('data: ')
                            if len(data_match) > 1:
                                event_data = json.loads(data_match[1].rstrip('\n\n'))
                                event_type = event_data.get('type', 'unknown')

                                if event_type == 'message_start':
                                    msg_id = event_data.get('message', {}).get('id')
                                    usage = event_data.get('message', {}).get('usage', {})
                                    messages_logger.info(f"[STREAM] message_start - ID: {msg_id}")
                                    messages_logger.debug(f"  Initial usage: input_tokens={usage.get('input_tokens', 0)}, output_tokens={usage.get('output_tokens', 0)}")

                                elif event_type == 'content_block_start':
                                    messages_logger.debug(f"[STREAM] content_block_start")

                                elif event_type == 'content_block_delta':
                                    text = event_data.get('delta', {}).get('text', '')
                                    if text:
                                        messages_logger.info(f"[STREAM] content_block_delta - Text: {text}")
                                    else:
                                        messages_logger.debug(f"[STREAM] content_block_delta (no text)")

                                elif event_type == 'content_block_stop':
                                    messages_logger.debug(f"[STREAM] content_block_stop")

                                elif event_type == 'message_delta':
                                    stop_reason = event_data.get('delta', {}).get('stop_reason')
                                    usage = event_data.get('usage', {})
                                    messages_logger.info(f"[STREAM] message_delta - Stop Reason: {stop_reason}")
                                    messages_logger.debug(f"  Final usage: input_tokens={usage.get('input_tokens', 0)}, output_tokens={usage.get('output_tokens', 0)}, cache_read={usage.get('cache_read_input_tokens', 0)}, cache_creation={usage.get('cache_creation_input_tokens', 0)}")

                                elif event_type == 'message_stop':
                                    messages_logger.debug(f"[STREAM] message_stop")

                                else:
                                    messages_logger.info(f"[STREAM] {event_type}")
                            else:
                                messages_logger.debug(f"[STREAM] Event: {anthropic_event}")
                        except Exception as log_error:
                            logger.debug(f"Error logging stream event: {log_error}")
                            messages_logger.debug(f"[STREAM] Event (raw): {anthropic_event}")

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
                            if "choices" in openrouter_chunk:
                                choices = openrouter_chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        accumulated_text += delta["content"]

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
                                # Extract event type and data from the SSE line
                                try:
                                    data_match = anthropic_event.split('data: ')
                                    if len(data_match) > 1:
                                        event_data = json.loads(data_match[1].rstrip('\n\n'))
                                        event_type = event_data.get('type', 'unknown')

                                        if event_type == 'message_start':
                                            msg_id = event_data.get('message', {}).get('id')
                                            usage = event_data.get('message', {}).get('usage', {})
                                            messages_logger.info(f"[STREAM] message_start - ID: {msg_id}")
                                            messages_logger.debug(f"  Initial usage: input_tokens={usage.get('input_tokens', 0)}, output_tokens={usage.get('output_tokens', 0)}")

                                        elif event_type == 'content_block_start':
                                            messages_logger.debug(f"[STREAM] content_block_start")

                                        elif event_type == 'content_block_delta':
                                            text = event_data.get('delta', {}).get('text', '')
                                            if text:
                                                messages_logger.info(f"[STREAM] content_block_delta - Text: {text}")
                                            else:
                                                messages_logger.debug(f"[STREAM] content_block_delta (no text)")

                                        elif event_type == 'content_block_stop':
                                            messages_logger.debug(f"[STREAM] content_block_stop")

                                        elif event_type == 'message_delta':
                                            stop_reason = event_data.get('delta', {}).get('stop_reason')
                                            usage = event_data.get('usage', {})
                                            messages_logger.info(f"[STREAM] message_delta - Stop Reason: {stop_reason}")
                                            messages_logger.debug(f"  Final usage: input_tokens={usage.get('input_tokens', 0)}, output_tokens={usage.get('output_tokens', 0)}, cache_read={usage.get('cache_read_input_tokens', 0)}, cache_creation={usage.get('cache_creation_input_tokens', 0)}")

                                        elif event_type == 'message_stop':
                                            messages_logger.debug(f"[STREAM] message_stop")

                                        else:
                                            messages_logger.info(f"[STREAM] {event_type}")
                                    else:
                                        messages_logger.debug(f"[STREAM] Event: {anthropic_event}")
                                except Exception as log_error:
                                    logger.debug(f"Error logging stream event: {log_error}")
                                    messages_logger.debug(f"[STREAM] Event (raw): {anthropic_event}")

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
                messages_logger.info("=" * 80)
                messages_logger.info("STREAMING RESPONSE COMPLETED")
                messages_logger.info(f"Total Events: {events_received}")
                messages_logger.info(f"Total Text Characters: {len(accumulated_text)}")
                if final_usage:
                    messages_logger.info(f"Final Usage: {final_usage}")
                messages_logger.info("=" * 80)

                if trace:
                    try:
                        logger.debug(f"Updating trace with output: events={events_received}, text length={len(accumulated_text)}, usage={final_usage}")
                        trace.update(
                            output={
                                "events_received": events_received,
                                "text": accumulated_text,
                                "usage": final_usage,
                            },
                            metadata={
                                "status": "streaming_completed",
                                "events_count": events_received,
                                "text_length": len(accumulated_text),
                            }
                        )
                        logger.info(f"Trace updated successfully for streaming request with {len(accumulated_text)} chars")
                        # Note: DO NOT call flush() here - it's blocking and will freeze streaming
                    except Exception as e:
                        logger.error(f"Failed to update LangFuse trace: {e}", exc_info=True)

        except httpx.TimeoutException:
            logger.error("Request to OpenRouter timed out during streaming")
            if trace:
                try:
                    trace.update(
                        output={
                            "events_received": events_received,
                            "text": accumulated_text,
                            "text_length": len(accumulated_text),
                        },
                        metadata={"status": "timeout", "text_partial": len(accumulated_text) > 0}
                    )
                except Exception:
                    pass
            yield f"event: error\ndata: {json.dumps({'error': 'Request timed out'})}\n\n"
        except httpx.RequestError as e:
            logger.error(f"Error connecting to OpenRouter: {e}")
            if trace:
                try:
                    trace.update(
                        output={
                            "events_received": events_received,
                            "text": accumulated_text,
                            "text_length": len(accumulated_text),
                        },
                        metadata={"status": "connection_error", "error": str(e), "text_partial": len(accumulated_text) > 0}
                    )
                except Exception:
                    pass
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"Unexpected error during streaming: {e}", exc_info=True)
            if trace:
                try:
                    trace.update(
                        output={
                            "events_received": events_received,
                            "text": accumulated_text,
                            "text_length": len(accumulated_text),
                        },
                        metadata={"status": "error", "error": str(e), "text_partial": len(accumulated_text) > 0}
                    )
                except Exception:
                    pass
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
