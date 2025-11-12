"""
Request and response transformers for converting between Anthropic and OpenRouter APIs.
"""
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


class RequestTransformer:
    """Transform Anthropic API requests to OpenRouter format."""

    @staticmethod
    def transform_messages_request(
        anthropic_request: Dict[str, Any],
        openrouter_model: str,
    ) -> Dict[str, Any]:
        """
        Convert Anthropic /v1/messages request to OpenRouter chat completion format.

        Args:
            anthropic_request: Original request from Anthropic client
            openrouter_model: OpenRouter model ID (e.g., 'anthropic/claude-3.5-sonnet')

        Returns:
            OpenRouter-compatible request body
        """
        # Convert Anthropic system parameter to OpenRouter system message
        # Anthropic uses a separate "system" parameter, but OpenRouter expects a message with role "system"
        messages = RequestTransformer._transform_messages(anthropic_request.get("messages", []))
        system = anthropic_request.get("system")

        if system:
            # Convert system to message format and insert as first message
            if isinstance(system, str):
                # Simple string system prompt
                system_message = {"role": "system", "content": system}
            elif isinstance(system, list):
                # Array format (e.g., with cache_control hints)
                system_message = {"role": "system", "content": system}
            else:
                # Fallback for other formats
                system_message = {"role": "system", "content": str(system)}

            # Insert system message at the beginning
            messages.insert(0, system_message)

        openrouter_request = {
            "model": openrouter_model,
            "messages": messages,
            "max_tokens": anthropic_request.get("max_tokens", 1024),
            "temperature": anthropic_request.get("temperature", 1.0),
            "top_p": anthropic_request.get("top_p", 1.0),
            "stream": anthropic_request.get("stream", False),
        }

        # Optional parameters - only include if specified
        if "top_k" in anthropic_request:
            openrouter_request["top_k"] = anthropic_request["top_k"]

        if "stop_sequences" in anthropic_request:
            openrouter_request["stop"] = anthropic_request["stop_sequences"]

        # Handle tools/function calling
        if "tools" in anthropic_request:
            openrouter_request["tools"] = RequestTransformer._transform_tools(
                anthropic_request["tools"]
            )

        # Add user ID for abuse prevention if available
        if "user" in anthropic_request:
            openrouter_request["user"] = anthropic_request["user"]

        # Enable usage accounting for OpenRouter (always include for better tracking)
        openrouter_request["usage"] = {"include": True}

        # Include usage in stream if requested
        if anthropic_request.get("stream"):
            openrouter_request["stream_options"] = {"include_usage": True}

        return openrouter_request

    @staticmethod
    def _transform_messages(anthropic_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform Anthropic messages to OpenRouter format.

        Handles:
        - Content blocks with tool_use and tool_result
        - Converting Anthropic's content array format to OpenRouter's format

        Args:
            anthropic_messages: List of Anthropic-format messages

        Returns:
            List of OpenRouter-format messages
        """
        openrouter_messages = []

        for msg in anthropic_messages:
            role = msg.get("role")
            content = msg.get("content")

            # Handle string content (simple case)
            if isinstance(content, str):
                openrouter_messages.append({
                    "role": role,
                    "content": content
                })
                continue

            # Handle array content (complex case with tool use/results)
            if isinstance(content, list):
                # Check if this message contains tool_result blocks
                has_tool_results = any(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in content
                )

                if has_tool_results:
                    # Transform each tool_result to a separate tool message
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_content = block.get("content", "")
                            # Handle content that might be a list or dict
                            if isinstance(tool_content, list):
                                # Extract text from content blocks
                                tool_content = " ".join(
                                    item.get("text", "") if isinstance(item, dict) else str(item)
                                    for item in tool_content
                                )
                            elif isinstance(tool_content, dict):
                                tool_content = json.dumps(tool_content)

                            openrouter_messages.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": str(tool_content)
                            })
                else:
                    # Check if this message contains tool_use blocks (assistant response)
                    has_tool_use = any(
                        isinstance(block, dict) and block.get("type") == "tool_use"
                        for block in content
                    )

                    if has_tool_use:
                        # Extract text content and tool calls separately
                        text_parts = []
                        tool_calls = []

                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                                elif block.get("type") == "tool_use":
                                    # Convert to OpenRouter tool_call format
                                    tool_calls.append({
                                        "id": block.get("id", str(uuid.uuid4())),
                                        "type": "function",
                                        "function": {
                                            "name": block.get("name", ""),
                                            "arguments": json.dumps(block.get("input", {}))
                                        }
                                    })

                        openrouter_msg = {
                            "role": role,
                            "content": " ".join(text_parts) if text_parts else None
                        }

                        if tool_calls:
                            openrouter_msg["tool_calls"] = tool_calls

                        openrouter_messages.append(openrouter_msg)
                    else:
                        # Regular content blocks (text, image, etc.)
                        # For now, just extract text blocks
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))

                        openrouter_messages.append({
                            "role": role,
                            "content": " ".join(text_parts) if text_parts else ""
                        })
            else:
                # Fallback for unexpected formats
                openrouter_messages.append({
                    "role": role,
                    "content": str(content)
                })

        return openrouter_messages

    @staticmethod
    def _transform_tools(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform Anthropic tool definitions to OpenAI format.

        Anthropic uses 'input_schema' while OpenAI uses 'parameters'.
        """
        openai_tools = []
        for tool in anthropic_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools


class ResponseTransformer:
    """Transform OpenRouter API responses to Anthropic format."""

    @staticmethod
    def transform_messages_response(
        openrouter_response: Dict[str, Any],
        original_model: str,
    ) -> Dict[str, Any]:
        """
        Convert OpenRouter chat completion response to Anthropic format.

        Args:
            openrouter_response: Response from OpenRouter API
            original_model: Original model name from client

        Returns:
            Anthropic-compatible response body
        """
        # Extract first choice
        choice = openrouter_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = openrouter_response.get("usage", {})

        # Map finish_reason
        finish_reason = choice.get("finish_reason", "unknown")
        stop_reason = ResponseTransformer._map_finish_reason(finish_reason)

        # Create content array
        content = []
        message_text = message.get("content", "")
        if message_text:
            content.append({"type": "text", "text": message_text})

        # Handle tool calls if present
        if "tool_calls" in message:
            content.extend(
                ResponseTransformer._transform_tool_calls(message["tool_calls"])
            )

        # Extract cache-related tokens if available
        prompt_tokens_details = usage.get("prompt_tokens_details") or {}
        completion_tokens_details = usage.get("completion_tokens_details") or {}

        anthropic_response = {
            "id": ResponseTransformer._generate_message_id(),
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": original_model,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_read_input_tokens": prompt_tokens_details.get("cached_tokens", 0),
                "cache_creation_input_tokens": prompt_tokens_details.get("cache_creation_input_tokens", 0),
            },
        }

        return anthropic_response

    @staticmethod
    def transform_streaming_chunk(
        openrouter_chunk: Dict[str, Any],
        original_model: str,
        skip_message_start: bool = False,
        skip_content_block_start: bool = False,
        accumulated_usage: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Convert OpenRouter streaming chunk to Anthropic SSE event format.

        Args:
            openrouter_chunk: Streaming chunk from OpenRouter
            original_model: Original model name
            skip_message_start: If True, skip message_start event (already sent)
            skip_content_block_start: If True, skip content_block_start event (already sent)
            accumulated_usage: Usage data collected from previous chunks (for finish events)

        Returns:
            Anthropic-style SSE event (string) or None if this is a keep-alive
        """
        # Skip keep-alive comments
        if "choices" not in openrouter_chunk:
            return None

        choices = openrouter_chunk.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # First chunk: message_start event (only if not already sent)
        if "role" in delta and not skip_message_start:
            event_data = {
                "type": "message_start",
                "message": {
                    "id": ResponseTransformer._generate_message_id(),
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": original_model,
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
            return f"event: message_start\ndata: {json.dumps(event_data)}\n\n"

        # Tool calls (OpenRouter sends these in delta)
        if "tool_calls" in delta and delta["tool_calls"]:
            output = ""

            # For each tool call in the delta
            for idx, tool_call in enumerate(delta["tool_calls"]):
                tool_index = tool_call.get("index", idx)

                # If this is a new tool call (has id), send content_block_start
                if "id" in tool_call:
                    content_block_event = {
                        "type": "content_block_start",
                        "index": tool_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call.get("function", {}).get("name", ""),
                            "input": {}
                        },
                    }
                    output += f"event: content_block_start\ndata: {json.dumps(content_block_event)}\n\n"

                # If there are arguments, send content_block_delta with input_json_delta
                if "function" in tool_call and "arguments" in tool_call["function"]:
                    delta_event = {
                        "type": "content_block_delta",
                        "index": tool_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tool_call["function"]["arguments"]
                        },
                    }
                    output += f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

            return output if output else None

        # Content chunks
        if "content" in delta and delta["content"]:
            output = ""

            # content_block_start (only on first content chunk)
            if not skip_content_block_start:
                content_block_event = {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text"},
                }
                output = f"event: content_block_start\ndata: {json.dumps(content_block_event)}\n\n"

            # content_block_delta
            delta_event = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": delta["content"]},
            }
            output += f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

            return output

        # End of stream
        if finish_reason is not None:
            stop_reason = ResponseTransformer._map_finish_reason(finish_reason)

            # content_block_stop
            content_stop_event = {"type": "content_block_stop", "index": 0}
            output = f"event: content_block_stop\ndata: {json.dumps(content_stop_event)}\n\n"

            # message_delta with final usage
            # Use accumulated usage if provided (from next chunk), otherwise fall back to current chunk
            usage = accumulated_usage if accumulated_usage else openrouter_chunk.get("usage", {})

            # Extract cache-related tokens if available
            prompt_tokens_details = usage.get("prompt_tokens_details") or {}
            completion_tokens_details = usage.get("completion_tokens_details") or {}

            message_delta_event = {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                },
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cache_read_input_tokens": prompt_tokens_details.get("cached_tokens", 0),
                    "cache_creation_input_tokens": prompt_tokens_details.get("cache_creation_input_tokens", 0),
                },
            }
            output += f"event: message_delta\ndata: {json.dumps(message_delta_event)}\n\n"

            # message_stop
            message_stop_event = {"type": "message_stop"}
            output += f"event: message_stop\ndata: {json.dumps(message_stop_event)}\n\n"

            return output

        return None

    @staticmethod
    def _map_finish_reason(openrouter_reason: str) -> str:
        """Map OpenRouter finish_reason to Anthropic stop_reason."""
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "end_turn",
            "error": "end_turn",
        }
        return mapping.get(openrouter_reason, "end_turn")

    @staticmethod
    def _transform_tool_calls(
        openai_tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Transform OpenAI tool calls to Anthropic tool use format."""
        tool_blocks = []
        for tool_call in openai_tool_calls:
            tool_block = {
                "type": "tool_use",
                "id": tool_call.get("id", str(uuid.uuid4())),
                "name": tool_call.get("function", {}).get("name", ""),
                "input": json.loads(
                    tool_call.get("function", {}).get("arguments", "{}")
                ),
            }
            tool_blocks.append(tool_block)
        return tool_blocks

    @staticmethod
    def _generate_message_id() -> str:
        """Generate an Anthropic-style message ID."""
        # Anthropic message IDs start with 'msg_' followed by random string
        import secrets
        random_str = secrets.token_hex(12)  # 24 character hex string
        return f"msg_{random_str}"


class ModelsResponseTransformer:
    """Transform models list responses."""

    @staticmethod
    def transform_models_response(openrouter_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transform OpenRouter models list to Anthropic format.

        Args:
            openrouter_models: List of models from OpenRouter

        Returns:
            Anthropic-compatible models response
        """
        anthropic_models = []

        for model in openrouter_models:
            # Only include Claude models for this proxy
            model_id = model.get("id", "")
            if not model_id.startswith("anthropic/"):
                continue

            # Extract Claude model name from 'anthropic/claude-3.5-sonnet'
            # For Anthropic, we use the full ID like 'claude-3-5-sonnet-20241022'
            claude_name = model_id.replace("anthropic/", "").replace(".", "-")

            anthropic_model = {
                "id": claude_name,  # Use simplified model ID
                "type": "model",
                "display_name": model.get("name", model_id),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "input_token_limit": model.get("architecture", {}).get(
                    "context_length", 128000
                ),
                "output_token_limit": model.get("top_provider", {}).get(
                    "max_completion_tokens", 4096
                ),
            }
            anthropic_models.append(anthropic_model)

        return {
            "data": anthropic_models,
            "has_more": False,
        }

    @staticmethod
    def transform_single_model_response(
        openrouter_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform single OpenRouter model to Anthropic format.

        Args:
            openrouter_model: Single model from OpenRouter

        Returns:
            Anthropic-compatible model response
        """
        model_id = openrouter_model.get("id", "")
        claude_name = model_id.replace("anthropic/", "").replace(".", "-")

        return {
            "id": claude_name,
            "type": "model",
            "display_name": openrouter_model.get("name", model_id),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "input_token_limit": openrouter_model.get("architecture", {}).get(
                "context_length", 128000
            ),
            "output_token_limit": openrouter_model.get("top_provider", {}).get(
                "max_completion_tokens", 4096
            ),
        }
