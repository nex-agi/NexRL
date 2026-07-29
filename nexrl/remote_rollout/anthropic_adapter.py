# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anthropic-compatible routes backed by an inference service client."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..inference_service_client.base_inference_service_client import InferenceServiceClient


def create_anthropic_router(client: InferenceServiceClient) -> APIRouter:
    """Create the Anthropic Messages routes used inside a rollout sandbox."""

    router = APIRouter()

    @router.post("/v1/messages")
    async def messages(request: Request):
        try:
            model, openai_messages, tools, stream = _parse_request(await request.json())
        except (TypeError, ValueError) as exc:
            return _error(str(exc))

        try:
            result = await client.generate(messages=openai_messages, tools=tools)
            response = _message_response(result, model)
        except Exception:  # pylint: disable=broad-exception-caught
            return _error("Inference request failed", error_type="api_error", status_code=500)
        if not stream:
            return response
        return StreamingResponse(
            _message_events(response),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


def _parse_request(
    payload: Any,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], bool]:
    body = _object(payload, "request body")
    model = _non_empty_string(body.get("model"), "model")

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty array")

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise TypeError("stream must be a boolean")

    if "tool_choice" in body and body["tool_choice"] != {"type": "auto"}:
        raise ValueError("tool_choice must be omitted or {'type': 'auto'}")

    messages: list[dict[str, Any]] = []
    if body.get("system") is not None:
        messages.append({"role": "system", "content": _text_content(body["system"])})
    for message in raw_messages:
        messages.extend(_message_to_openai(message))

    return model, messages, _tools_to_openai(body.get("tools", [])), stream


def _message_to_openai(value: Any) -> list[dict[str, Any]]:
    message = _object(value, "message")
    role = message.get("role")
    if role not in {"user", "assistant"}:
        raise ValueError(f"unsupported message role: {role!r}")

    content = message.get("content", "")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        raise TypeError("message content must be a string or content block array")

    if role == "assistant":
        text: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for raw_block in content:
            block = _object(raw_block, "assistant content block")
            block_type = block.get("type")
            if block_type == "text":
                text.append(_string(block.get("text", ""), "text"))
            elif block_type == "tool_use":
                tool_calls.append(_tool_use_to_openai(block))
            else:
                raise ValueError(f"unsupported assistant content block: {block_type!r}")
        assistant: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        return [assistant]

    result: list[dict[str, Any]] = []
    text = []
    for raw_block in content:
        block = _object(raw_block, "user content block")
        block_type = block.get("type")
        if block_type == "text":
            text.append(_string(block.get("text", ""), "text"))
            continue
        if block_type != "tool_result":
            raise ValueError(f"unsupported user content block: {block_type!r}")
        if text:
            result.append({"role": "user", "content": "".join(text)})
            text = []
        is_error = block.get("is_error", False)
        if not isinstance(is_error, bool):
            raise TypeError("tool_result.is_error must be a boolean")
        result.append(
            {
                "role": "tool",
                "tool_call_id": _non_empty_string(
                    block.get("tool_use_id"), "tool_result.tool_use_id"
                ),
                "content": _text_content(block.get("content", "")),
                "is_error": is_error,
            }
        )
    if text or not result:
        result.append({"role": "user", "content": "".join(text)})
    return result


def _tool_use_to_openai(block: dict[str, Any]) -> dict[str, Any]:
    arguments = _object(block.get("input", {}), "tool_use.input")
    return {
        "id": _non_empty_string(block.get("id"), "tool_use.id"),
        "type": "function",
        "function": {
            "name": _non_empty_string(block.get("name"), "tool_use.name"),
            "arguments": json.dumps(arguments, separators=(",", ":"), ensure_ascii=False),
        },
    }


def _tools_to_openai(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("tools must be an array")

    tools: list[dict[str, Any]] = []
    for raw_tool in value:
        tool = _object(raw_tool, "tool")
        function: dict[str, Any] = {
            "name": _non_empty_string(tool.get("name"), "tool.name"),
            "description": _string(tool.get("description", ""), "tool.description"),
            "parameters": _object(tool.get("input_schema"), "tool.input_schema"),
        }
        if isinstance(tool.get("strict"), bool):
            function["strict"] = tool["strict"]
        tools.append({"type": "function", "function": function})
    return tools


def _message_response(result: dict[str, Any], model: str) -> dict[str, Any]:
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TypeError("inference response must contain a choice")
    choice = _object(choices[0], "inference choice")
    message = _object(choice.get("message"), "inference message")

    content: list[dict[str, Any]] = []
    text = message.get("content")
    if text is not None:
        if not isinstance(text, str):
            raise TypeError("inference message content must be a string or null")
        if text:
            content.append({"type": "text", "text": text})

    raw_tool_calls = message.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list):
        raise TypeError("inference message tool_calls must be an array")
    for raw_tool_call in raw_tool_calls:
        tool_call = _object(raw_tool_call, "inference tool call")
        function = _object(tool_call.get("function"), "inference tool call function")
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise TypeError("inference tool call arguments must be a JSON object")
        content.append(
            {
                "type": "tool_use",
                "id": _non_empty_string(tool_call.get("id"), "inference tool call id"),
                "name": _non_empty_string(
                    function.get("name"), "inference tool call function name"
                ),
                "input": arguments,
            }
        )

    usage = _object(result.get("usage"), "inference usage")
    finish_reason = choice.get("finish_reason")
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _stop_reason(finish_reason, bool(raw_tool_calls)),
        "stop_sequence": None,
        "usage": {
            "input_tokens": _token_count(usage.get("prompt_tokens"), "prompt_tokens"),
            "output_tokens": _token_count(usage.get("completion_tokens"), "completion_tokens"),
        },
    }


def _message_events(message: dict[str, Any]) -> Iterator[str]:
    start: dict[str, Any] = {
        **message,
        "content": [],
        "stop_reason": None,
        "usage": {
            "input_tokens": message["usage"]["input_tokens"],
            "output_tokens": 0,
        },
    }
    yield _event("message_start", {"type": "message_start", "message": start})

    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            initial: dict[str, Any] = {"type": "text", "text": ""}
            delta: dict[str, Any] = {"type": "text_delta", "text": block["text"]}
        else:
            initial = {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": {},
            }
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(
                    block["input"], separators=(",", ":"), ensure_ascii=False
                ),
            }
        yield _event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": initial,
            },
        )
        yield _event(
            "content_block_delta",
            {"type": "content_block_delta", "index": index, "delta": delta},
        )
        yield _event(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        )

    yield _event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": message["stop_reason"],
                "stop_sequence": message["stop_sequence"],
            },
            "usage": {"output_tokens": message["usage"]["output_tokens"]},
        },
    )
    yield _event("message_stop", {"type": "message_stop"})


def _stop_reason(value: Any, has_tool_calls: bool) -> str:
    if has_tool_calls or value in {"tool_call", "tool_calls"}:
        return "tool_use"
    if value == "length":
        return "max_tokens"
    return "end_turn"


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError("content must be a string or text block array")
    text: list[str] = []
    for raw_block in value:
        block = _object(raw_block, "text content block")
        if block.get("type") != "text":
            raise ValueError("only text blocks are accepted in this content position")
        text.append(_string(block.get("text", ""), "text"))
    return "".join(text)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _non_empty_string(value: Any, name: str) -> str:
    value = _string(value, name)
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _token_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"inference usage {name} must be a non-negative integer")
    return value


def _event(name: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"event: {name}\ndata: {data}\n\n"


def _error(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    status_code: int = 400,
) -> JSONResponse:
    return JSONResponse(
        {
            "type": "error",
            "error": {"type": error_type, "message": message},
        },
        status_code=status_code,
    )
