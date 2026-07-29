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

"""OpenAI Chat Completions routes for remote rollout harnesses."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from ..inference_service_client.base_inference_service_client import (
    InferenceServiceClient,
)

_PUBLIC_FIELDS = {
    "id",
    "object",
    "created",
    "choices",
    "usage",
    "service_tier",
    "system_fingerprint",
}


def create_openai_router(client: InferenceServiceClient) -> APIRouter:
    """Create an OpenAI-compatible Chat Completions router."""

    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any]):
        model = payload.get("model")
        messages = payload.get("messages")
        stream = payload.get("stream", False)
        if not isinstance(model, str) or not model:
            return _error("model must be a non-empty string")
        if not isinstance(messages, list) or not messages:
            return _error("messages must be a non-empty array")
        if not isinstance(stream, bool):
            return _error("stream must be a boolean")
        if payload.get("n", 1) != 1:
            return _error("only n=1 is supported")

        try:
            result = await client.generate(
                messages=messages,
                tools=payload.get("tools") or [],
            )
            response = _public_response(result, model)
        except Exception:  # pylint: disable=broad-exception-caught
            return _error(
                "Inference request failed",
                status_code=500,
                error_type="server_error",
                code="inference_error",
            )

        if not stream:
            return response
        stream_options = payload.get("stream_options")
        include_usage = (
            isinstance(stream_options, dict) and stream_options.get("include_usage") is True
        )
        return StreamingResponse(
            _stream_response(response, include_usage=include_usage),
            media_type="text/event-stream",
        )

    return router


def _public_response(result: dict[str, Any], model: str) -> dict[str, Any]:
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Inference result must contain at least one choice")
    response = {key: value for key, value in result.items() if key in _PUBLIC_FIELDS}
    response.setdefault("id", f"chatcmpl_{uuid.uuid4().hex}")
    response.setdefault("object", "chat.completion")
    response.setdefault("created", int(time.time()))
    response["model"] = model
    return response


def _stream_response(
    response: dict[str, Any],
    *,
    include_usage: bool,
) -> Iterable[str]:
    base = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": response["model"],
    }
    choice = response["choices"][0]
    message = choice["message"]
    yield _sse(
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    )
    if message.get("content") is not None:
        yield _sse(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {"content": message["content"]}, "finish_reason": None}
                ],
            }
        )
    if message.get("tool_calls"):
        tool_calls = [
            {"index": index, **tool_call} for index, tool_call in enumerate(message["tool_calls"])
        ]
        yield _sse(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}
                ],
            }
        )
    yield _sse(
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason")}],
        }
    )
    if include_usage:
        yield _sse({**base, "choices": [], "usage": response.get("usage")})
    yield "data: [DONE]\n\n"


def _error(
    message: str,
    *,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    code: str = "invalid_request",
) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
        status_code=status_code,
    )


def _sse(value: dict[str, Any]) -> str:
    return f"data: {json.dumps(value, separators=(',', ':'), ensure_ascii=False)}\n\n"
