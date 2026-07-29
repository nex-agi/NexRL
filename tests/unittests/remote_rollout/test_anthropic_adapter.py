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

"""Tests for the Anthropic-compatible rollout adapter."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexrl.inference_service_client.base_inference_service_client import InferenceServiceClient
from nexrl.remote_rollout.anthropic_adapter import create_anthropic_router


class _FakeInferenceClient(InferenceServiceClient):
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        super().__init__()
        self.response = response
        self.messages: list[dict[str, Any]] | None = None
        self.tools: list[dict[str, Any]] | None = None

    async def completion(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.messages = messages
        self.tools = kwargs["tools"]
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        raise NotImplementedError


def _inference_response(*, tool_call: bool = False) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "I will check." if tool_call else "Hello",
    }
    finish_reason = "stop"
    if tool_call:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": '{"city":"Paris"}',
                },
            }
        ]
        finish_reason = "tool_calls"
    return {
        "id": "chatcmpl_internal",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
        },
        "nexrl_train": {
            "prompt_tokens": [1, 2],
            "response_tokens": [3],
            "response_logprobs": [-0.1],
        },
        "sampling_mask": [[3]],
        "weight_version": "actor-step-7",
    }


def _test_client(
    response: dict[str, Any] | Exception | None = None,
) -> tuple[TestClient, _FakeInferenceClient]:
    inference_client = _FakeInferenceClient(response or _inference_response())
    app = FastAPI()
    app.include_router(create_anthropic_router(inference_client))
    return TestClient(app), inference_client


def _request(**updates: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": "claude-alias",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    request.update(updates)
    return request


def test_messages_maps_system_tools_and_tool_history() -> None:
    http, inference = _test_client(_inference_response(tool_call=True))
    response = http.post(
        "/v1/messages",
        json=_request(
            system=[
                {
                    "type": "text",
                    "text": "Use tools",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Weather?",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_0",
                            "name": "weather",
                            "input": {"city": "Rome"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_0",
                            "content": [{"type": "text", "text": "sunny"}],
                            "is_error": False,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
            tools=[
                {
                    "name": "weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tool_choice={"type": "auto"},
        ),
    )

    assert response.status_code == 200
    assert inference.messages == [
        {"role": "system", "content": "Use tools"},
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Rome"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": "sunny",
            "is_error": False,
        },
    ]
    assert inference.tools == [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]

    body = response.json()
    assert body["model"] == "claude-alias"
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {"type": "text", "text": "I will check."},
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "weather",
            "input": {"city": "Paris"},
        },
    ]
    assert body["usage"] == {"input_tokens": 12, "output_tokens": 3}
    assert set(body) == {
        "id",
        "type",
        "role",
        "model",
        "content",
        "stop_reason",
        "stop_sequence",
        "usage",
    }
    assert "nexrl_train" not in response.text
    assert "sampling_mask" not in response.text
    assert "weight_version" not in response.text


def test_messages_stream_emits_anthropic_events() -> None:
    http, _ = _test_client(_inference_response(tool_call=True))

    response = http.post("/v1/messages", json=_request(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        (lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
        for chunk in response.text.strip().split("\n\n")
        if (lines := chunk.splitlines())
    ]
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2][1]["delta"] == {
        "type": "text_delta",
        "text": "I will check.",
    }
    assert events[5][1]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"city":"Paris"}',
    }
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"
    assert "nexrl_train" not in response.text
    assert "sampling_mask" not in response.text
    assert "weight_version" not in response.text


def test_tool_result_error_and_following_text_keep_their_order() -> None:
    http, inference = _test_client()

    response = http.post(
        "/v1/messages",
        json=_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_0",
                            "content": "failed",
                            "is_error": True,
                        },
                        {"type": "text", "text": "Try something else"},
                    ],
                }
            ]
        ),
    )

    assert response.status_code == 200
    assert inference.messages == [
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": "failed",
            "is_error": True,
        },
        {"role": "user", "content": "Try something else"},
    ]


def test_request_validation_uses_anthropic_error_shape() -> None:
    http, _ = _test_client()

    response = http.post(
        "/v1/messages",
        json=_request(tool_choice={"type": "any"}),
    )

    assert response.status_code == 400
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "tool_choice must be omitted or {'type': 'auto'}",
        },
    }


def test_backend_error_hides_internal_details() -> None:
    http, _ = _test_client(RuntimeError("private backend detail"))

    response = http.post("/v1/messages", json=_request())

    assert response.status_code == 500
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "Inference request failed"},
    }
    assert "private backend detail" not in response.text
