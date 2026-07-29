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

"""Tests for the OpenAI Chat Completions adapter."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

import httpx
from fastapi import FastAPI
from openai import AsyncOpenAI

from nexrl.remote_rollout.openai_adapter import create_openai_router


def _result(*, tool_call: bool = False) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": "hello"}
    finish_reason = "stop"
    if tool_call:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
        }
        finish_reason = "tool_calls"
    return {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 123,
        "model": "backend-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
        "nexrl_train": {"prompt_tokens": [1, 2, 3], "response_tokens": [4, 5]},
        "sampling_mask": [1, 1],
        "weight_version": "actor-step-7",
    }


class FakeInferenceClient:
    """Return one response and record calls."""

    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    async def generate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((messages, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return deepcopy(self.result)


def _app(client: FakeInferenceClient) -> FastAPI:
    app = FastAPI()
    app.include_router(create_openai_router(client))  # type: ignore[arg-type]
    return app


def test_non_streaming_request_ignores_extensions_and_hides_internal_fields() -> None:
    async def scenario() -> None:
        fake = FakeInferenceClient(_result())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app(fake)),
            base_url="http://adapter",
        ) as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": "harness-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.4,
                    "custom_extension": {"enabled": True},
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "harness-model"
        assert body["choices"][0]["message"]["content"] == "hello"
        assert "nexrl_train" not in body
        assert "sampling_mask" not in body
        assert "weight_version" not in body
        assert "custom_extension" not in body
        assert fake.calls == [
            (
                [{"role": "user", "content": "hi"}],
                {"tools": []},
            )
        ]

    asyncio.run(scenario())


def test_streaming_text_and_usage_are_consumable_by_openai_sdk() -> None:
    async def scenario() -> None:
        fake = FakeInferenceClient(_result())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app(fake)),
            base_url="http://adapter",
        ) as http:
            client = AsyncOpenAI(
                api_key="test",
                base_url="http://adapter/v1",
                http_client=http,
            )
            stream = await client.chat.completions.create(
                model="harness-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks = [chunk async for chunk in stream]

        assert "".join(
            chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices
        ) == ("hello")
        assert all(chunk.model == "harness-model" for chunk in chunks)
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.total_tokens == 5

    asyncio.run(scenario())


def test_streaming_tool_call_is_consumable_by_openai_sdk() -> None:
    async def scenario() -> None:
        fake = FakeInferenceClient(_result(tool_call=True))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app(fake)),
            base_url="http://adapter",
        ) as http:
            client = AsyncOpenAI(
                api_key="test",
                base_url="http://adapter/v1",
                http_client=http,
            )
            stream = await client.chat.completions.create(
                model="harness-model",
                messages=[{"role": "user", "content": "weather?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "description": "Get weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                stream=True,
            )
            chunks = [chunk async for chunk in stream]

        tool_chunks = [
            chunk for chunk in chunks if chunk.choices and chunk.choices[0].delta.tool_calls
        ]
        tool_call = tool_chunks[0].choices[0].delta.tool_calls[0]
        assert tool_call.function is not None
        assert tool_call.function.name == "weather"
        assert tool_call.function.arguments == '{"city":"Paris"}'

    asyncio.run(scenario())


def test_validation_and_backend_errors_use_openai_error_shape() -> None:
    async def scenario() -> None:
        invalid = FakeInferenceClient(_result())
        failing = FakeInferenceClient(RuntimeError("private backend detail"))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app(invalid)),
            base_url="http://adapter",
        ) as http:
            invalid_response = await http.post(
                "/v1/chat/completions",
                json={"model": "", "messages": [{"role": "user", "content": "hi"}]},
            )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app(failing)),
            base_url="http://adapter",
        ) as http:
            failed_response = await http.post(
                "/v1/chat/completions",
                json={"model": "alias", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert invalid_response.status_code == 400
        assert invalid_response.json()["error"]["type"] == "invalid_request_error"
        assert failed_response.status_code == 500
        assert failed_response.json()["error"] == {
            "message": "Inference request failed",
            "type": "server_error",
            "param": None,
            "code": "inference_error",
        }
        assert "private backend detail" not in failed_response.text

    asyncio.run(scenario())
