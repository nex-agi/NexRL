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

from __future__ import annotations

from typing import Any

import pytest
from omegaconf import OmegaConf

pytest.importorskip("ray")

from nexrl.weaver import WeaverInferenceServiceClient
from nexrl.weaver.weaver_service_holder import WeaverServiceHolder


def _config():
    return OmegaConf.create(
        {
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 32,
            "inference_service": {
                "model": "Qwen/Qwen3-8B",
                "identifier": "default",
                "max_tokens": 128,
                "freeze_for_weight_sync": False,
            },
        }
    )


def _sample_result() -> dict[str, Any]:
    return {
        "response": "ok",
        "prompt_tokens": [1, 2],
        "response_tokens": [3],
        "response_logprobs": [-0.1],
        "response_old_logprobs": [-0.1],
        "finish_reason": "stop",
        "tool_string": None,
        "reasoning_string": None,
        "is_valid": True,
    }


class FakeServiceHolder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def sample_from_prompt(self, **kwargs):
        self.calls.append(("prompt", kwargs))
        return _sample_result()

    def sample_from_messages(self, **kwargs):
        self.calls.append(("messages", kwargs))
        return _sample_result()

    def sample_from_token_ids(self, **kwargs):
        self.calls.append(("token_ids", kwargs))
        return _sample_result()


def test_weaver_inference_client_forwards_configured_sampling_params():
    holder = FakeServiceHolder()
    client = WeaverInferenceServiceClient(_config(), holder)  # type: ignore[arg-type]

    client.completion("hello")
    client.generate([{"role": "user", "content": "hello"}])

    assert holder.calls[0][0] == "prompt"
    assert holder.calls[0][1]["temperature"] == 0.4
    assert holder.calls[0][1]["top_p"] == 0.8
    assert holder.calls[0][1]["top_k"] == 32

    assert holder.calls[1][0] == "messages"
    assert holder.calls[1][1]["temperature"] == 0.4
    assert holder.calls[1][1]["top_p"] == 0.8
    assert holder.calls[1][1]["top_k"] == 32


def test_weaver_generate_with_token_allows_sampling_param_overrides():
    holder = FakeServiceHolder()
    client = WeaverInferenceServiceClient(_config(), holder)  # type: ignore[arg-type]

    client.generate_with_token(
        [1, 2],
        sampling_params={
            "max_new_tokens": 16,
            "temperature": 0.2,
            "top_p": 0.6,
            "top_k": 8,
        },
    )

    call = holder.calls[0][1]
    assert call["max_tokens"] == 16
    assert call["temperature"] == 0.2
    assert call["top_p"] == 0.6
    assert call["top_k"] == 8


class FakeTokenizer:
    def encode(self, prompt, add_special_tokens=False):  # pylint: disable=unused-argument
        return [1, 2]

    def decode(self, tokens, skip_special_tokens=True):  # pylint: disable=unused-argument
        return "ok"


class FakeSamplingClient:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def sample(self, **kwargs):
        self.payload = kwargs["sampling_params"].to_payload()
        return {"sequences": [{"tokens": [3], "logprobs": [-0.1], "text": "ok"}]}


def test_weaver_service_holder_writes_sampling_params_to_sdk_payload():
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._tokenizer = FakeTokenizer()  # pylint: disable=protected-access
    holder._sampling_client = FakeSamplingClient()  # pylint: disable=protected-access

    holder.sample_from_prompt(
        prompt="hello",
        max_tokens=16,
        temperature=0.2,
        top_p=0.6,
        top_k=8,
    )

    payload = holder._sampling_client.payload  # pylint: disable=protected-access
    assert payload == {
        "temperature": 0.2,
        "top_p": 0.6,
        "top_k": 8,
        "max_tokens": 16,
    }
