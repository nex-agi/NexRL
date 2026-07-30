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

"""Tests for the Driver-side E2B rollout executor."""

from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType
from typing import Any

import pytest
from omegaconf import OmegaConf

from nexrl.remote_rollout import e2b_executor
from nexrl.remote_rollout.contracts import (
    RemoteRolloutRequest,
    RemoteRolloutResult,
    RemoteTrajectory,
)


class _FakeFiles:
    def __init__(self, result_json: str) -> None:
        self._result_json = result_json
        self.writes: list[tuple[str, str]] = []
        self.reads: list[str] = []

    async def write(self, path: str, data: str) -> None:
        self.writes.append((path, data))

    async def read(self, path: str) -> str:
        self.reads.append(path)
        return self._result_json


class _FakeCommands:
    def __init__(self, error: Exception | None) -> None:
        self._error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, command: str, **kwargs: Any) -> None:
        self.calls.append((command, kwargs))
        if self._error is not None:
            raise self._error


class _FakeSandbox:
    def __init__(
        self,
        result_json: str,
        command_error: Exception | None,
        kill_error: Exception | None,
    ) -> None:
        self.files = _FakeFiles(result_json)
        self.commands = _FakeCommands(command_error)
        self._kill_error = kill_error
        self.killed = False

    async def kill(self) -> None:
        self.killed = True
        if self._kill_error is not None:
            raise self._kill_error


class _FakeAsyncSandbox:
    result_json = ""
    command_error: Exception | None = None
    kill_error: Exception | None = None
    create_calls: list[dict[str, Any]] = []
    instances: list[_FakeSandbox] = []

    @classmethod
    async def create(cls, **kwargs: Any) -> _FakeSandbox:
        cls.create_calls.append(kwargs)
        sandbox = _FakeSandbox(cls.result_json, cls.command_error, cls.kill_error)
        cls.instances.append(sandbox)
        return sandbox


def _result(rollout_id: str) -> RemoteRolloutResult:
    return RemoteRolloutResult(
        rollout_id=rollout_id,
        trajectories=[
            RemoteTrajectory(
                tokens=[1, 2],
                loss_mask=[0, 1],
                log_probs=[0.0, -0.1],
                old_log_probs=[0.0, -0.2],
            )
        ],
        reward=1.0,
        metrics={"tests_passed": 1.0},
    )


def _config():
    return OmegaConf.create(
        {
            "models": {
                "actor": {
                    "name": "${oc.env:ACTOR_MODEL,actor-default}",
                    "tokenizer": "tokenizer-path",
                }
            },
            "rollout": {
                "num_workers": 2,
                "max_concurrent_requests": 4,
                "sampling": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 20,
                    "min_p": 0.0,
                    "max_tokens": 512,
                },
                "parsers": {
                    "type": "builtin",
                    "reasoning": "think_tag",
                    "tool": "qwen25",
                },
            },
            "backend": {
                "type": "weaver",
                "base_url": "https://weaver.example",
                "api_key": "config-key",
            },
            "training": {"weight_sync": {"freeze_inflight": True}},
            "algorithm": {"sampling_logprob_source": "inference"},
            "data": {
                "max_prompt_length": 1024,
                "max_sequence_length": 4096,
            },
            "runtime": {
                "logger": {
                    "feishu": {"url": "${oc.env:UNRELATED_SECRET,}"},
                }
            },
        }
    )


@pytest.fixture(autouse=True)
def _fake_e2b(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAsyncSandbox.result_json = _result("rollout-1").model_dump_json()
    _FakeAsyncSandbox.command_error = None
    _FakeAsyncSandbox.kill_error = None
    _FakeAsyncSandbox.create_calls.clear()
    _FakeAsyncSandbox.instances.clear()
    e2b_module = ModuleType("e2b")
    e2b_module.AsyncSandbox = _FakeAsyncSandbox
    monkeypatch.setitem(sys.modules, "e2b", e2b_module)
    monkeypatch.delenv("WEAVER_API_KEY", raising=False)
    monkeypatch.delenv("WEAVER_BASE_URL", raising=False)


def test_run_rollout_in_e2b_transfers_json_and_closes_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACTOR_MODEL", "resolved-actor")
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-send")
    request = RemoteRolloutRequest(
        rollout_id="rollout-1",
        task={"instruction": "修复它"},
    )
    config = _config()

    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            request,
            config,
            model_path="weaver://weights/actor-step-7",
            template="swebench-template",
            timeout_seconds=3600,
        )
    )

    assert result == _result("rollout-1")
    assert _FakeAsyncSandbox.create_calls == [
        {
            "template": "swebench-template",
            "timeout": 3600,
            "envs": {
                "WEAVER_API_KEY": "config-key",
                "WEAVER_BASE_URL": "https://weaver.example",
            },
        }
    ]

    sandbox = _FakeAsyncSandbox.instances[0]
    input_path, raw_input = sandbox.files.writes[0]
    assert input_path == "/tmp/nexrl-rollout-input.json"
    assert sandbox.files.reads == ["/tmp/nexrl-rollout-result.json"]
    assert sandbox.commands.calls == [
        (
            "python /opt/nexrl/run_rollout.py "
            "--input /tmp/nexrl-rollout-input.json "
            "--output /tmp/nexrl-rollout-result.json",
            {"timeout": 3600},
        )
    ]
    assert sandbox.killed

    payload = json.loads(raw_input)
    assert payload["request"] == request.model_dump(mode="json")
    assert payload["model_path"] == "weaver://weights/actor-step-7"
    assert payload["config"] == {
        "models": {
            "actor": {
                "name": "resolved-actor",
                "tokenizer": "tokenizer-path",
            }
        },
        "rollout": {
            "num_workers": 2,
            "max_concurrent_requests": 4,
            "sampling": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.0,
                "max_tokens": 512,
            },
            "parsers": {
                "type": "builtin",
                "reasoning": "think_tag",
                "tool": "qwen25",
            },
        },
        "training": {"weight_sync": {"freeze_inflight": False}},
        "algorithm": {"sampling_logprob_source": "inference"},
        "data": {
            "max_prompt_length": 1024,
            "max_sequence_length": 4096,
        },
        "backend": {
            "type": "weaver",
            "base_url": None,
            "api_key": None,
        },
    }
    assert config.backend.api_key == "config-key"
    assert config.training.weight_sync.freeze_inflight
    assert "config-key" not in raw_input
    assert "do-not-send" not in raw_input


def test_run_rollout_in_e2b_rejects_wrong_result_id_and_closes_sandbox() -> None:
    _FakeAsyncSandbox.result_json = _result("wrong-rollout").model_dump_json()

    with pytest.raises(ValueError, match="does not match request"):
        asyncio.run(
            e2b_executor.run_rollout_in_e2b(
                RemoteRolloutRequest(rollout_id="rollout-1", task={}),
                _config(),
                model_path="weaver://weights/actor-step-7",
                template="swebench-template",
                timeout_seconds=3600,
            )
        )

    assert _FakeAsyncSandbox.instances[-1].killed


def test_run_rollout_in_e2b_closes_sandbox_when_entrypoint_fails() -> None:
    _FakeAsyncSandbox.command_error = RuntimeError("entrypoint failed")
    _FakeAsyncSandbox.kill_error = RuntimeError("kill failed")

    with pytest.raises(RuntimeError, match="entrypoint failed"):
        asyncio.run(
            e2b_executor.run_rollout_in_e2b(
                RemoteRolloutRequest(rollout_id="rollout-1", task={}),
                _config(),
                model_path="weaver://weights/actor-step-7",
                template="swebench-template",
                timeout_seconds=3600,
            )
        )

    sandbox = _FakeAsyncSandbox.instances[-1]
    assert not sandbox.files.reads
    assert sandbox.killed
