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

"""Execute one remote rollout in an E2B sandbox."""

from __future__ import annotations

import json
import os
from contextlib import suppress

from omegaconf import DictConfig, OmegaConf

from .contracts import RemoteRolloutRequest, RemoteRolloutResult

_INPUT_PATH = "/tmp/nexrl-rollout-input.json"
_RESULT_PATH = "/tmp/nexrl-rollout-result.json"
_ENTRYPOINT_PATH = "/opt/nexrl/run_rollout.py"
_ENTRYPOINT_COMMAND = f"python {_ENTRYPOINT_PATH} --input {_INPUT_PATH} --output {_RESULT_PATH}"


async def run_rollout_in_e2b(
    request: RemoteRolloutRequest,
    config: DictConfig,
    *,
    model_path: str,
    template: str,
    timeout_seconds: int,
) -> RemoteRolloutResult:
    """Run the fixed rollout entrypoint provided by an E2B template.

    The template entrypoint reads the input JSON passed via ``--input`` and
    writes one ``RemoteRolloutResult`` JSON document to ``--output``. Only
    configuration consumed by ``E2BRolloutRuntime`` is sent to the sandbox.
    """

    if not model_path.strip():
        raise ValueError("model_path must not be empty")
    if not template.strip():
        raise ValueError("template must not be empty")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 1
    ):
        raise ValueError("timeout_seconds must be a positive integer")
    try:
        from e2b import AsyncSandbox  # pylint: disable=import-error,import-outside-toplevel
    except ImportError as exc:
        raise ImportError("The e2b package is required to execute remote rollouts") from exc

    api_key = os.getenv("WEAVER_API_KEY") or OmegaConf.select(config, "backend.api_key")
    if not api_key:
        raise ValueError("WEAVER_API_KEY is required inside the E2B sandbox")
    base_url = os.getenv("WEAVER_BASE_URL") or OmegaConf.select(config, "backend.base_url")

    config_data = {
        "models": {
            "actor": {
                "name": config.models.actor.name,
                "tokenizer": config.models.actor.tokenizer,
            }
        },
        "rollout": {
            "num_workers": config.rollout.num_workers,
            "max_concurrent_requests": config.rollout.get("max_concurrent_requests"),
            "sampling": OmegaConf.to_container(
                config.rollout.sampling,
                resolve=True,
                throw_on_missing=True,
            ),
            "parsers": OmegaConf.to_container(
                config.rollout.parsers,
                resolve=True,
                throw_on_missing=True,
            ),
        },
        # One fixed model_path is used, so there is no weight-sync controller.
        "training": {"weight_sync": {"freeze_inflight": False}},
        "algorithm": {
            "sampling_logprob_source": config.algorithm.sampling_logprob_source,
        },
        "data": {
            "max_prompt_length": config.data.max_prompt_length,
            "max_sequence_length": config.data.max_sequence_length,
        },
        "backend": {
            "type": config.backend.type,
            "base_url": None,
            "api_key": None,
        },
    }

    sandbox_env = {"WEAVER_API_KEY": str(api_key)}
    if base_url:
        sandbox_env["WEAVER_BASE_URL"] = str(base_url)

    input_json = json.dumps(
        {
            "request": request.model_dump(mode="json"),
            "config": config_data,
            "model_path": model_path,
        },
        ensure_ascii=False,
    )

    sandbox = await AsyncSandbox.create(
        template=template,
        timeout=timeout_seconds,
        envs=sandbox_env,
    )
    try:
        await sandbox.files.write(_INPUT_PATH, input_json)
        await sandbox.commands.run(
            _ENTRYPOINT_COMMAND,
            timeout=timeout_seconds,
        )
        result = RemoteRolloutResult.model_validate_json(await sandbox.files.read(_RESULT_PATH))
        if result.rollout_id != request.rollout_id:
            raise ValueError(
                "E2B rollout result ID does not match request: "
                f"expected={request.rollout_id!r}, actual={result.rollout_id!r}"
            )
    except BaseException:
        with suppress(Exception):
            await sandbox.kill()
        raise
    else:
        await sandbox.kill()
        return result
