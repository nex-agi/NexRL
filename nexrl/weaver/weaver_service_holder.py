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

"""
Weaver Service Holder - Centralized Weaver client manager

This class owns tokenizer and all Weaver clients (service, training, sampling).
All Weaver interactions go through this single instance. It mirrors the Tinker
holder but uses the synchronous Weaver SDK.
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from weaver import ServiceClient, types
except ImportError:
    ServiceClient = None
    types = None

from ..utils.url_utils import ensure_url_scheme

logger = logging.getLogger(__name__)


class WeaverServiceHolder:
    """
    Centralized manager for Weaver services.
    """

    def __init__(
        self,
        base_model: str = "",
        lora_rank: int = 32,
        base_url: str | None = None,
        tokenizer_path: str | None = None,
        training_mode: str = "lora",
    ):
        assert ServiceClient is not None, "weaver package not installed"
        assert types is not None, "weaver package not installed"

        logger.info(
            f"Initializing WeaverServiceHolder with base_model: {base_model}, base_url: {base_url}"
        )

        # Ensure base_url has proper http:// scheme
        normalized_url = ensure_url_scheme(base_url, default_scheme="https") if base_url else None
        self._service_client = ServiceClient(
            base_url=normalized_url, api_key=os.getenv("WEAVER_API_KEY")
        )
        self._base_model = base_model
        self._lora_rank = lora_rank

        # Ensure connection/session
        self._service_client.connect()

        # Create training client
        logger.info("Creating Weaver training client")
        self._training_client = self._service_client.create_model(
            base_model=base_model, training_mode=training_mode, lora_config={"rank": lora_rank}
        )

        if tokenizer_path is None:
            # Initialize tokenizer from training client helper
            self._tokenizer = self._training_client.get_tokenizer()
        else:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # Save initial weights and create sampling client
        logger.info("Saving initial weights for sampler")
        sampling_path = self._training_client.save_weights_for_sampler(name="initial")
        self._current_sampling_path = sampling_path

        logger.info(f"Creating initial sampling client from path: {self._current_sampling_path}")
        self._sampling_client = self._service_client.create_sampling_client(
            model_path=self._current_sampling_path,
            base_model=self._base_model,
            model_id=getattr(self._training_client, "model_id", None),
        )

        logger.info("Initialized WeaverServiceHolder")

    def close(self):
        """
        Close the Weaver service client and cleanup resources.
        """
        if hasattr(self, "_service_client") and self._service_client is not None:
            logger.info("Closing WeaverServiceHolder and service client")
            try:
                self._service_client.close()
            except Exception as e:
                logger.warning(f"Error closing service client: {e}")

    def __del__(self):
        """
        Destructor to ensure cleanup when the object is garbage collected.
        """
        self.close()

    # --------------------------------------------------------------------- #
    # Public API used by NexRL components
    # --------------------------------------------------------------------- #
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        return self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            tools=tools,
        )

    def sample_from_messages(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        num_samples: int = 1,
        tools: list[dict[str, Any]] | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        from weaver import types as weaver_types

        prompt_tokens = self.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True
        )
        model_input = weaver_types.ModelInput.from_ints(prompt_tokens)  # type: ignore[arg-type]

        sampling_params = weaver_types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )

        sample_result = self._sampling_client.sample(
            prompt=model_input,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=False,
            wait=True,
        )

        sequence = sample_result.get("sequences")[0]
        response_tokens = list(sequence["tokens"]) or []
        response_logprobs = list(sequence["logprobs"]) or []

        response = sequence.get("text") or self._tokenizer.decode(
            response_tokens, skip_special_tokens=True
        )

        tool_calls: list[dict[str, Any]] | None = None
        import re

        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
        if tool_call_match:
            tool_calls = self._parse_and_normalize_tool_call(tool_call_match.group(1))

        finish_reason = sequence.get("stop_reason") or (
            "stop" if len(response_tokens) < max_tokens else "length"
        )

        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "response_logprobs": response_logprobs,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
            "is_valid": True,
        }

    def sample_from_prompt(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 1.0,
        num_samples: int = 1,
        stop: list[str] | None = None,
    ) -> dict:
        from weaver import types as weaver_types

        prompt_tokens = self._tokenizer.encode(prompt, add_special_tokens=False)
        model_input = weaver_types.ModelInput.from_ints(prompt_tokens)
        sampling_params = weaver_types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )

        sample_result = self._sampling_client.sample(
            prompt=model_input,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=False,
            wait=True,
        )

        sequence = (sample_result.get("sequences") or [{}])[0]
        response_tokens = sequence.get("tokens") or []
        response_logprobs = sequence.get("logprobs") or []
        response = sequence.get("text") or self._tokenizer.decode(
            response_tokens, skip_special_tokens=True
        )
        finish_reason = sequence.get("stop_reason") or (
            "stop" if len(response_tokens) < max_tokens else "length"
        )

        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "response_logprobs": response_logprobs,
            "finish_reason": finish_reason,
            "tool_calls": None,
            "is_valid": True,
        }

    def compute_logprobs(self, tokens: list[int]) -> list[float | None]:
        from weaver import types as weaver_types

        model_input = weaver_types.ModelInput.from_ints(tokens=tokens)
        logprobs = self._sampling_client.compute_logprobs(prompt=model_input)
        return list(logprobs) if logprobs else []

    def forward_backward(
        self,
        datums_data: list[dict],
        loss_fn: str = "importance_sampling",
        loss_fn_config: dict | None = None,
    ) -> dict:
        import torch
        from weaver.types.tensor import tensor_payload

        def _extract_metrics_from_result(result_obj: Any) -> dict[str, Any]:
            """
            Weaver responses are not perfectly stable across versions.

            We try (in order):
            - result["metrics"]
            - result["result"]["metrics"]
            - aggregate metrics from per-task results in result["forward_backward_task_results"]
              (or nested under result["result"][...]).
            """
            if not isinstance(result_obj, dict):
                return {}

            # Normalize one level of nesting
            top = result_obj
            inner = result_obj.get("result")
            if isinstance(inner, dict):
                # Prefer inner container if present (common pattern in SDKs)
                top = inner

            metrics = top.get("metrics")
            if isinstance(metrics, dict):
                return metrics

            fb_task_results = top.get("forward_backward_task_results")
            if not isinstance(fb_task_results, dict) or not fb_task_results:
                return {}

            # Aggregate per-task metrics.
            # - Sum keys that are naturally counts/sums (tokens, "*:sum")
            # - Mean everything else to keep scales stable across batch sizes.
            sums: dict[str, float] = {}
            counts: dict[str, int] = {}
            for _, task_res in fb_task_results.items():
                if not isinstance(task_res, dict):
                    continue
                tm = task_res.get("metrics")
                if not isinstance(tm, dict):
                    continue
                for k, v in tm.items():
                    if isinstance(v, (int, float)):
                        sums[k] = sums.get(k, 0.0) + float(v)
                        counts[k] = counts.get(k, 0) + 1

            aggregated: dict[str, Any] = {}
            for k, total in sums.items():
                if k == "tokens" or str(k).endswith(":sum"):
                    aggregated[k] = total
                else:
                    aggregated[k] = total / max(counts.get(k, 1), 1)
            return aggregated

        datums = []
        for d in datums_data:
            loss_fn_inputs = {}
            for key, value in d["loss_fn_inputs"].items():
                loss_fn_inputs[key] = tensor_payload(torch.tensor(value))

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=d["input_tokens"]),
                loss_fn_inputs=loss_fn_inputs,
            )
            datums.append(datum)

        result = self._training_client.forward_backward(
            datums, loss_fn=loss_fn, loss_fn_config=loss_fn_config, wait=True
        )
        metrics = _extract_metrics_from_result(result)
        return {"loss": metrics.get("loss", 0.0), "metrics": metrics}

    def optim_step(
        self,
        learning_rate: float = 2e-6,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay: float = 1e-2,
        eps: float = 1e-8,
        grad_clip_norm: float = 1.0,
    ) -> dict:
        adam_params = types.AdamParams(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            grad_clip_norm=grad_clip_norm,
        )

        result = self._training_client.optim_step(adam_params, wait=True)

        # Extract metrics from optimizer step result
        optim_metrics = {}
        if isinstance(result, dict) and "metrics" in result:
            optim_metrics = result["metrics"]

        return {"status": "success", "metrics": optim_metrics}

    def forward_backward_and_optim_step(
        self,
        datums_data: list[dict],
        loss_fn: str = "importance_sampling",
        loss_fn_config: dict | None = None,
        learning_rate: float = 2e-6,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay: float = 1e-2,
        eps: float = 1e-8,
        grad_clip_norm: float = 1.0,
    ) -> dict:
        fwd = self.forward_backward(datums_data, loss_fn=loss_fn, loss_fn_config=loss_fn_config)
        optim = self.optim_step(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            grad_clip_norm=grad_clip_norm,
        )

        # Combine metrics from both operations
        combined_metrics = fwd.get("metrics", {}).copy()
        optim_metrics = optim.get("metrics", {})
        combined_metrics.update(optim_metrics)
        print(f"combined_metrics: {combined_metrics}")
        return combined_metrics

    def save_weights_for_sampler(self, name: str) -> str:
        save_path = self._training_client.save_weights_for_sampler(name=name)
        return str(save_path)

    def get_current_sampling_path(self) -> str:
        return self._current_sampling_path

    def set_current_sampling_path(self, new_path: str) -> None:
        self._current_sampling_path = new_path

    def update_sampling_client(self) -> None:
        self._sampling_client = self._service_client.create_sampling_client(
            model_path=self._current_sampling_path,
            base_model=self._base_model,
            model_id=getattr(self._training_client, "model_id", None),
        )
        logger.info(f"Sampling client updated to path: {self._current_sampling_path}")

    def get_service_client(self):
        return self._service_client

    def get_tokenizer(self):
        return self._tokenizer

    def _parse_and_normalize_tool_call(self, tool_call_str: str) -> list[dict[str, Any]] | None:
        """
        Parse tool call JSON string and normalize to OpenAI format.

        Expects JSON string like: {"name": str, "args": dict, "id": Optional[str]}
        Returns normalized format: [{"id": str, "type": "function", "function": {"name": str, "arguments": str}}]
        """
        import json
        import uuid

        try:
            tool_call = json.loads(tool_call_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(tool_call, dict):
            return None

        name = tool_call.get("name")
        args = tool_call.get("arguments")
        tool_id = tool_call.get("id")

        if not isinstance(name, str):
            return None

        # Normalize arguments to JSON string per OpenAI format
        if args is None:
            arguments_str = "{}"
        elif isinstance(args, str):
            arguments_str = args
        elif isinstance(args, dict):
            try:
                arguments_str = json.dumps(args)
            except Exception:
                logger.warning(
                    f"Failed to json-serialize tool_call args, falling back to str: {args}"
                )
                arguments_str = str(args)
        else:
            arguments_str = str(args)

        # Ensure tool_id is string or generate one
        if tool_id is not None and not isinstance(tool_id, str):
            tool_id = None
        if tool_id is None:
            tool_id = f"tinker-tool-call-{uuid.uuid4().hex}"

        normalized = [
            {
                "id": tool_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments_str},
            }
        ]

        logger.info(f"Normalized tool calls: {normalized}")
        return normalized
