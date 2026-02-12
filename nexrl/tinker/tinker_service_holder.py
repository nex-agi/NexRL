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
Tinker Service Holder - Centralized Tinker client manager

This class owns tokenizer, renderer, and all Tinker clients.
All tokenization and response parsing happens inside this holder.
"""

import json
import logging
from copy import deepcopy
from typing import Any

try:
    import tinker
except ImportError:
    tinker = None

from ..utils.url_utils import ensure_url_scheme

logger = logging.getLogger(__name__)


class TinkerServiceHolder:
    """
    Centralized manager for Tinker services.

    This class owns and manages:
    - tokenizer and renderer (for tokenization and response parsing)
    - tinker.ServiceClient (shared)
    - training_client (for training operations)
    - sampling_client (shared across all rollout workers)

    All Tinker interactions go through this single instance.
    Tokenization and response parsing happen inside this holder to avoid
    passing non-serializable objects through Ray.
    """

    def __init__(
        self,
        base_model: str = "",
        lora_rank: int = 32,
        base_url: str | None = None,
        renderer_name: str | None = None,
        tokenizer_path: str | None = None,
    ):
        """
        Initialize Tinker service holder and create clients.

        Args:
            base_model: Base model name for training
            lora_rank: LoRA rank for training
            base_url: Optional base URL for Tinker service
            renderer_name: Optional renderer name (defaults to recommended for model)
        """
        assert tinker is not None, "tinker package not installed"
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        logger.info(
            f"Initializing TinkerServiceHolder with base_model: {base_model}, base_url: {base_url}"
        )

        # Ensure base_url has proper http:// scheme
        normalized_url = ensure_url_scheme(base_url, default_scheme="https") if base_url else None
        self._service_client = tinker.ServiceClient(base_url=normalized_url)
        self._base_model = base_model
        self._lora_rank = lora_rank

        logger.info(f"Service client: {self._service_client}")

        # Initialize tokenizer and renderer
        logger.info(f"Initializing tokenizer and renderer for base_model: {base_model}")
        self._tokenizer = get_tokenizer(tokenizer_path or base_model)

        renderer_name = renderer_name or model_info.get_recommended_renderer_name(base_model)
        logger.info(f"Renderer name: {renderer_name}")

        self._renderer = renderers.get_renderer(renderer_name, self._tokenizer)
        self._stop_sequences = self._renderer.get_stop_sequences()
        logger.info(f"Initialized tokenizer from {base_model}, renderer: {renderer_name}")

        # Create training client
        logger.info(f"Creating training client: base_model={base_model}, rank={lora_rank}")
        self._training_client = self._service_client.create_lora_training_client(
            base_model=base_model, rank=lora_rank
        )

        # Get initial sampling path and create initial sampling client
        logger.info("Getting initial sampling path...")
        sampling_path = self._training_client.save_weights_for_sampler(name="initial").result().path
        self._current_sampling_path = sampling_path

        logger.info(f"Creating initial sampling client from path: {self._current_sampling_path}")
        self._sampling_client = self._service_client.create_sampling_client(
            model_path=self._current_sampling_path
        )

        logger.info("Initialized TinkerServiceHolder")

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        """
        Apply chat template to messages to get a prompt string or tokens.

        Args:
            messages: List of message dicts (chat format, serializable)
            tools: List of tool dictionaries
            add_generation_prompt: Whether to add the generation prompt
            tokenize: If True, return token IDs; if False, return string

        Returns:
            str | list[int]: Formatted prompt string or token IDs
        """
        # Deep-copy so we don't mutate the caller's data
        messages_copy = deepcopy(messages)

        # Ensure tool_call function.arguments are dicts, not JSON strings.
        # Chat templates use Jinja2 .items() on arguments, which requires a mapping.
        for message in messages_copy:
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                continue
            for tool_call in tool_calls:
                function = tool_call.get("function")
                if not function:
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    if arguments != "":
                        try:
                            function["arguments"] = json.loads(arguments, strict=False)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse tool call arguments as JSON: {e}, "
                                f"keeping as string: {arguments[:100]}"
                            )
                    else:
                        function["arguments"] = {}
                elif arguments is None:
                    function["arguments"] = {}

        return self._tokenizer.apply_chat_template(
            messages_copy,
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
    ) -> dict:
        """
        Sample from the model given chat messages.

        Handles tokenization using renderer.build_generation_prompt() and
        response parsing using renderer.parse_response() internally.

        Args:
            messages: List of message dicts (chat format, serializable)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_samples: Number of samples to generate
            tools: List of tool definitions (OpenAI format)

        Returns:
            Dictionary with serializable results (same structure as sample_from_prompt)
        """
        from tinker import types

        # Build prompt using renderer (handles chat template)
        logger.debug(f"Building generation prompt for messages: {messages}")

        prompt_tokens = self.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )

        model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

        sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self._stop_sequences,
        )

        sample_future = self._sampling_client.sample(
            prompt=model_input,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=False,
        )

        sample_result = sample_future.result()

        # Process first sample (for single sample case)
        sequence = sample_result.sequences[0]
        response_tokens = list(sequence.tokens)
        response_logprobs = list(sequence.logprobs) if sequence.logprobs else []

        # Parse response using renderer
        # parsed_message, is_valid = self._renderer.parse_response(response_tokens)
        parsed_message, is_valid = self._parse_response(response_tokens)

        logger.debug(f"Parsed message: {parsed_message}")

        response = parsed_message["content"]
        tool_string = parsed_message.get("tool_string")

        logger.debug(
            f"parsed_message keys: {parsed_message.keys()}, is_valid: {is_valid}, response: {response}, tool_string: {tool_string}"
        )

        # Determine finish reason
        finish_reason = "stop" if len(response_tokens) < max_tokens else "length"

        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "response_logprobs": response_logprobs,
            "finish_reason": finish_reason,
            "tool_string": tool_string,
            "is_valid": is_valid,
        }

    def _parse_response(self, response_tokens: list[int]) -> tuple[dict[str, Any], bool]:
        """
        Parse response tokens and extract raw tool string and reasoning string if present.
        Remove tool call and reasoning tags from the main content.
        Tool and reasoning parsing happens in inference client, not here.
        """
        full_response = self._tokenizer.decode(response_tokens, skip_special_tokens=True)
        import re

        tool_string = None
        reasoning_string = None
        cleaned_content = full_response

        # Extract reasoning string (only the reasoning portion, not the rest)
        if "</think>" in full_response:
            # Extract everything up to and including </think>
            reasoning_match = re.search(r"^(.*?</think>)", full_response, flags=re.DOTALL)
            if reasoning_match:
                reasoning_string = reasoning_match.group(0)  # group(0) = entire match
            # Remove reasoning content from response
            cleaned_content = re.sub(
                r"^(<think>)?(.*?)</think>\s*", "", full_response, flags=re.DOTALL
            )

        # Extract tool string (from the full response)
        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", full_response, re.DOTALL)
        if tool_call_match:
            tool_string = tool_call_match.group(0)

        # Remove tool call tags from content
        if tool_string:
            cleaned_content = re.sub(
                r"<tool_call>.*?</tool_call>\s*", "", cleaned_content, flags=re.DOTALL
            )

        cleaned_content = cleaned_content.strip()

        return {
            "role": "assistant",
            "content": cleaned_content,
            "tool_string": tool_string,
            "reasoning_string": reasoning_string,
        }, True

    def sample_from_prompt(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 1.0,
        num_samples: int = 1,
    ) -> dict:
        """
        Sample from the model given a raw prompt string.

        Handles tokenization internally using the tokenizer directly
        (no chat template formatting).

        Args:
            prompt: Raw prompt string (serializable)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_samples: Number of samples to generate

        Returns:
            Dictionary with serializable results (same structure as sample_from_messages)
        """
        from tinker import types

        # Tokenize prompt directly (no chat template)
        prompt_tokens = self._tokenizer.encode(prompt, add_special_tokens=False)

        model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

        sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self._stop_sequences,
        )

        sample_future = self._sampling_client.sample(
            prompt=model_input,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=False,
        )

        sample_result = sample_future.result()

        # Process first sample
        sequence = sample_result.sequences[0]
        response_tokens = list(sequence.tokens)
        response_logprobs = list(sequence.logprobs) if sequence.logprobs else []

        # Decode response
        response = self._tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Determine finish reason
        finish_reason = "stop" if len(response_tokens) < max_tokens else "length"

        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "response_logprobs": response_logprobs,
            "finish_reason": finish_reason,
            "tool_string": None,
            "is_valid": True,
        }

    def compute_logprobs(self, tokens: list[int]) -> list[float | None]:
        """
        Compute log probabilities for token sequence.

        Args:
            tokens: Token IDs to compute logprobs for

        Returns:
            List of log probabilities
        """
        from tinker import types

        model_input = types.ModelInput.from_ints(tokens=tokens)
        logprobs_future = self._sampling_client.compute_logprobs(prompt=model_input)
        logprobs = logprobs_future.result()

        return list(logprobs) if logprobs else []

    def forward_backward(
        self,
        datums_data: list[dict],
        loss_fn: str = "importance_sampling",
        loss_fn_config: dict | None = None,
    ) -> dict:
        """
        Run forward-backward pass using the training client.

        Args:
            datums_data: List of datum dictionaries with serializable data
            loss_fn: Loss function name
            loss_fn_config: Optional loss function configuration

        Returns:
            Dictionary with loss and metrics
        """
        import torch
        from tinker import types
        from tinker.types.tensor_data import TensorData

        # Convert serializable dicts back to Datum objects
        datums = []
        for d in datums_data:
            loss_fn_inputs = {}
            for key, value in d["loss_fn_inputs"].items():
                loss_fn_inputs[key] = TensorData.from_torch(torch.tensor(value))

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=d["input_tokens"]),
                loss_fn_inputs=loss_fn_inputs,
            )
            datums.append(datum)

        # Call forward_backward
        fwd_bwd_future = self._training_client.forward_backward(
            datums, loss_fn=loss_fn, loss_fn_config=loss_fn_config
        )
        result = fwd_bwd_future.result()

        return {
            "loss": result.loss if hasattr(result, "loss") else 0.0,
            "metrics": dict(result.metrics) if hasattr(result, "metrics") else {},
        }

    def optim_step(
        self,
        learning_rate: float = 2e-6,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> dict:
        """
        Run optimizer step using the training client.

        Args:
            learning_rate: Learning rate
            beta1: Adam beta1
            beta2: Adam beta2
            eps: Adam epsilon

        Returns:
            Dictionary with optimizer step result
        """
        from tinker import types

        adam_params = types.AdamParams(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )

        optim_future = self._training_client.optim_step(adam_params)
        _ = optim_future.result()

        return {"status": "success"}

    def forward_backward_and_optim_step(
        self,
        datums_data: list[dict],
        loss_fn: str = "importance_sampling",
        loss_fn_config: dict | None = None,
        learning_rate: float = 2e-6,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> dict:
        """
        Run forward-backward pass and optimizer step together, waiting for both futures.

        This method calls both forward_backward and optim_step on the training client,
        gets both futures, and then waits for both results together (similar to rl_loop.py).
        This allows the operations to run in parallel when possible.

        Args:
            datums_data: List of datum dictionaries with serializable data
            loss_fn: Loss function name
            loss_fn_config: Optional loss function configuration
            learning_rate: Learning rate
            beta1: Adam beta1
            beta2: Adam beta2
            eps: Adam epsilon

        Returns:
            Dictionary with all metrics from forward_backward (no hard-coded "loss" field)
        """
        import torch
        from tinker import types
        from tinker.types.tensor_data import TensorData

        # Convert serializable dicts back to Datum objects
        datums = []
        for d in datums_data:
            loss_fn_inputs = {}
            for key, value in d["loss_fn_inputs"].items():
                loss_fn_inputs[key] = TensorData.from_torch(torch.tensor(value))

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=d["input_tokens"]),
                loss_fn_inputs=loss_fn_inputs,
            )
            datums.append(datum)

        # Call forward_backward and optim_step together to get futures
        fwd_bwd_future = self._training_client.forward_backward(
            datums, loss_fn=loss_fn, loss_fn_config=loss_fn_config
        )

        adam_params = types.AdamParams(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        optim_future = self._training_client.optim_step(adam_params)

        # Wait for both futures together
        fwd_bwd_result = fwd_bwd_future.result()
        _ = optim_future.result()

        # Return all metrics from forward_backward (no hard-coded "loss" field)
        return dict(fwd_bwd_result.metrics) if hasattr(fwd_bwd_result, "metrics") else {}

    def save_weights_for_sampler(self, name: str) -> str:
        """
        Save weights for sampler and return the path.

        Args:
            name: Name for the saved weights

        Returns:
            Path to saved weights (string)
        """
        save_result = self._training_client.save_weights_for_sampler(name=name).result()
        return save_result.path

    def get_current_sampling_path(self) -> str:
        """Get current sampling path."""
        return self._current_sampling_path

    def set_current_sampling_path(self, new_path: str) -> None:
        """Set current sampling path."""
        self._current_sampling_path = new_path

    def update_sampling_client(self) -> None:
        """Update sampling client with new weights from current path."""
        self._sampling_client = self._service_client.create_sampling_client(
            model_path=self._current_sampling_path
        )
        logger.info(f"Sampling client updated to path: {self._current_sampling_path}")

    def get_service_client(self):
        """Get the underlying Tinker ServiceClient."""
        return self._service_client

    def get_tokenizer(self):
        """Expose the tokenizer for downstream helpers that need decoding."""
        return self._tokenizer
