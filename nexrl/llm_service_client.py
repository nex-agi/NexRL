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
LLM Service Client for NexRL framework
"""

import logging
import time
import warnings
from typing import Any

import openai
from omegaconf import DictConfig
from transformers import AutoTokenizer

from .executor import execute

logger = logging.getLogger(__name__)

# Suppress httpx INFO logs (HTTP request logs)
logging.getLogger("httpx").setLevel(logging.WARNING)


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}")


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer.

    Args:
        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.
        **kwargs: The keyword arguments for the tokenizer.

    Returns:
        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107."
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    logger.debug(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    return tokenizer


class LLMServiceClient:
    """
    A service client for interacting with LLM APIs.
    Encapsulates OpenAI client and provides completion and generation methods.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the LLM service client.

        Args:
            config: Configuration containing LLM settings
        """
        self._config = config
        # Initialize OpenAI client based on config
        self._oai_llm = openai.OpenAI(
            api_key=config.inference_service.api_key,
            base_url=config.inference_service.base_url + "/v1",
            timeout=1000,
        )
        self._weight_sync_controller = None  # Will be set by rollout worker
        self._model_tag = config.inference_service.get("model_tag", "default")
        self._freeze_for_weight_sync = config.inference_service.get("freeze_for_weight_sync", True)

        # Initialize tokenizer using the hf_tokenizer utility
        tokenizer_path = config.inference_service.get("tokenizer", config.inference_service.model)
        self.tokenizer = hf_tokenizer(tokenizer_path)

    def completion(self, prompt: str, **kwargs) -> dict[str, Any]:
        """
        Call LLM inference through OpenAI completions API.
        Passes in prompt, and gets model, max_tokens, temperature and other parameters from kwargs.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters for completion

        Returns:
            dict[str, Any]: LLM response containing:
                - prompt: the input prompt
                - response: the generated text
                - finish_reason: why generation stopped
                - tokens: list of token IDs (prompt + response)
                - prompt_num_tokens: number of prompt tokens
                - loss_mask: list indicating which tokens to compute loss on
        """
        # Check weight sync status and block if necessary
        if self._freeze_for_weight_sync:
            self._wait_for_weight_sync()

        extra_body = kwargs.pop("extra_body", {})
        extra_body["include_stop_str_in_output"] = True
        extra_body["no_stop_trim"] = True
        extra_body["min_tokens"] = 10
        extra_body["skip_special_tokens"] = False
        extra_body["return_tokens_as_token_ids"] = True

        max_retries = self._config.inference_service.max_retries
        completion = None

        for _ in range(max_retries):
            try:
                completion = self._oai_llm.completions.create(
                    model=self._config.inference_service.model,
                    prompt=prompt,
                    max_tokens=self._config.inference_service.max_tokens,
                    temperature=self._config.temperature,
                    logprobs=True,  # Request logprobs to get token information
                    extra_body=extra_body,
                    **kwargs,
                )
                break
            except Exception as e:
                logger.error(f"Error in LLM completion: {e}")
                time.sleep(1)
                continue

        if completion is None:
            raise ValueError("Failed to get completion after all retries")
        if completion.choices[0].finish_reason == "length":
            logger.warning("OpenAI response is truncated")

        response = completion.choices[0].text

        logger.info(f"OpenAI response: {response}")

        if response is None:
            raise ValueError(f"OpenAI response is None, completion:{completion}")

        # Extract token information
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0].tolist()

        # Extract response tokens from logprobs
        if hasattr(completion.choices[0], "logprobs") and completion.choices[0].logprobs:
            backend = self._config.inference_service.get("backend", "vllm")
            if backend == "vllm":
                # in format of "token_id:{token_id}"
                response_tokens = [
                    int(_.split(":")[1])
                    for _ in completion.choices[0].logprobs.tokens
                    if _ is not None
                ]
            else:
                logger.info(f"OpenAI response logprobs: {completion.choices[0].logprobs}")
                response_tokens = [
                    int(_.token) for _ in completion.choices[0].logprobs.content if _ is not None
                ]
        else:
            raise ValueError(f"OpenAI response tokens is None, completion:{completion}")

        prompt_num_tokens = len(prompt_tokens)
        tokens = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)

        return {
            "prompt": prompt,
            "response": response,
            "finish_reason": completion.choices[0].finish_reason,
            "tokens": tokens,
            "prompt_num_tokens": prompt_num_tokens,
            "loss_mask": loss_mask,
            **kwargs,
        }

    def generate(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """
        Call LLM inference through OpenAI chat completions API.
        Passes in messages, and gets model, max_tokens, temperature and other parameters from kwargs.

        Args:
            messages: list of message dictionaries
            **kwargs: Additional parameters for generation

        Returns:
            dict[str, Any]: LLM response containing:
                - messages: the input messages
                - response: the generated text
                - tool_calls: any tool calls made
                - finish_reason: why generation stopped
                - response_tokens: list of response token IDs
                - prompt_tokens_decoded: list of decoded prompt tokens (currently empty)
        """
        # Check weight sync status and block if necessary
        if self._freeze_for_weight_sync:
            self._wait_for_weight_sync()

        extra_body = kwargs.pop("extra_body", {})
        extra_body["return_tokens_as_token_ids"] = True
        extra_body["skip_special_tokens"] = False
        extra_body["include_stop_str_in_output"] = True
        extra_body["min_tokens"] = 2

        max_retries = self._config.inference_service.max_retries
        completion = None
        for _ in range(max_retries):
            try:
                completion = self._oai_llm.chat.completions.create(
                    model=self._config.inference_service.model,
                    messages=messages,  # type: ignore  # mypy does not recognize this
                    max_tokens=self._config.inference_service.max_tokens,
                    temperature=self._config.temperature,
                    logprobs=True,  # Request logprobs to get token information
                    extra_body=extra_body,
                    **kwargs,
                )
                break
            except Exception as e:
                logger.error(f"Error in LLM generation: {e}")
                time.sleep(1)
                continue

        if completion is None:
            raise ValueError("Failed to get completion after all retries")

        response = completion.choices[0].message.content

        tool_calls = completion.choices[0].message.tool_calls
        if response is None and tool_calls is None:
            raise ValueError(f"OpenAI response and tool_calls is None, messages:{completion}")
        if response is None:
            response = ""

        try:
            if hasattr(completion.choices[0], "logprobs") and completion.choices[0].logprobs:
                backend = self._config.inference_service.get("backend", "vllm")
                if backend == "vllm":
                    # in format of "token_id:{token_id}"
                    response_tokens = [
                        int(_.token.split(":")[1])
                        for _ in completion.choices[0].logprobs.content
                        if _ is not None
                    ]
                else:
                    if hasattr(completion.choices[0].logprobs, "tokens"):
                        response_tokens = [
                            int(_) for _ in completion.choices[0].logprobs.tokens if _ is not None
                        ]
                    elif hasattr(completion.choices[0].logprobs, "content"):
                        response_tokens = [
                            int(_.token)
                            for _ in completion.choices[0].logprobs.content
                            if _ is not None
                        ]
                    else:
                        raise ValueError(
                            f"Unknown logprobs format: {completion.choices[0].logprobs}"
                        )
            else:
                response_tokens = []
        except Exception as e:
            logger.error(f"Error extracting response tokens: {e}")
            response_tokens = []

        return {
            "response": response,
            "tool_calls": tool_calls,
            "finish_reason": completion.choices[0].finish_reason,
            "response_tokens": response_tokens,
            **kwargs,
        }

    def set_weight_sync_controller(self, weight_sync_controller) -> None:
        """
        Set reference to weight manager for checking sync status

        Args:
            weight_sync_controller: WeightSyncController instance
        """
        self._weight_sync_controller = weight_sync_controller

    def _wait_for_weight_sync(self) -> None:
        """
        Wait for weight synchronization to complete before proceeding with LLM calls.
        Blocks until rollout service status is 'running'.
        """
        assert self._weight_sync_controller is not None

        max_wait_seconds = 60  # Maximum time to wait
        wait_interval = 0.1  # Sleep interval between checks
        total_waited = 0

        while total_waited < max_wait_seconds:
            status = execute(
                self._weight_sync_controller.check_rollout_service_status, self._model_tag
            )

            if status == "continue":
                return  # Can proceed
            elif status == "block":
                time.sleep(wait_interval)
                total_waited += wait_interval
            else:
                raise ValueError(f"Unknown status from weight manager: {status}")

        # If we've waited too long, log a warning but proceed
        raise RuntimeError(
            f"LLM service timed out after waiting {max_wait_seconds}s for weight sync"
        )
