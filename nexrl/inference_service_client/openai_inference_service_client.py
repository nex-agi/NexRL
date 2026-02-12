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
OpenAI-compatible Inference Service Client for NexRL framework
"""

import json
import logging
import threading
import time
import traceback
from copy import deepcopy
from typing import Any

import openai
from omegaconf import DictConfig

from ..utils.url_utils import ensure_url_scheme
from .base_inference_service_client import InferenceServiceClient, hf_tokenizer

logger = logging.getLogger(__name__)


# Module-level tokenizer cache (shared within each Python process)
# In local mode: all workers share this cache (same process)
# In Ray mode: each actor has its own cache (separate processes)
_TOKENIZER_CACHE: dict[str, Any] = {}
_TOKENIZER_LOCK = threading.Lock()


def _get_cached_tokenizer(tokenizer_path: str):
    """
    Get or create a cached tokenizer (process-local singleton).

    This cache is shared across all clients in the same Python process:
    - Local mode: All rollout workers share one tokenizer
    - Ray mode: Each actor process has one tokenizer

    Args:
        tokenizer_path: Path to the tokenizer

    Returns:
        Cached tokenizer instance
    """
    with _TOKENIZER_LOCK:
        if tokenizer_path not in _TOKENIZER_CACHE:
            logger.info(
                f"Loading tokenizer for path: {tokenizer_path} (first time in this process)"
            )
            _TOKENIZER_CACHE[tokenizer_path] = hf_tokenizer(tokenizer_path)
        else:
            logger.debug(f"Reusing cached tokenizer for path: {tokenizer_path}")
        return _TOKENIZER_CACHE[tokenizer_path]


class OpenAIInferenceServiceClient(InferenceServiceClient):
    """
    An inference service client using OpenAI-compatible APIs.
    Encapsulates OpenAI client and provides completion and generation methods.
    """

    # Class-level tokenizer cache to share tokenizers across instances
    _tokenizer_cache: dict[str, Any] = {}

    def __init__(self, config: DictConfig):
        """
        Initialize the OpenAI inference service client.

        Args:
            config: Configuration containing LLM settings
        """
        super().__init__()
        self._config = config
        # identifier serves as model_tag for weight sync coordination
        # Support both old 'model_tag' and new 'identifier' fields
        identifier = config.inference_service.get("identifier")
        model_tag = config.inference_service.get("model_tag")

        if identifier is None and model_tag is not None:
            import warnings

            warnings.warn(
                "The 'model_tag' field is deprecated. Please use 'identifier' instead. "
                "See migration guide in docs/developer-guide/09-recipes/.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._identifier = model_tag
        else:
            self._identifier = identifier or "default"
        self._freeze_for_weight_sync = config.inference_service.get("freeze_for_weight_sync", True)
        self._parse_tool_call_arguments = config.inference_service.get(
            "parse_tool_call_arguments", False
        )

        # Initialize OpenAI client based on config
        # Ensure base_url has proper http:// scheme
        base_url = ensure_url_scheme(config.inference_service.base_url)
        if not base_url:
            raise ValueError("base_url is required for OpenAI inference service")
        self._oai_llm = openai.OpenAI(
            api_key=config.inference_service.api_key,
            base_url=base_url + "/v1",
            timeout=1000,
        )

        # Initialize tokenizer using cached tokenizer (shared within process)
        tokenizer_path = config.inference_service.get("tokenizer", config.inference_service.model)
        self.tokenizer = _get_cached_tokenizer(tokenizer_path)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        """
        Apply chat template to messages using the tokenizer.

        Args:
            messages: List of message dictionaries (chat format)
            tools: List of tool dictionaries
            add_generation_prompt: Whether to add the generation prompt
            tokenize: If True, return token IDs; if False, return string

        Returns:
            str | list[int]: Formatted prompt string or token IDs
        """
        try:
            messages_copy = deepcopy(messages)

            # Parse tool call arguments if configured to do so
            if self._parse_tool_call_arguments:
                for message in messages_copy:
                    tool_calls = message.get("tool_calls")
                    if not tool_calls:
                        continue
                    for tool_call in tool_calls:
                        function = tool_call.get("function")
                        arguments = function.get("arguments")
                        if not arguments:
                            continue
                        if isinstance(arguments, str):
                            if arguments != "":
                                try:
                                    function["arguments"] = json.loads(arguments, strict=False)
                                except json.JSONDecodeError as e:
                                    logger.warning(
                                        f"Failed to parse tool call arguments as JSON: {e}, "
                                        f"keeping as string: {arguments[:100]}"
                                    )
                                    # Keep as string if JSON parsing fails
                            else:
                                function["arguments"] = {}
                        elif arguments is None:
                            function["arguments"] = {}
                        # If arguments is already a dict, leave it as is

            token_ids = self.tokenizer.apply_chat_template(
                messages_copy,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                tools=tools,
            )
            return token_ids
        except Exception as e:
            logger.error(f"Error in apply_chat_template: {e}")
            logger.error(traceback.format_exc())
            raise e

    def completion(self, prompt: str, **kwargs) -> dict[str, Any]:
        """
        Call LLM inference through OpenAI completions API.
        Passes in prompt, and gets model, max_tokens, temperature and other parameters from kwargs.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters for completion

        Returns:
            dict[str, Any]: LLM response dict containing all completion fields plus:
                - nexrl_train: dict with prompt_tokens, response_tokens, response_logprobs
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

        kwargs.pop("model", None)  # Remove model if present, don't error if missing

        for _ in range(max_retries):
            try:
                kwargs.pop("logprobs", None)  # Remove logprobs if present, don't error if missing
                completion = self._oai_llm.completions.create(
                    model=self._config.inference_service.model,
                    prompt=prompt,
                    max_tokens=kwargs.pop("max_tokens", self._config.inference_service.max_tokens),
                    temperature=kwargs.pop("temperature", self._config.temperature),
                    logprobs=True,
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

        logger.debug(f"OpenAI response: {response}")

        if response is None:
            raise ValueError(f"OpenAI response is None, completion:{completion}")

        # Extract token information for training (nexrl_train)
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0].tolist()

        response_tokens: list[int] = []
        response_logprobs: list[float] = []
        if hasattr(completion.choices[0], "logprobs") and completion.choices[0].logprobs:
            backend = self._config.inference_service.get("backend", "vllm")
            if backend == "vllm":
                response_tokens = [
                    int(_.split(":")[1])
                    for _ in completion.choices[0].logprobs.tokens
                    if _ is not None
                ]
                response_logprobs = [
                    float(lp)
                    for lp in completion.choices[0].logprobs.token_logprobs
                    if lp is not None
                ]
            else:
                logger.debug(f"OpenAI response logprobs: {completion.choices[0].logprobs}")
                response_tokens = [
                    int(_.token) for _ in completion.choices[0].logprobs.content if _ is not None
                ]
                response_logprobs = [
                    float(_.logprob)
                    for _ in completion.choices[0].logprobs.content
                    if _ is not None
                ]
        else:
            raise ValueError(f"OpenAI response tokens is None, completion:{completion}")

        out: dict[str, Any] = {
            **completion.model_dump(),
            "nexrl_train": {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_logprobs": response_logprobs,
            },
        }
        return out

    def generate(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """
        Call LLM inference through OpenAI chat completions API.
        Passes in messages, and gets model, max_tokens, temperature and other parameters from kwargs.

        Args:
            messages: list of message dictionaries
            **kwargs: Additional parameters for generation

        Returns:
            dict[str, Any]: LLM response dict containing all completion fields plus:
                - nexrl_train: dict with prompt_tokens, response_tokens, response_logprobs
        """
        # Check weight sync status and block if necessary
        if self._freeze_for_weight_sync:
            self._wait_for_weight_sync()

        tools = kwargs.get("tools", [])
        extra_body = kwargs.pop("extra_body", {})
        extra_body["return_tokens_as_token_ids"] = True
        extra_body["skip_special_tokens"] = False
        extra_body["include_stop_str_in_output"] = True
        extra_body["min_tokens"] = 2

        kwargs.pop("model", None)  # Remove model if present, don't error if missing

        max_retries = self._config.inference_service.max_retries
        completion = None
        for _ in range(max_retries):
            try:
                kwargs.pop("logprobs", None)  # Remove logprobs if present, don't error if missing
                completion = self._oai_llm.chat.completions.create(
                    model=self._config.inference_service.model,
                    messages=messages,  # type: ignore  # mypy does not recognize this
                    max_tokens=kwargs.pop("max_tokens", self._config.inference_service.max_tokens),
                    temperature=kwargs.pop("temperature", self._config.temperature),
                    logprobs=True,
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

        # Build nexrl_train fields (tokens/logprobs) from OpenAI response
        prompt_tokens = self.apply_chat_template(messages, tools=tools, tokenize=True)  # type: ignore

        response_tokens: list[int] = []
        response_logprobs: list[float] = []
        try:
            if hasattr(completion.choices[0], "logprobs") and completion.choices[0].logprobs:
                backend = self._config.inference_service.get("backend", "vllm")
                if backend == "vllm":
                    response_tokens = [
                        int(_.token.split(":")[1])
                        for _ in completion.choices[0].logprobs.content
                        if _ is not None
                    ]
                    response_logprobs = [
                        float(_.logprob)
                        for _ in completion.choices[0].logprobs.content
                        if _ is not None and _.logprob is not None
                    ]
                else:
                    if hasattr(completion.choices[0].logprobs, "tokens"):
                        response_tokens = [
                            int(_) for _ in completion.choices[0].logprobs.tokens if _ is not None
                        ]
                        response_logprobs = [
                            float(lp)
                            for lp in completion.choices[0].logprobs.token_logprobs
                            if lp is not None
                        ]
                    elif hasattr(completion.choices[0].logprobs, "content"):
                        response_tokens = [
                            int(_.token)
                            for _ in completion.choices[0].logprobs.content
                            if _ is not None
                        ]
                        response_logprobs = [
                            float(_.logprob)
                            for _ in completion.choices[0].logprobs.content
                            if _ is not None and _.logprob is not None
                        ]
            else:
                logger.warning(
                    "OpenAI chat completion missing logprobs; nexrl_train.response_logprobs will be empty"
                )
        except Exception as e:
            logger.error(f"Error extracting response tokens/logprobs: {e}")

        out: dict[str, Any] = {
            **completion.model_dump(),
            "nexrl_train": {
                "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, list) else [],
                "response_tokens": response_tokens,
                "response_logprobs": response_logprobs,
            },
        }
        return out
