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
Remote API Inference Service Client - Base class for remote API backends (Tinker/Weaver)
"""

import json
import logging
import time
import uuid
from abc import abstractmethod
from typing import Any

from omegaconf import DictConfig

from ..executor import execute
from ..utils.reasoning_parser import create_reasoning_parser
from ..utils.tool_parser import create_tool_parser
from .base_inference_service_client import InferenceServiceClient

logger = logging.getLogger(__name__)


class RemoteApiInferenceServiceClient(InferenceServiceClient):
    """
    Base inference service client for remote API backends.

    This client is a thin wrapper that passes serializable data (strings, dicts)
    to service holders (Tinker/Weaver), which own the tokenizer and renderer.

    Derived classes should:
    - Set _service_holder in their __init__ or setter method
    - Optionally override methods for custom behavior
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the remote API inference service client.

        Args:
            config: Configuration containing LLM settings
        """
        super().__init__()
        self._config = config
        self._service_holder = None  # Set by derived class
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

        # Initialize tool parser based on config
        parser_type = config.inference_service.get("tool_parser", "qwen25")
        try:
            self._tool_parser = create_tool_parser(parser_type)
            logger.info(f"Initialized tool parser: {parser_type}")
        except ValueError as e:
            logger.warning(f"Failed to create tool parser: {e}. Using qwen25 as fallback.")
            self._tool_parser = create_tool_parser("qwen25")

        # Initialize reasoning parser based on config
        reasoning_parser_type = config.inference_service.get("reasoning_parser", "think_tag")
        self._reasoning_parser = create_reasoning_parser(reasoning_parser_type)
        logger.info(f"Initialized reasoning parser: {reasoning_parser_type}")

    @abstractmethod
    def set_service_holder(self, service_holder) -> None:
        """
        Set the service holder (must be implemented by derived classes).

        Args:
            service_holder: Service holder instance (TinkerServiceHolder, WeaverServiceHolder, etc.)
        """

    def _make_id(self, prefix: str) -> str:
        """Generate a unique identifier with prefix (mimics OpenAI ids)."""
        return f"{prefix}-{uuid.uuid4().hex}"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        """
        Apply chat template to messages using service holder.

        Args:
            messages: List of message dictionaries (chat format)
            tools: List of tool dictionaries (None or empty list)
            add_generation_prompt: Whether to add the generation prompt
            tokenize: If True, return token IDs; if False, return string

        Returns:
            str | list[int]: Formatted prompt string or token IDs
        """
        assert self._service_holder is not None, "ServiceHolder not set"

        # Normalize tools to empty list if None
        if tools is None:
            tools = []

        return execute(
            self._service_holder.apply_chat_template,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            tools=tools,
        )

    def completion(self, prompt: str, **kwargs) -> dict[str, Any]:
        """
        Call LLM inference for prompt completion using service's sample API and
        return an OpenAI-style text completion payload.

        Args:
            prompt: Input prompt string (serializable)
            **kwargs: Additional parameters (passed through to result)

        Returns:
            dict[str, Any]: LLM response in NexRL format
        """
        assert self._service_holder is not None, "ServiceHolder not set"

        # Check weight sync status and block if necessary
        if self._freeze_for_weight_sync:
            self._wait_for_weight_sync()

        max_tokens = self._config.inference_service.max_tokens
        temperature = self._config.temperature

        result = execute(
            self._service_holder.sample_from_prompt,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            num_samples=1,
        )

        prompt_tokens: list[int] = result["prompt_tokens"]
        response_tokens: list[int] = result["response_tokens"]
        response_logprobs: list[float] = result["response_logprobs"]

        openai_like = {
            "id": self._make_id("cmpl"),
            "object": "text_completion",
            "created": int(time.time()),
            "model": self._config.inference_service.get("model", "remote-api"),
            "choices": [
                {
                    "index": 0,
                    "text": result["response"],
                    "finish_reason": result["finish_reason"],
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(response_tokens),
                "total_tokens": len(prompt_tokens) + len(response_tokens),
            },
            "nexrl_train": {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_logprobs": response_logprobs,
            },
        }

        return {**openai_like, **kwargs}

    def generate(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """
        Call LLM inference for chat completion using service's sample API.

        Args:
            messages: List of message dictionaries (chat format, serializable)
            **kwargs: Additional parameters (passed through to result)

        Returns:
            dict[str, Any]: LLM response in NexRL format
        """
        assert self._service_holder is not None, "ServiceHolder not set"

        # Check weight sync status and block if necessary
        if self._freeze_for_weight_sync:
            self._wait_for_weight_sync()

        max_tokens = self._config.inference_service.max_tokens
        temperature = self._config.temperature

        # Extract tools from kwargs if present
        tools = kwargs.pop("tools", [])

        result = execute(
            self._service_holder.sample_from_messages,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            num_samples=1,
            tools=tools,
        )

        # Log validation status if available
        if "is_valid" in result:
            if not result["is_valid"]:
                logger.warning(
                    f"Response parsing indicated invalid format: {result['response'][:100]}..."
                )
            else:
                logger.debug(f"Response is valid: {result['response'][:100]}...")

        prompt_tokens: list[int] = result["prompt_tokens"]
        response_tokens: list[int] = result["response_tokens"]
        response_logprobs: list[float] = result["response_logprobs"]
        tool_string: str | None = result.get("tool_string")
        reasoning_string: str | None = result.get("reasoning_string")

        # Response content is already cleaned by service holder
        response_content = result["response"]
        reasoning_content = None

        # Parse reasoning to extract structured content (if present)
        if reasoning_string:
            reasoning_result = self._reasoning_parser.parse(reasoning_string)
            if reasoning_result.is_valid:
                reasoning_content = reasoning_result.reasoning_content
                logger.debug(
                    f"Extracted reasoning: {reasoning_content[:100] if reasoning_content else 'None'}..."
                )
            elif not reasoning_result.is_valid:
                logger.warning("Failed to parse reasoning content")

        # Parse tool calls to structured format (if present)
        tool_calls = None
        if tool_string:
            parse_result = self._tool_parser.parse(tool_string)
            if parse_result.is_valid and parse_result.tool_calls:
                # Convert to OpenAI format and ensure arguments is JSON string
                tool_calls = []
                for tc in parse_result.tool_calls:
                    # Ensure function.arguments is a JSON string, not a dict
                    function = dict(tc.function)
                    if "arguments" in function and isinstance(function["arguments"], dict):
                        function["arguments"] = json.dumps(function["arguments"])
                    elif "arguments" not in function:
                        function["arguments"] = "{}"

                    tool_calls.append(
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": function,
                        }
                    )

                logger.debug(f"Parsed tool calls: {json.dumps(tool_calls, indent=2)}")
            elif not parse_result.is_valid:
                logger.warning(f"Failed to parse tool call from: {tool_string[:100]}...")

        # Build message with cleaned content
        message = {
            "role": "assistant",
            "content": response_content,
        }

        # Add reasoning_content if present
        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        # Add tool_calls if present
        if tool_calls:
            message["tool_calls"] = tool_calls

        openai_like = {
            "id": self._make_id("chatcmpl"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._config.inference_service.get("model", "remote-api"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": result["finish_reason"],
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(response_tokens),
                "total_tokens": len(prompt_tokens) + len(response_tokens),
            },
            "nexrl_train": {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_logprobs": response_logprobs,
            },
        }

        return {**openai_like, **kwargs}
