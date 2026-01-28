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
Mock Inference Service Client for testing purposes
"""

import logging
from typing import Any

from omegaconf import DictConfig

from ..inference_service_client import InferenceServiceClient

logger = logging.getLogger(__name__)


class MockTokenizer:
    """Simple mock tokenizer for testing"""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.vocab_size = 100000

    # pylint: disable=unused-argument
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Mock encode method - returns dummy token IDs"""
        # Simple mock: return token IDs based on text length
        return list(range(len(text) // 5 + 1))

    # pylint: disable=unused-argument
    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Mock decode method - returns dummy text"""
        return f"Mock decoded text with {len(token_ids)} tokens"

    def apply_chat_template(  # pylint: disable=unused-argument
        self,
        messages,
        tools=None,
        add_generation_prompt=False,
        tokenize=False,
        add_special_tokens=True,
    ):
        """Mock apply_chat_template method for chat completion"""
        # Convert messages to a simple string format
        if isinstance(messages, list):
            # Format: "role: content\nrole: content"
            formatted = "\n".join(
                [f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages]
            )
        else:
            formatted = str(messages)

        if add_generation_prompt:
            formatted += "\nassistant:"

        if tokenize:
            return self.encode(formatted, add_special_tokens=add_special_tokens)
        return formatted

    def __call__(self, text, **kwargs):
        """Mock tokenizer call"""
        tokens = self.encode(text)
        return {"input_ids": tokens}


class MockInferenceServiceClient(InferenceServiceClient):
    """
    Mock Inference Service Client for testing purposes.
    Implements InferenceServiceClient interface with simple mock implementations
    without requiring network access or real tokenizers.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the mock inference service client.

        Args:
            config: Configuration containing LLM settings
        """
        super().__init__()
        self._config = config
        # identifier serves as model_tag for weight sync coordination
        self._identifier = config.inference_service.get("identifier", "default")
        self._freeze_for_weight_sync = config.inference_service.get("freeze_for_weight_sync", True)

        # Use mock tokenizer instead of loading from HuggingFace
        self.tokenizer = MockTokenizer()
        logger.info("MockInferenceServiceClient initialized with mock tokenizer")

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        """
        Apply chat template to messages using the mock tokenizer.

        Args:
            messages: List of message dictionaries (chat format)
            tools: List of tool dictionaries
            add_generation_prompt: Whether to add the generation prompt
            tokenize: If True, return token IDs; if False, return string

        Returns:
            str | list[int]: Formatted prompt string or token IDs
        """
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            tools=tools,
        )

    def completion(self, prompt: str, **kwargs) -> dict[str, Any]:
        """
        Mock completion method that returns a simple response.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (ignored in mock)

        Returns:
            dict[str, Any]: Mock LLM response
        """
        logger.debug(f"Mock completion called with prompt length: {len(prompt)}")

        # Generate a simple mock response
        mock_response = f"Mock response to: {prompt[:50]}..."

        return {
            "prompt": prompt,
            "response": mock_response,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(mock_response) // 4,
                "total_tokens": (len(prompt) + len(mock_response)) // 4,
            },
        }

    def generate(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """
        Mock generate method for chat completion.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters (ignored in mock)

        Returns:
            dict[str, Any]: Mock LLM response
        """
        logger.debug(f"Mock generate called with {len(messages)} messages")

        # Extract the last user message for context
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        last_message = user_messages[-1]["content"] if user_messages else "No message"

        # Generate a simple mock response
        mock_response = f"Mock chat response to: {last_message[:50]}..."

        return {
            "messages": messages,
            "response": mock_response,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": sum(len(msg.get("content", "")) for msg in messages) // 4,
                "completion_tokens": len(mock_response) // 4,
                "total_tokens": (
                    sum(len(msg.get("content", "")) for msg in messages) + len(mock_response)
                )
                // 4,
            },
        }
