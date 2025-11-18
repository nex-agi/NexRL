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
Mock LLM Service Client for testing purposes
"""

import logging
from typing import Any

from omegaconf import DictConfig

from ..llm_service_client import LLMServiceClient

logger = logging.getLogger(__name__)


class MockTokenizer:
    """Simple mock tokenizer for testing"""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.vocab_size = 100000

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Mock encode method - returns dummy token IDs"""
        # Simple mock: return token IDs based on text length
        return list(range(len(text) // 5 + 1))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Mock decode method - returns dummy text"""
        return f"Mock decoded text with {len(token_ids)} tokens"

    def apply_chat_template(
        self, messages, add_generation_prompt=False, tokenize=False, add_special_tokens=True
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


class MockLLMServiceClient(LLMServiceClient):
    """
    Mock LLM Service Client for testing purposes.
    Inherits from LLMServiceClient but provides simple mock implementations
    without requiring network access or real tokenizers.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the mock LLM service client.

        Args:
            config: Configuration containing LLM settings
        """
        # Store config without calling parent __init__ to avoid tokenizer loading
        self._config = config
        self._oai_llm = None  # No real OpenAI client needed
        self._weight_sync_controller = None
        self._model_tag = config.inference_service.get("model_tag", "default")
        self._freeze_for_weight_sync = config.inference_service.get("freeze_for_weight_sync", True)

        # Use mock tokenizer instead of loading from HuggingFace
        self.tokenizer = MockTokenizer()
        logger.info("MockLLMServiceClient initialized with mock tokenizer")

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

    def set_weight_sync_controller(self, controller) -> None:
        """
        Set the weight sync controller.

        Args:
            controller: Weight sync controller instance
        """
        self._weight_sync_controller = controller
        logger.info("Mock weight sync controller set")

    def wait_for_weight_sync(self, step: int) -> None:
        """
        Mock implementation of waiting for weight sync.

        Args:
            step: Training step number
        """
        logger.debug(f"Mock wait_for_weight_sync called for step {step}")
        # In mock, we don't actually wait
        pass
