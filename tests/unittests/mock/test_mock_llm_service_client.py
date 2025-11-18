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
Tests for MockLLMServiceClient
"""

from nexrl.mock import MockLLMServiceClient


def test_mock_llm_service_client_initialization(rollout_worker_config):
    """Test MockLLMServiceClient initialization"""
    client = MockLLMServiceClient(rollout_worker_config)

    assert client._config is not None
    assert client.tokenizer is not None
    assert client._model_tag == "default"


def test_mock_tokenizer():
    """Test mock tokenizer functionality"""
    from nexrl.mock.mock_llm_service_client import MockTokenizer

    tokenizer = MockTokenizer()

    # Test attributes
    assert tokenizer.pad_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.vocab_size == 100000

    # Test encode
    tokens = tokenizer.encode("Hello world")
    assert isinstance(tokens, list)
    assert len(tokens) > 0

    # Test decode
    text = tokenizer.decode([1, 2, 3])
    assert isinstance(text, str)
    assert "Mock decoded" in text


def test_mock_llm_completion(rollout_worker_config):
    """Test mock LLM completion method"""
    client = MockLLMServiceClient(rollout_worker_config)

    prompt = "What is 2 + 2?"
    result = client.completion(prompt)

    # Check response structure
    assert "prompt" in result
    assert "response" in result
    assert "finish_reason" in result
    assert "usage" in result

    # Check values
    assert result["prompt"] == prompt
    assert isinstance(result["response"], str)
    assert result["finish_reason"] == "stop"
    assert "prompt_tokens" in result["usage"]
    assert "completion_tokens" in result["usage"]


def test_mock_llm_generate(rollout_worker_config):
    """Test mock LLM generate method (chat completion)"""
    client = MockLLMServiceClient(rollout_worker_config)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    result = client.generate(messages)

    # Check response structure
    assert "messages" in result
    assert "response" in result
    assert "finish_reason" in result
    assert "usage" in result

    # Check values
    assert result["messages"] == messages
    assert isinstance(result["response"], str)
    assert result["finish_reason"] == "stop"


def test_mock_llm_set_weight_sync_controller(rollout_worker_config):
    """Test setting weight sync controller"""
    client = MockLLMServiceClient(rollout_worker_config)

    class MockWeightSyncController:
        pass

    controller = MockWeightSyncController()
    client.set_weight_sync_controller(controller)

    assert client._weight_sync_controller is controller


def test_mock_llm_wait_for_weight_sync(rollout_worker_config):
    """Test wait for weight sync (should do nothing in mock)"""
    client = MockLLMServiceClient(rollout_worker_config)

    # Should not raise any errors
    client.wait_for_weight_sync(step=10)
