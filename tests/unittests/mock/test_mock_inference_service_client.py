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
Tests for MockInferenceServiceClient
"""

from nexrl.mock import MockInferenceServiceClient
from nexrl.mock.mock_inference_service_client import MockTokenizer


def test_mock_inference_service_client_initialization(rollout_worker_config):
    """Test MockInferenceServiceClient initialization"""
    client = MockInferenceServiceClient(rollout_worker_config)

    assert client._config is not None
    assert client.tokenizer is not None
    assert client._model_tag == "default"


def test_mock_tokenizer():
    """Test mock tokenizer functionality"""
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


def test_mock_inference_completion(rollout_worker_config):
    """Test mock inference completion method"""
    client = MockInferenceServiceClient(rollout_worker_config)

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


def test_mock_inference_generate(rollout_worker_config):
    """Test mock inference generate method (chat completion)"""
    client = MockInferenceServiceClient(rollout_worker_config)

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


def test_mock_inference_set_weight_sync_controller(rollout_worker_config):
    """Test setting weight sync controller"""
    client = MockInferenceServiceClient(rollout_worker_config)

    class MockWeightSyncController:
        pass

    controller = MockWeightSyncController()
    client.set_weight_sync_controller(controller)

    assert client._weight_sync_controller is controller
