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
Tests for GRPOProcessor
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from nexrl.algorithm_processor.grpo_processor import GRPOProcessor
from nexrl.mock import MockTrainServiceClient
from nexrl.nexrl_types import Batch


@pytest.fixture
def grpo_config():
    """Configuration for GRPOProcessor testing"""
    return OmegaConf.create(
        {
            "batch_size": 4,
            "model_tag": "test_model",
            "use_kl_in_reward": False,
            "do_old_log_prob_compute": False,
            "inference_service": {
                "model": "mock_model",
                "tokenizer": None,
            },
            "train_service": {
                "backend": "mock",
                "url": "http://localhost:8080",
                "identifier": "test_grpo",
            },
        }
    )


@pytest.fixture
def grpo_processor(grpo_config):
    """Create GRPOProcessor with mock dependencies"""
    # Mock the tokenizer loading
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    with patch(
        "nexrl.algorithm_processor.grpo_processor.hf_tokenizer", return_value=mock_tokenizer
    ):
        with patch(
            "nexrl.algorithm_processor.grpo_processor.create_train_service_client",
            return_value=MockTrainServiceClient("test", "test"),
        ):
            processor = GRPOProcessor(grpo_config)

    return processor


def test_grpo_processor_initialization(grpo_processor):
    """Test GRPOProcessor initialization"""
    assert grpo_processor._config is not None
    assert grpo_processor._pad_token_id == 0
    assert grpo_processor._train_service_client is not None


def test_grpo_processor_reward_fn(grpo_processor):
    """Test GRPO reward function"""
    # Create a simple batch
    batch_size = 2
    response_length = 4

    batch_data = {
        "responses": torch.ones(batch_size, response_length, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 8),  # 4 prompt + 4 response tokens
        "reward": torch.tensor([1.0, 2.0]),
    }
    batch = Batch(values=batch_data, metadata={"batch_size": batch_size})

    reward_tensor = grpo_processor._reward_fn(batch)

    # Check shape
    assert reward_tensor.shape == (batch_size, response_length)

    # Check that reward is assigned to last token of each response
    # First sample should have reward 1.0 at last position
    assert reward_tensor[0, -1].item() == 1.0
    # Second sample should have reward 2.0 at last position
    assert reward_tensor[1, -1].item() == 2.0


def test_grpo_processor_compute_advantage(grpo_processor):
    """Test GRPO advantage computation"""
    batch_size = 4
    response_length = 5

    batch_data = {
        "responses": torch.ones(batch_size, response_length, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 10),  # 5 prompt + 5 response
        "token_level_rewards": torch.rand(batch_size, response_length),
        "group_id": [0, 0, 1, 1],  # Two groups - use list instead of tensor
    }
    batch = Batch(values=batch_data, metadata={"batch_size": batch_size})

    batch = grpo_processor._compute_advantage(batch)

    # Check that advantages and returns are computed
    assert "advantages" in batch.values
    assert "returns" in batch.values
    assert "grpo_std" in batch.metadata

    # Check shapes
    assert batch.values["advantages"].shape == (batch_size, response_length)
    assert batch.values["returns"].shape == (batch_size, response_length)


def test_grpo_processor_cal_reward_kl(grpo_processor):
    """Test reward KL calculation (when not using KL in reward)"""
    batch_size = 2
    response_length = 4

    batch_data = {
        "responses": torch.ones(batch_size, response_length, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 8),
    }
    batch = Batch(values=batch_data, metadata={"batch_size": batch_size})

    metrics = grpo_processor._cal_reward_kl(batch)

    assert "critic/kl" in metrics
    assert "critic/kl_coeff" in metrics
    assert metrics["critic/kl_coeff"] == 0


def test_grpo_processor_init_kl_controller_fixed(grpo_config):
    """Test KL controller initialization with fixed type"""
    grpo_config.use_kl_in_reward = True
    grpo_config.critic = {"kl_ctrl": {"type": "fixed", "kl_reward_coef": 0.5}}

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    with patch(
        "nexrl.algorithm_processor.grpo_processor.hf_tokenizer", return_value=mock_tokenizer
    ):
        with patch(
            "nexrl.algorithm_processor.grpo_processor.create_train_service_client",
            return_value=MockTrainServiceClient("test", "test"),
        ):
            processor = GRPOProcessor(grpo_config)

    assert processor._kl_ctrl_in_reward is not None
    assert processor._kl_ctrl_in_reward.value == 0.5


def test_grpo_processor_init_kl_controller_adaptive(grpo_config):
    """Test KL controller initialization with adaptive type"""
    grpo_config.use_kl_in_reward = True
    grpo_config.critic = {
        "kl_ctrl": {
            "type": "adaptive",
            "kl_reward_coef": 0.1,
            "target_kl": 6.0,
            "horizon": 10000,
        }
    }

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    with patch(
        "nexrl.algorithm_processor.grpo_processor.hf_tokenizer", return_value=mock_tokenizer
    ):
        with patch(
            "nexrl.algorithm_processor.grpo_processor.create_train_service_client",
            return_value=MockTrainServiceClient("test", "test"),
        ):
            processor = GRPOProcessor(grpo_config)

    assert processor._kl_ctrl_in_reward is not None
    assert processor._kl_ctrl_in_reward.value == 0.1
    assert processor._kl_ctrl_in_reward.target == 6.0
