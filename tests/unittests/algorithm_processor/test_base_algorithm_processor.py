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
Tests for BaseAlgorithmProcessor using MockAlgorithmProcessor
"""

import pytest
import torch
from omegaconf import OmegaConf

from nexrl.algorithm_processor import BaseAlgorithmProcessor
from nexrl.mock import MockAlgorithmProcessor
from nexrl.nexrl_types import Batch


@pytest.fixture
def algorithm_processor_config():
    """Configuration for algorithm processor testing"""
    return OmegaConf.create(
        {
            "batch_size": 4,
            "model_tag": "test_model",
        }
    )


def test_base_algorithm_processor_initialization(algorithm_processor_config):
    """Test BaseAlgorithmProcessor initialization via MockAlgorithmProcessor"""
    processor = MockAlgorithmProcessor(algorithm_processor_config)

    assert processor._config is not None
    assert processor._stop_event is not None
    assert processor._batch_count == 0


def test_base_algorithm_processor_module_name(algorithm_processor_config):
    """Test algorithm processor module name functionality"""
    processor = MockAlgorithmProcessor(algorithm_processor_config)

    processor.set_module_name("test_processor")
    assert processor.get_module_name() == "test_processor"


def test_base_algorithm_processor_health_check(algorithm_processor_config):
    """Test algorithm processor health check"""
    processor = MockAlgorithmProcessor(algorithm_processor_config)
    assert processor.health_check() is True


def test_base_algorithm_processor_set_module_references(algorithm_processor_config):
    """Test setting module references"""
    from nexrl.train_batch_pool import TrainBatchPool
    from nexrl.trajectory_pool import TrajectoryPool

    processor = MockAlgorithmProcessor(algorithm_processor_config)

    trajectory_pool = TrajectoryPool(OmegaConf.create({"max_size": 100}))
    train_batch_pool = TrainBatchPool(
        OmegaConf.create({"max_size": 100, "model_tags": ["default"]})
    )

    processor.set_module_references(trajectory_pool, train_batch_pool)

    assert processor._trajectory_pool is trajectory_pool
    assert processor._train_batch_pool is train_batch_pool


def test_base_algorithm_processor_reduce_metrics():
    """Test reduce_metrics static method"""
    import numpy as np

    metrics = {
        "reward_mean": [1.0, 2.0, 3.0],
        "accuracy": [0.5, 0.6, 0.7],
    }

    reduced = BaseAlgorithmProcessor.reduce_metrics(metrics)

    assert reduced["reward_mean"] == 2.0
    assert abs(reduced["accuracy"] - 0.6) < 1e-6


def test_base_algorithm_processor_compute_response_info():
    """Test _compute_response_info static method"""
    # Create a simple batch with responses
    batch_data = {
        "responses": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),  # 2 samples, 4 tokens each
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],  # 8 tokens (4 prompt + 4 response)
                [1, 1, 1, 1, 1, 1, 0, 0],  # 6 tokens (2 prompt + 4 response, 2 padding)
            ]
        ),
    }
    batch = Batch(values=batch_data, metadata={"batch_size": 2})

    response_info = BaseAlgorithmProcessor._compute_response_info(batch)

    assert "response_mask" in response_info
    assert "prompt_length" in response_info
    assert "response_length" in response_info
    assert response_info["response_mask"].shape == (2, 4)  # 2 samples, 4 response tokens


def test_base_algorithm_processor_compute_data_metrics():
    """Test compute_data_metrics static method"""
    # Create a batch with all required fields
    batch_size = 2
    response_length = 4
    sequence_length = 8

    batch_data = {
        "responses": torch.ones(batch_size, response_length),
        "attention_mask": torch.ones(batch_size, sequence_length),
        "token_level_scores": torch.rand(batch_size, response_length),
        "token_level_rewards": torch.rand(batch_size, response_length),
        "advantages": torch.rand(batch_size, response_length),
        "returns": torch.rand(batch_size, response_length),
    }
    batch = Batch(values=batch_data, metadata={"batch_size": batch_size})

    metrics = BaseAlgorithmProcessor.compute_data_metrics(batch, use_critic=False)

    # Check that all expected metrics are present
    assert "critic/score/mean" in metrics
    assert "critic/rewards/mean" in metrics
    assert "critic/advantages/mean" in metrics
    assert "critic/returns/mean" in metrics
    assert "response_length/mean" in metrics
    assert "prompt_length/mean" in metrics
