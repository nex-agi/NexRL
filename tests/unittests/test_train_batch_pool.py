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
Tests for train_batch_pool module
"""

import torch

from nexrl.nexrl_types import Batch
from nexrl.train_batch_pool import TrainBatchPool


def test_train_batch_pool_initialization(train_batch_pool_config):
    """Test TrainBatchPool initialization"""
    pool = TrainBatchPool(train_batch_pool_config)
    assert pool._config is not None
    assert pool.is_empty()


def test_put_and_get_batch(train_batch_pool_config):
    """Test putting and getting batches"""
    pool = TrainBatchPool(train_batch_pool_config)

    # Create a test batch
    values = {"input_ids": torch.tensor([[1, 2, 3]])}
    metadata = {"batch_size": 1, "model_tag": "test_model"}
    batch = Batch(values, metadata)

    # Put batch
    success = pool.put_batch(batch, "test_update")
    assert success is True
    assert not pool.is_empty()

    # Get batch
    retrieved_batch = pool.get_batch("test_model")
    assert retrieved_batch is not None
    assert retrieved_batch.metadata["model_tag"] == "test_model"
    assert pool.is_empty()


def test_get_batch_empty_pool(train_batch_pool_config):
    """Test getting batch from empty pool"""
    pool = TrainBatchPool(train_batch_pool_config)
    batch = pool.get_batch("test_model")
    assert batch is None


def test_multiple_batches(train_batch_pool_config):
    """Test handling multiple batches"""
    pool = TrainBatchPool(train_batch_pool_config)

    # Add multiple batches
    for i in range(3):
        values = {"input_ids": torch.tensor([[i, i + 1, i + 2]])}
        metadata = {"batch_size": 1, "model_tag": "test_model"}
        batch = Batch(values, metadata)
        pool.put_batch(batch, "test_update")

    # Retrieve all batches
    batches = []
    while not pool.is_empty():
        batch = pool.get_batch("test_model")
        if batch:
            batches.append(batch)

    assert len(batches) == 3


def test_multiple_models(train_batch_pool_config):
    """Test handling batches for multiple models"""
    pool = TrainBatchPool(train_batch_pool_config)

    # Add batches for different models
    for model_tag in ["model_a", "model_b"]:
        values = {"input_ids": torch.tensor([[1, 2, 3]])}
        metadata = {"batch_size": 1, "model_tag": model_tag}
        batch = Batch(values, metadata)
        pool.put_batch(batch, "test_update")

    # Get batches for each model
    batch_a = pool.get_batch("model_a")
    batch_b = pool.get_batch("model_b")

    assert batch_a is not None
    assert batch_b is not None
    assert batch_a.metadata["model_tag"] == "model_a"
    assert batch_b.metadata["model_tag"] == "model_b"
