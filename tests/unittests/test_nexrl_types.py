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
Tests for nexrl_types module
"""

import torch

from nexrl.nexrl_types import Batch, NexRLRole


def test_nexrl_role_enum():
    """Test NexRLRole enum values"""
    assert NexRLRole.ROLLOUT_WORKER.value == "rollout_worker"
    assert NexRLRole.TRAIN_WORKER.value == "train_worker"
    assert NexRLRole.ALGORITHM_PROCESSOR.value == "algorithm_processor"


def test_batch_creation():
    """Test basic Batch creation"""
    values = {"data": torch.tensor([1, 2, 3])}
    metadata = {"batch_size": 3}
    batch = Batch(values, metadata)

    assert len(batch) == 3
    assert "data" in batch.values
    assert batch.metadata["batch_size"] == 3


def test_batch_copy():
    """Test Batch copy method"""
    values = {"data": torch.tensor([1, 2, 3])}
    metadata = {"batch_size": 3, "model_tag": "test"}
    batch = Batch(values, metadata)

    batch_copy = batch.copy()
    assert len(batch_copy) == 3
    assert batch_copy.metadata["model_tag"] == "test"
    # Verify it's a copy, not the same object
    batch_copy.metadata["model_tag"] = "modified"
    assert batch.metadata["model_tag"] == "test"


def test_batch_to_dict():
    """Test Batch to_dict method"""
    values = {"data": torch.tensor([1, 2, 3])}
    metadata = {"batch_size": 3, "model_tag": "test"}
    batch = Batch(values, metadata)

    result = batch.to_dict()
    assert "data" in result
    assert result["batch_size"] == 3
    assert result["model_tag"] == "test"


def test_batch_remove_redundant_left_padding():
    """Test removing left padding from batch"""
    # Create batch with left padding
    input_ids = torch.tensor(
        [
            [0, 0, 1, 2, 3],
            [0, 0, 4, 5, 6],
        ]
    )
    values = {"input_ids": input_ids}
    metadata = {"batch_size": 2}
    batch = Batch(values, metadata)

    # Remove left padding (pad_token_id=0)
    stripped_batch = Batch.remove_redundant_left_padding(batch, pad_token_id=0)

    # Should remove 2 padding tokens from the left
    assert stripped_batch.values["input_ids"].shape[1] == 3
    assert torch.equal(stripped_batch.values["input_ids"][0], torch.tensor([1, 2, 3]))


def test_batch_remove_redundant_right_padding():
    """Test removing right padding from batch"""
    # Create batch with right padding
    input_ids = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 0, 0],
        ]
    )
    values = {"input_ids": input_ids}
    metadata = {"batch_size": 2}
    batch = Batch(values, metadata)

    # Remove right padding (pad_token_id=0)
    stripped_batch = Batch.remove_redundant_right_padding(batch, pad_token_id=0)

    # Should remove 2 padding tokens from the right
    assert stripped_batch.values["input_ids"].shape[1] == 3
    assert torch.equal(stripped_batch.values["input_ids"][0], torch.tensor([1, 2, 3]))


def test_batch_to_nextrainer_batch():
    """Test converting batch to nextrainer format"""
    tensor_values = {"input_ids": torch.tensor([[1, 2, 3]])}
    metadata = {"batch_size": 1, "model_tag": "test"}
    batch = Batch(tensor_values, metadata)

    nextrainer_batch = batch.to_nextrainer_batch()

    assert "batch" in nextrainer_batch
    assert "non_tensor_batch" in nextrainer_batch
    assert "meta_info" in nextrainer_batch
    assert "input_ids" in nextrainer_batch["batch"]
    assert nextrainer_batch["meta_info"]["batch_size"] == 1
