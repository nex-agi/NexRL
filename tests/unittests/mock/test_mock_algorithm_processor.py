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
Tests for MockAlgorithmProcessor
"""

import torch

from nexrl.mock import MockAlgorithmProcessor


def test_mock_algorithm_processor_initialization(basic_config):
    """Test MockAlgorithmProcessor initialization"""
    processor = MockAlgorithmProcessor(basic_config)
    assert processor._processed_count == 0
    assert processor._mock_batch_size == 4
    assert processor._vocab_size == 151936


def test_create_mock_training_batch(basic_config):
    """Test creating mock training batch"""
    processor = MockAlgorithmProcessor(basic_config)
    tensor_batch, non_tensor_batch = processor.create_mock_training_batch(batch_size=2)

    # Check tensor batch
    assert "input_ids" in tensor_batch
    assert "attention_mask" in tensor_batch
    assert "advantages" in tensor_batch
    assert "old_log_probs" in tensor_batch

    # Check shapes
    assert tensor_batch["input_ids"].shape[0] == 2
    assert isinstance(tensor_batch["input_ids"], torch.Tensor)

    # Check non-tensor batch
    assert "sequence_ids" in non_tensor_batch
    assert len(non_tensor_batch["sequence_ids"]) == 2


def test_create_mock_inference_batch(basic_config):
    """Test creating mock inference batch"""
    processor = MockAlgorithmProcessor(basic_config)
    batch = processor.create_mock_inference_batch(batch_size=2)

    # Check required fields
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "responses" in batch
    assert "position_ids" in batch

    # Check shapes
    assert batch["input_ids"].shape[0] == 2
    assert isinstance(batch["input_ids"], torch.Tensor)


def test_get_mock_batch_with_metadata(basic_config):
    """Test creating mock batch with complete metadata"""
    processor = MockAlgorithmProcessor(basic_config)
    data = processor.get_mock_batch_with_metadata(batch_size=2)

    # Check structure
    assert "batch" in data
    assert "non_tensor_batch" in data
    assert "meta_info" in data

    # Check meta_info
    assert "temperature" in data["meta_info"]
    assert "ppo_epochs" in data["meta_info"]
    assert "global_token_num" in data["meta_info"]

    # Check that global_token_num is a list
    assert isinstance(data["meta_info"]["global_token_num"], list)
    assert len(data["meta_info"]["global_token_num"]) == 2
