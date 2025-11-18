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
Tests for MockDataLoader
"""

import numpy as np

from nexrl.mock import MockDataLoader


def test_mock_data_loader_initialization(data_loader_config):
    """Test MockDataLoader initialization"""
    loader = MockDataLoader(data_loader_config)
    assert loader._batch_size == 4
    assert loader._api_type == "completion"
    assert len(loader._data) > 0


def test_mock_data_loader_completion_mode(data_loader_config):
    """Test MockDataLoader with completion API type"""
    loader = MockDataLoader(data_loader_config)

    # Fetch a batch
    batch = loader._fetch_batch_data()
    assert len(batch) <= loader._batch_size
    assert all("prompt" in item for item in batch)
    assert all(isinstance(item["prompt"], str) for item in batch)


def test_mock_data_loader_generate_mode():
    """Test MockDataLoader with generate API type"""
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "batch_size": 4,
            "mock_api_type": "generate",
        }
    )
    loader = MockDataLoader(config)

    # Fetch a batch
    batch = loader._fetch_batch_data()
    assert len(batch) <= loader._batch_size

    # Check that prompts are in message array format
    for item in batch:
        assert "prompt" in item
        assert isinstance(item["prompt"], np.ndarray)
        messages = item["prompt"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


def test_mock_data_loader_is_finished(data_loader_config):
    """Test is_finished method"""
    loader = MockDataLoader(data_loader_config)

    # Initially not finished
    assert not loader.is_finished()

    # Fetch all data
    while not loader.is_finished():
        loader._fetch_batch_data()

    # Now should be finished
    assert loader.is_finished()


def test_mock_data_loader_add_item_back(data_loader_config):
    """Test add_item_back method"""
    loader = MockDataLoader(data_loader_config)
    initial_size = len(loader._data)

    # Add an item back
    new_item = {"id": 999, "prompt": "test prompt", "mock_generated": False}
    loader.add_item_back(new_item)

    assert len(loader._data) == initial_size + 1


def test_mock_data_loader_reset(data_loader_config):
    """Test reset iterator"""
    loader = MockDataLoader(data_loader_config)

    # Fetch some data
    loader._fetch_batch_data()
    assert loader._fetched_data_index > 0

    # Reset
    loader._reset_iterator()
    assert loader._fetched_data_index == 0
