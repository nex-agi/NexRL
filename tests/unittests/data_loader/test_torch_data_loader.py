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
Tests for TorchDataLoader
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nexrl.data_loader import TorchDataLoader


@pytest.fixture
def torch_data_loader_config():
    """Configuration for TorchDataLoader testing"""
    test_data_path = Path(__file__).parent / "test_data" / "test_dataset.parquet"

    return OmegaConf.create(
        {
            "batch_size": 2,
            "data_files": [str(test_data_path)],  # TorchDataLoader expects data_files as a list
            "num_workers": 0,  # Use 0 for testing to avoid multiprocessing issues
            "shuffle": False,
            "keep_batch_order": False,
            "rollout_repeat_n": 1,
        }
    )


def test_torch_data_loader_initialization(torch_data_loader_config):
    """Test TorchDataLoader initialization"""
    loader = TorchDataLoader(torch_data_loader_config)

    assert loader._batch_size == 2
    assert loader._dataset is not None
    assert loader._dataloader is not None


def test_torch_data_loader_get_next_item(torch_data_loader_config):
    """Test getting items from TorchDataLoader"""
    loader = TorchDataLoader(torch_data_loader_config)

    # Get first item
    item = loader.get_next_item()
    assert item is not None
    assert "prompt" in item
    assert "id" in item


def test_torch_data_loader_iteration(torch_data_loader_config):
    """Test iterating through TorchDataLoader"""
    loader = TorchDataLoader(torch_data_loader_config)

    items = []
    while not loader.is_finished():
        item = loader.get_next_item()
        if item is None:
            break
        items.append(item)

    # Should have loaded all 8 items from test dataset
    assert len(items) == 8


def test_torch_data_loader_reset(torch_data_loader_config):
    """Test resetting TorchDataLoader"""
    loader = TorchDataLoader(torch_data_loader_config)

    # Fetch some items
    loader.get_next_item()
    loader.get_next_item()
    assert loader._data_index > 0

    # Reset
    loader.reset()
    assert loader._data_index == 0

    # Should be able to fetch items again
    item = loader.get_next_item()
    assert item is not None


def test_torch_data_loader_is_finished(torch_data_loader_config):
    """Test TorchDataLoader is_finished method"""
    loader = TorchDataLoader(torch_data_loader_config)

    # Initially not finished
    assert not loader.is_finished()

    # Consume all items
    while not loader.is_finished():
        item = loader.get_next_item()
        if item is None:
            break

    # Should be finished
    assert loader.is_finished()


def test_torch_data_loader_add_item_back(torch_data_loader_config):
    """Test adding item back to TorchDataLoader"""
    loader = TorchDataLoader(torch_data_loader_config)

    # Get an item first
    item = loader.get_next_item()
    assert item is not None

    # Modify it and add back
    item["id"] = 999
    # Note: TorchDataLoader logs a warning but doesn't maintain a separate pending_items deque
    # It adds items back to the current batch buffer
    loader.add_item_back(item)
    # Just verify it doesn't crash


def test_torch_data_loader_add_item_front(torch_data_loader_config):
    """Test adding item front to TorchDataLoader"""
    loader = TorchDataLoader(torch_data_loader_config)

    # Get an item first
    item = loader.get_next_item()
    assert item is not None

    # Modify it and add to front
    item["id"] = 888
    loader.add_item_front(item)
    # Just verify it doesn't crash


def test_torch_data_loader_with_shuffle(torch_data_loader_config):
    """Test TorchDataLoader with shuffle enabled"""
    torch_data_loader_config.shuffle = True
    loader = TorchDataLoader(torch_data_loader_config)

    # Should initialize successfully with shuffle
    assert loader._dataset is not None

    # Should be able to get items
    item = loader.get_next_item()
    assert item is not None


def test_torch_data_loader_can_return_item(torch_data_loader_config):
    """Test can_return_item method"""
    loader = TorchDataLoader(torch_data_loader_config)

    # Get an item to load buffer
    item = loader.get_next_item()
    assert item is not None

    # Should be able to return items
    assert loader.can_return_item() or loader._buffer_index < len(loader._data_buffer)
