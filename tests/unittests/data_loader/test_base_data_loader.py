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
Tests for BaseDataLoader and SequentialDataLoader (via MockDataLoader)
"""

from nexrl.data_loader import BaseDataLoader
from nexrl.mock import MockDataLoader

# =============================================================================
# Tests for BaseDataLoader
# =============================================================================


def test_base_data_loader_repeat_item():
    """Test static repeat_item method"""
    original_item = {"id": 1, "prompt": "test"}
    n = 3

    repeated_items = BaseDataLoader.repeat_item(original_item, n)

    # Should have n items
    assert len(repeated_items) == n

    # All should have the same group_id
    group_ids = [item["group_id"] for item in repeated_items]
    assert len(set(group_ids)) == 1

    # Each should have different run_id
    run_ids = [item["run_id"] for item in repeated_items]
    assert run_ids == list(range(n))

    # All should have original fields
    for item in repeated_items:
        assert item["id"] == 1
        assert item["prompt"] == "test"


def test_base_data_loader_module_name(data_loader_config):
    """Test data loader module name functionality"""
    loader = MockDataLoader(data_loader_config)

    # Set module name
    loader.set_module_name("test_data_loader")
    assert loader.get_module_name() == "test_data_loader"


def test_base_data_loader_health_check(data_loader_config):
    """Test data loader health check"""
    loader = MockDataLoader(data_loader_config)

    # Health check should return True
    assert loader.health_check() is True


def test_base_data_loader_set_module_references(data_loader_config):
    """Test setting weight sync controller reference"""
    loader = MockDataLoader(data_loader_config)

    # Create mock weight sync controller
    class MockWeightSyncController:
        pass

    controller = MockWeightSyncController()
    loader.set_module_references(controller)

    assert loader._weight_sync_controller is controller


def test_base_data_loader_add_item(data_loader_config):
    """Test add_item method (should call add_item_back by default)"""
    loader = MockDataLoader(data_loader_config)
    initial_size = len(loader._data)

    # Add item using generic add_item
    new_item = {"id": 1000, "prompt": "another test prompt"}
    loader.add_item(new_item)

    assert len(loader._data) == initial_size + 1


# =============================================================================
# Tests for SequentialDataLoader (via MockDataLoader)
# =============================================================================


def test_sequential_data_loader_initialization(data_loader_config):
    """Test SequentialDataLoader initialization"""
    loader = MockDataLoader(data_loader_config)

    assert loader._batch_size == 4
    assert len(loader._data) > 0
    assert loader._data_index == 0
    assert loader._buffer_index == 0


def test_sequential_data_loader_get_next_item(data_loader_config):
    """Test getting next item from sequential data loader"""
    loader = MockDataLoader(data_loader_config)

    # Get first item
    item = loader.get_next_item()
    assert item is not None
    assert "prompt" in item
    assert "id" in item
    assert loader._data_index == 1


def test_sequential_data_loader_can_return_item(data_loader_config):
    """Test checking if sequential data loader can return items"""
    loader = MockDataLoader(data_loader_config)

    # Initially can't return (buffer not loaded)
    # After first get_next_item, buffer is loaded
    item = loader.get_next_item()
    assert item is not None
    assert loader.can_return_item() or loader._buffer_index < len(loader._data_buffer)


def test_sequential_data_loader_is_finished(data_loader_config):
    """Test checking if sequential data loader is finished"""
    loader = MockDataLoader(data_loader_config)

    # Initially not finished
    assert not loader.is_finished()

    # Fetch all data
    while not loader.is_finished():
        item = loader.get_next_item()
        if item is None:
            break

    # Now should be finished
    assert loader.is_finished()


def test_sequential_data_loader_reset(data_loader_config):
    """Test resetting sequential data loader"""
    loader = MockDataLoader(data_loader_config)

    # Fetch some data
    loader.get_next_item()
    loader.get_next_item()
    assert loader._data_index > 0

    # Reset
    loader.reset()
    assert loader._data_index == 0
    assert loader._buffer_index == 0


def test_sequential_data_loader_add_item_back(data_loader_config):
    """Test adding item back to sequential data loader"""
    loader = MockDataLoader(data_loader_config)
    initial_size = len(loader._data)

    # Add a new item
    new_item = {"id": 999, "prompt": "new test prompt"}
    loader.add_item_back(new_item)

    assert len(loader._data) == initial_size + 1


def test_sequential_data_loader_unlock_for_weight_sync(data_loader_config):
    """Test unlock for weight sync"""
    loader = MockDataLoader(data_loader_config)

    # Set the lock event
    loader._lock_for_weight_sync_event.set()
    assert loader._lock_for_weight_sync_event.is_set()

    # Unlock
    loader.unlock_for_weight_sync()
    assert not loader._lock_for_weight_sync_event.is_set()
