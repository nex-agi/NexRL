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
Tests for dataloader resume functionality (skip_batches method).

This test suite verifies that the skip_batches() method correctly skips
the specified number of batches when resuming training from a checkpoint.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from nexrl.data_loader import TorchDataLoader


@pytest.fixture
def test_data_file():
    """Create a temporary parquet file with test data."""
    num_items = 100
    df = pd.DataFrame(
        {
            "prompt": [f"prompt_{i}" for i in range(num_items)],
            "response": [f"response_{i}" for i in range(num_items)],
            "index": list(range(num_items)),
        }
    )

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".parquet", delete=False)
    temp_file.close()
    df.to_parquet(temp_file.name)

    yield temp_file.name

    # Cleanup
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)


@pytest.fixture
def dataloader_config(test_data_file):
    """Configuration for TorchDataLoader testing."""
    return OmegaConf.create(
        {
            "data_files": [test_data_file],
            "batch_size": 10,
            "rollout_repeat_n": 2,
            "shuffle": False,  # Important: no shuffle for testing
            "drop_last": False,
            "filter_prompts": False,
            "prompt_key": "prompt",
        }
    )


def test_skip_zero_batches(dataloader_config):
    """Test that skip_batches(0) starts from the beginning."""
    dataloader = TorchDataLoader(dataloader_config, is_validate=False)
    dataloader.skip_batches(0)

    first_item = dataloader.get_next_item()
    assert first_item is not None, "Should get an item"
    assert first_item["index"] == 0, f"Expected index 0, got {first_item['index']}"


def test_skip_batches_basic(dataloader_config):
    """Test skipping a specific number of batches."""
    dataloader = TorchDataLoader(dataloader_config, is_validate=False)
    num_batches_to_skip = 2
    batch_size = dataloader_config["batch_size"]

    dataloader.skip_batches(num_batches_to_skip)
    first_item_after_skip = dataloader.get_next_item()

    # After skipping 2 batches (2*10=20 underlying items), next should be item 20
    expected_index = num_batches_to_skip * batch_size

    assert first_item_after_skip is not None, "Should get an item after skip"
    assert (
        first_item_after_skip["index"] == expected_index
    ), f"Expected index {expected_index}, got {first_item_after_skip['index']}"

    # Verify the item has rollout_repeat_n metadata
    assert "group_id" in first_item_after_skip, "Item should have group_id"
    assert "run_id" in first_item_after_skip, "Item should have run_id"


def test_skip_more_than_available(dataloader_config):
    """Test that skipping more batches than available is handled gracefully."""
    dataloader = TorchDataLoader(dataloader_config, is_validate=False)

    # Calculate total batches
    num_items = 100
    batch_size = dataloader_config["batch_size"]
    total_batches = (num_items + batch_size - 1) // batch_size  # Ceiling division

    # Skip more than available
    dataloader.skip_batches(total_batches + 5)
    next_item = dataloader.get_next_item()

    assert next_item is None, "Should return None when all data is exhausted"


def test_cannot_skip_after_iteration_started(dataloader_config):
    """Test that skip_batches raises error after iteration has started."""
    dataloader = TorchDataLoader(dataloader_config, is_validate=False)

    # Start iteration by getting one item
    _ = dataloader.get_next_item()

    # Try to skip - should raise RuntimeError
    with pytest.raises(RuntimeError, match="Cannot skip batches after dataloader has started"):
        dataloader.skip_batches(1)


def test_skip_batches_with_rollout_repeat(dataloader_config):
    """Test that skip_batches works correctly with rollout_repeat_n > 1."""
    # Modify config to use rollout_repeat_n = 3
    config = dataloader_config.copy()
    config["rollout_repeat_n"] = 3

    dataloader = TorchDataLoader(config, is_validate=False)

    # Skip 1 batch
    dataloader.skip_batches(1)

    # Get items from the second batch
    # Each underlying batch has 10 items, after skip we should start from index 10
    first_item = dataloader.get_next_item()
    assert first_item is not None
    assert first_item["index"] == 10, f"Expected index 10, got {first_item['index']}"


def test_validation_dataloader_no_rollout_repeat(dataloader_config):
    """Test that validation dataloader doesn't apply rollout_repeat_n."""
    config = dataloader_config.copy()
    config["rollout_repeat_n"] = 3
    config["batch_size"] = 5

    # Validation dataloader should NOT apply rollout_repeat_n
    val_dataloader = TorchDataLoader(config, is_validate=True)

    # Get items from first batch
    items_in_batch = []
    for _ in range(5):  # batch_size
        item = val_dataloader.get_next_item()
        if item:
            items_in_batch.append(item)

    # Items should not have group_id (sign of no repetition)
    has_group_id = any("group_id" in item for item in items_in_batch)
    assert not has_group_id, "Validation items should not have group_id (no rollout_repeat_n)"


def test_skip_batches_sequential_access(dataloader_config):
    """Test that items after skipping are in correct sequential order."""
    dataloader = TorchDataLoader(dataloader_config, is_validate=False)
    batch_size = dataloader_config["batch_size"]

    # Skip 3 batches
    dataloader.skip_batches(3)

    # Get next few items and verify they're sequential
    expected_start_index = 3 * batch_size  # 30
    for i in range(5):
        item = dataloader.get_next_item()
        assert item is not None
        # Items might be repeated due to rollout_repeat_n, but underlying index should match
        assert (
            item["index"] >= expected_start_index
        ), f"Item index {item['index']} should be >= {expected_start_index}"


def test_skip_batches_logging(dataloader_config, caplog):
    """Test that skip_batches produces appropriate log messages."""
    import logging

    caplog.set_level(logging.INFO)

    dataloader = TorchDataLoader(dataloader_config, is_validate=False)

    # Skip batches
    dataloader.skip_batches(2)

    # Check that appropriate log messages were produced
    assert "Skipping 2 batches for resume" in caplog.text
    assert "Successfully skipped" in caplog.text
