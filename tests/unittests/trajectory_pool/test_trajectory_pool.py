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
Tests for TrajectoryPool and TrajectoryPoolInstance
"""

import pytest
import torch
from omegaconf import OmegaConf

from nexrl.trajectory_pool import TrajectoryPool, TrajectoryPoolInstance

# =============================================================================
# Tests for TrajectoryPoolInstance
# =============================================================================


@pytest.fixture
def trajectory_pool_instance_config():
    """Configuration for TrajectoryPoolInstance testing"""
    return OmegaConf.create(
        {
            "batch_size": 4,
            "check_batch_ready_function": "batch_size_reached",
        }
    )


def test_trajectory_pool_instance_initialization(trajectory_pool_instance_config):
    """Test TrajectoryPoolInstance initialization"""
    instance = TrajectoryPoolInstance(trajectory_pool_instance_config, model_tag="default")

    assert instance.model_tag == "default"
    assert instance._store is not None
    assert instance._weight_sync_event is not None
    assert not instance._weight_sync_event.is_set()  # Initially unlocked


def test_trajectory_pool_instance_put_trajectory(trajectory_pool_instance_config):
    """Test putting trajectories into instance"""
    instance = TrajectoryPoolInstance(trajectory_pool_instance_config, model_tag="default")

    trajectory = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "reward": 1.0,
        "model_tag": "default",
    }

    result = instance.put_trajectory(trajectory)
    assert result == "success"


def test_trajectory_pool_instance_lock_unlock(trajectory_pool_instance_config):
    """Test locking and unlocking for weight sync"""
    instance = TrajectoryPoolInstance(trajectory_pool_instance_config, model_tag="default")

    assert not instance.is_locked()

    instance.lock_for_weight_sync()
    assert instance.is_locked()

    instance.unlock_for_weight_sync()
    assert not instance.is_locked()


def test_trajectory_pool_instance_put_while_locked(trajectory_pool_instance_config):
    """Test that putting trajectory while locked returns re-rollout"""
    instance = TrajectoryPoolInstance(trajectory_pool_instance_config, model_tag="default")

    instance.lock_for_weight_sync()

    trajectory = {
        "input_ids": torch.tensor([1, 2, 3]),
        "reward": 1.0,
        "model_tag": "default",
    }

    result = instance.put_trajectory(trajectory)
    assert result == "re-rollout"


def test_trajectory_pool_instance_is_empty(trajectory_pool_instance_config):
    """Test is_empty for instance"""
    instance = TrajectoryPoolInstance(trajectory_pool_instance_config, model_tag="default")

    assert instance.is_empty()

    trajectory = {
        "input_ids": torch.tensor([1, 2, 3]),
        "reward": 1.0,
        "model_tag": "default",
    }
    instance.put_trajectory(trajectory)

    assert not instance.is_empty()


def test_trajectory_pool_instance_set_module_references(trajectory_pool_instance_config):
    """Test setting module references"""
    instance = TrajectoryPoolInstance(trajectory_pool_instance_config, model_tag="default")

    class MockDataLoader:
        def can_return_item(self):
            return False

    class MockWeightSyncController:
        def trajectory_pool_notify_batch_ready(self, model_tag):
            pass

    class MockActivityTracker:
        def is_rollout_worker_quiescent(self):
            return True

        def experiment_logger_post(self, backend, data, step=None):
            pass

    dataloader = MockDataLoader()
    controller = MockWeightSyncController()
    tracker = MockActivityTracker()

    instance.set_module_references(dataloader, controller, tracker)

    assert instance._dataloader is dataloader
    assert instance._weight_sync_controller is controller
    assert instance._activity_tracker is tracker


# =============================================================================
# Tests for TrajectoryPool
# =============================================================================


@pytest.fixture
def trajectory_pool_config():
    """Configuration for TrajectoryPool testing"""
    return OmegaConf.create(
        {
            "batch_size": 4,
            "check_batch_ready_function": "batch_size_reached",
        }
    )


@pytest.fixture
def mock_module_references():
    """Create mock module references for TrajectoryPool"""

    class MockDataLoader:
        def can_return_item(self):
            return False

    class MockWeightSyncController:
        def trajectory_pool_notify_batch_ready(self, model_tag):
            pass

    class MockActivityTracker:
        def is_rollout_worker_quiescent(self):
            return True

        def experiment_logger_post(self, backend, data, step=None):
            pass

    return {
        "dataloader": MockDataLoader(),
        "weight_sync_controller": MockWeightSyncController(),
        "activity_tracker": MockActivityTracker(),
    }


def test_trajectory_pool_initialization(trajectory_pool_config):
    """Test TrajectoryPool initialization"""
    pool = TrajectoryPool(trajectory_pool_config)

    assert pool._config is not None
    assert pool._instances == {}


def test_trajectory_pool_put_trajectory(trajectory_pool_config, mock_module_references):
    """Test putting trajectory into pool"""
    pool = TrajectoryPool(trajectory_pool_config)
    pool.set_module_references(
        dataloader=mock_module_references["dataloader"],
        weight_sync_controller=mock_module_references["weight_sync_controller"],
    )
    pool._activity_tracker = mock_module_references["activity_tracker"]

    trajectory = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "reward": 1.0,
        "model_tag": "default",
    }

    result = pool.put_trajectory(trajectory)
    assert result == "success"
    assert "default" in pool._instances


def test_trajectory_pool_multiple_models(trajectory_pool_config, mock_module_references):
    """Test managing multiple model tags"""
    pool = TrajectoryPool(trajectory_pool_config)
    pool.set_module_references(
        dataloader=mock_module_references["dataloader"],
        weight_sync_controller=mock_module_references["weight_sync_controller"],
    )
    pool._activity_tracker = mock_module_references["activity_tracker"]

    # Add trajectories for different models
    for model_tag in ["model1", "model2"]:
        trajectory = {
            "input_ids": torch.tensor([1, 2, 3]),
            "reward": 1.0,
            "model_tag": model_tag,
        }
        pool.put_trajectory(trajectory)

    assert len(pool._instances) == 2
    assert "model1" in pool._instances
    assert "model2" in pool._instances


def test_trajectory_pool_get_model_tags(trajectory_pool_config, mock_module_references):
    """Test getting all model tags"""
    pool = TrajectoryPool(trajectory_pool_config)
    pool.set_module_references(
        dataloader=mock_module_references["dataloader"],
        weight_sync_controller=mock_module_references["weight_sync_controller"],
    )
    pool._activity_tracker = mock_module_references["activity_tracker"]

    # Add trajectories for different models
    for model_tag in ["model1", "model2", "model3"]:
        trajectory = {
            "input_ids": torch.tensor([1, 2, 3]),
            "reward": 1.0,
            "model_tag": model_tag,
        }
        pool.put_trajectory(trajectory)

    model_tags = pool.get_model_tags()
    assert len(model_tags) == 3
    assert "model1" in model_tags
    assert "model2" in model_tags
    assert "model3" in model_tags


def test_trajectory_pool_is_empty(trajectory_pool_config, mock_module_references):
    """Test is_empty for pool"""
    pool = TrajectoryPool(trajectory_pool_config)
    pool.set_module_references(
        dataloader=mock_module_references["dataloader"],
        weight_sync_controller=mock_module_references["weight_sync_controller"],
    )
    pool._activity_tracker = mock_module_references["activity_tracker"]

    # Empty pool
    assert pool.is_empty()

    # Add trajectory
    trajectory = {
        "input_ids": torch.tensor([1, 2, 3]),
        "reward": 1.0,
        "model_tag": "default",
    }
    pool.put_trajectory(trajectory)

    assert not pool.is_empty()


def test_trajectory_pool_is_empty_specific_model(trajectory_pool_config, mock_module_references):
    """Test is_empty for specific model tag"""
    pool = TrajectoryPool(trajectory_pool_config)
    pool.set_module_references(
        dataloader=mock_module_references["dataloader"],
        weight_sync_controller=mock_module_references["weight_sync_controller"],
    )
    pool._activity_tracker = mock_module_references["activity_tracker"]

    # Add trajectory to model1
    trajectory = {
        "input_ids": torch.tensor([1, 2, 3]),
        "reward": 1.0,
        "model_tag": "model1",
    }
    pool.put_trajectory(trajectory)

    # model1 should not be empty
    assert not pool.is_empty(model_tag="model1")

    # model2 doesn't exist, should be empty
    assert pool.is_empty(model_tag="model2")


def test_trajectory_pool_notify_weight_sync(trajectory_pool_config, mock_module_references):
    """Test notifying weight sync for specific model"""
    pool = TrajectoryPool(trajectory_pool_config)
    pool.set_module_references(
        dataloader=mock_module_references["dataloader"],
        weight_sync_controller=mock_module_references["weight_sync_controller"],
    )
    pool._activity_tracker = mock_module_references["activity_tracker"]

    # Create instance by adding trajectory
    trajectory = {
        "input_ids": torch.tensor([1, 2, 3]),
        "reward": 1.0,
        "model_tag": "default",
    }
    pool.put_trajectory(trajectory)

    # Lock for weight sync
    pool.notify_need_weight_sync("default")
    assert pool._instances["default"].is_locked()

    # Unlock
    pool.unlock_for_weight_sync("default")
    assert not pool._instances["default"].is_locked()


def test_trajectory_pool_module_name(trajectory_pool_config):
    """Test trajectory pool module name functionality"""
    pool = TrajectoryPool(trajectory_pool_config)

    pool.set_module_name("test_trajectory_pool")
    assert pool.get_module_name() == "test_trajectory_pool"


def test_trajectory_pool_health_check(trajectory_pool_config):
    """Test trajectory pool health check"""
    pool = TrajectoryPool(trajectory_pool_config)
    assert pool.health_check() is True
