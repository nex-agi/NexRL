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
Tests for BaseRolloutWorker
"""

from nexrl.mock import MockRolloutWorker


def test_base_rollout_worker_initialization(rollout_worker_config):
    """Test BaseRolloutWorker initialization through MockRolloutWorker"""
    worker = MockRolloutWorker(rollout_worker_config)

    assert worker._config is not None
    assert worker._llm_client is not None
    assert worker._stop_event is not None


def test_rollout_worker_has_module_name(rollout_worker_config):
    """Test that rollout worker has module name functionality"""
    worker = MockRolloutWorker(rollout_worker_config)

    # Set module name
    worker.set_module_name("test_rollout_worker")
    assert worker.get_module_name() == "test_rollout_worker"


def test_rollout_worker_health_check(rollout_worker_config):
    """Test rollout worker health check"""
    worker = MockRolloutWorker(rollout_worker_config)

    # Health check should return True
    assert worker.health_check() is True


def test_rollout_worker_set_module_references(rollout_worker_config):
    """Test setting module references"""
    worker = MockRolloutWorker(rollout_worker_config)

    # Create mock references
    class MockTrajectoryPool:
        pass

    class MockDataLoader:
        pass

    class MockWeightSyncController:
        pass

    class MockValidator:
        pass

    trajectory_pool = MockTrajectoryPool()
    dataloader = MockDataLoader()
    weight_sync_controller = MockWeightSyncController()
    validator = MockValidator()

    # Set references
    worker.set_module_references(
        trajectory_pool=trajectory_pool,
        dataloader=dataloader,
        weight_sync_controller=weight_sync_controller,
        validate_dataloader=dataloader,
        validator=validator,
    )

    assert worker._trajectory_pool is trajectory_pool
    assert worker._dataloader is dataloader
    assert worker._weight_sync_controller is weight_sync_controller
    assert worker._validator is validator


def test_rollout_worker_stop_event(rollout_worker_config):
    """Test stop event functionality"""
    worker = MockRolloutWorker(rollout_worker_config)

    # Initially not set
    assert not worker._stop_event.is_set()

    # Set the stop event
    worker._stop_event.set()
    assert worker._stop_event.is_set()

    # Clear it
    worker._stop_event.clear()
    assert not worker._stop_event.is_set()
