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
Tests for SimpleRolloutWorker using real implementation with mocked inference client
"""

import pytest

from nexrl.mock import MockInferenceServiceClient
from nexrl.rollout_worker import SimpleRolloutWorker


@pytest.fixture
def simple_rollout_worker(rollout_worker_config):
    """Create SimpleRolloutWorker with MockInferenceServiceClient"""
    worker = SimpleRolloutWorker(rollout_worker_config)
    # Manually set the inference client for testing
    worker._inference_client = MockInferenceServiceClient(rollout_worker_config)
    return worker


def test_simple_rollout_worker_initialization(simple_rollout_worker):
    """Test SimpleRolloutWorker initialization"""
    assert simple_rollout_worker._config is not None
    assert simple_rollout_worker._inference_client is not None
    assert isinstance(simple_rollout_worker._inference_client, MockInferenceServiceClient)


def test_simple_rollout_worker_rollout(simple_rollout_worker):
    """Test SimpleRolloutWorker rollout method"""
    # Mock the _put_trajectory method to capture trajectories
    trajectories = []

    def mock_put_trajectory(trajectory):
        trajectories.append(trajectory)
        return "trajectory_id_123"

    simple_rollout_worker._put_trajectory = mock_put_trajectory

    # Execute a rollout
    task = {"prompt": "What is 2+2?", "id": 1}
    result = simple_rollout_worker.rollout(task)

    # Should have processed the task
    assert result == "trajectory_id_123"
    assert len(trajectories) == 1

    # Check trajectory structure
    traj = trajectories[0]
    assert "prompt" in traj
    assert "response" in traj
    assert "finish_reason" in traj
    assert traj["prompt"] == task["prompt"]
    assert "Mock response" in traj["response"]


def test_simple_rollout_worker_rollout_missing_prompt(simple_rollout_worker):
    """Test SimpleRolloutWorker rollout with missing prompt"""
    # Mock the _put_trajectory method
    simple_rollout_worker._put_trajectory = lambda x: "id"

    # Execute a rollout without prompt
    task = {"id": 1}
    result = simple_rollout_worker.rollout(task)

    # Should return None when prompt is missing
    assert result is None


def test_simple_rollout_worker_inference_integration(simple_rollout_worker):
    """Test that SimpleRolloutWorker correctly uses inference client"""
    # Mock the _put_trajectory method to capture trajectories
    trajectories = []
    simple_rollout_worker._put_trajectory = lambda x: trajectories.append(x) or "id"

    # Execute a rollout
    task = {"prompt": "Test prompt", "id": 1, "extra_field": "extra_value"}
    simple_rollout_worker.rollout(task)

    # Check that extra fields from task are included in trajectory
    traj = trajectories[0]
    assert traj["id"] == 1
    assert traj["extra_field"] == "extra_value"
