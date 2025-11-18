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
Tests for MockRolloutWorker
"""

from nexrl.mock import MockRolloutWorker


def test_mock_rollout_worker_initialization(rollout_worker_config):
    """Test MockRolloutWorker initialization"""
    worker = MockRolloutWorker(rollout_worker_config)
    assert worker._processed_count == 0
    assert worker._mock_delay == 0.1


def test_mock_rollout_worker_step(rollout_worker_config):
    """Test MockRolloutWorker step method"""
    worker = MockRolloutWorker(rollout_worker_config)

    # Mock the _put_trajectory method to avoid errors
    trajectories = []

    def mock_put_trajectory(trajectory):
        trajectories.append(trajectory)

    worker._put_trajectory = mock_put_trajectory

    # Execute a step
    task = {"prompt": "test prompt"}
    worker.step(task)

    assert worker._processed_count == 1
    assert len(trajectories) == 1
    assert "response" in trajectories[0]
    assert "reward" in trajectories[0]
