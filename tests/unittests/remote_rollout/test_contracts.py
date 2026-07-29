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

import pytest
from pydantic import ValidationError

from nexrl.remote_rollout import RemoteRolloutRequest, RemoteRolloutResult, RemoteTrajectory


def _trajectory(**overrides) -> RemoteTrajectory:
    values = {
        "tokens": [10, 20, 30],
        "loss_mask": [0, 1, 1],
        "old_log_probs": [0.0, -0.2, -0.3],
    }
    values.update(overrides)
    return RemoteTrajectory(**values)


def test_request_is_json_round_trip_safe():
    request = RemoteRolloutRequest(
        rollout_id="rollout-1",
        task={"problem": "fix the bug", "attempt": 1, "flags": [True, None]},
    )

    assert RemoteRolloutRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize(
    "task",
    [
        {"not_json": (1, 2)},
        {"not_json": float("nan")},
        {1: "not a JSON object key"},
    ],
)
def test_request_rejects_non_json_task_values(task):
    with pytest.raises(ValidationError):
        RemoteRolloutRequest(rollout_id="rollout-1", task=task)


def test_trajectory_accepts_response_aligned_sampling_mask():
    trajectory = _trajectory(sampling_mask=[[20, 21], [30]])

    assert trajectory.sampling_mask == [[20, 21], [30]]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"loss_mask": [0, 1]}, "equal length"),
        ({"old_log_probs": [0.0, -0.2]}, "equal length"),
        ({"loss_mask": [1, 1, 1]}, r"loss_mask\[0\] must be 0"),
        ({"loss_mask": [0, 0, 0]}, "at least one response token"),
        ({"old_log_probs": [-0.1, -0.2, -0.3]}, r"old_log_probs\[0\]"),
        (
            {"loss_mask": [0, 0, 1], "old_log_probs": [0.0, -0.2, -0.3]},
            "context token old_log_probs",
        ),
        ({"sampling_mask": [[20]]}, "one entry for each response token"),
        ({"sampling_mask": [[21], [30]]}, "must contain sampled token 20"),
    ],
)
def test_trajectory_rejects_invalid_alignment(overrides, message):
    with pytest.raises(ValidationError, match=message):
        _trajectory(**overrides)


def test_result_supports_multiple_trajectories_and_json_round_trip():
    result = RemoteRolloutResult(
        rollout_id="rollout-1",
        trajectories=[
            _trajectory(),
            _trajectory(tokens=[40, 50], loss_mask=[0, 1], old_log_probs=[0.0, -0.4]),
        ],
        reward=1,
        metrics={"tests_passed": 2},
    )

    restored = RemoteRolloutResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert len(restored.trajectories) == 2


def test_result_requires_a_trajectory():
    with pytest.raises(ValidationError, match="at least 1 item"):
        RemoteRolloutResult(rollout_id="rollout-1", trajectories=[], reward=0.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.0"])
def test_result_rejects_invalid_reward(value):
    with pytest.raises(ValidationError):
        RemoteRolloutResult(
            rollout_id="rollout-1",
            trajectories=[_trajectory()],
            reward=value,
        )


def test_contracts_reject_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RemoteRolloutRequest(
            rollout_id="rollout-1",
            task={},
            provider="openai",
        )
