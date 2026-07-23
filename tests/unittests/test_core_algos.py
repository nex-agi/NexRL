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

"""Regression tests for core RL algorithms."""

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from nexrl.algorithm.core_algos import (
    compute_grpo_advantage_for_trajectories,
    compute_grpo_outcome_advantage,
)
from nexrl.nexrl_types import Trajectory


def _make_trajectories(
    rewards: Sequence[float], run_ids: Sequence[int] | None = None
) -> list[Trajectory]:
    if run_ids is None:
        run_ids = range(len(rewards))

    return [
        Trajectory(
            tokens=[1, 2],
            loss_mask=[0, 1],
            reward=reward,
            is_val=False,
            extra_fields={"group_id": "group", "run_id": run_id},
        )
        for reward, run_id in zip(rewards, run_ids, strict=True)
    ]


def _compute_advantages(
    rewards: Sequence[float],
    run_ids: Sequence[int] | None = None,
    *,
    use_run_ids: bool = True,
) -> torch.Tensor:
    trajectories = _make_trajectories(rewards, run_ids)
    compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=use_run_ids)
    return torch.stack([torch.as_tensor(trajectory["advantage"]) for trajectory in trajectories])


def _compute_tensor_advantages(
    rewards: Sequence[float], run_ids: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[Any, torch.Tensor]]:
    reward_tensor = torch.tensor(rewards).unsqueeze(-1)
    group_ids = torch.zeros(len(rewards), dtype=torch.long)
    eos_mask = torch.ones_like(reward_tensor)
    advantages, _, group_stds = compute_grpo_outcome_advantage(
        reward_tensor,
        eos_mask,
        group_ids,
        run_ids=run_ids,
    )
    return advantages.squeeze(-1), group_stds


@pytest.mark.parametrize(
    "run_ids",
    [
        None,
        torch.arange(16),
        torch.tensor([0] * 8 + [1] * 8),
        torch.zeros(16, dtype=torch.long),
    ],
    ids=["no-run-ids", "unique-run-ids", "repeated-run-ids", "single-run-id"],
)
def test_tensor_grpo_identical_decimal_rewards_have_zero_advantage(
    run_ids: torch.Tensor | None,
) -> None:
    advantages, _ = _compute_tensor_advantages([0.1] * 16, run_ids)

    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_identical_decimal_rewards_have_exactly_zero_advantage() -> None:
    """A constant reward group must not create a synthetic policy gradient."""
    advantages = _compute_advantages([0.1] * 16)

    assert torch.equal(advantages, torch.zeros_like(advantages))


@pytest.mark.parametrize(
    "rewards",
    [
        [0] * 16,
        [1] * 16,
        [0, 1] * 8,
    ],
    ids=["all-zero", "all-one", "mixed-zero-one"],
)
def test_integer_rewards_are_normalized_as_floats(rewards: list[int]) -> None:
    advantages = _compute_advantages(rewards)

    assert torch.isfinite(advantages).all()
    if len(set(rewards)) == 1:
        assert torch.equal(advantages, torch.zeros_like(advantages))


def test_identical_decimal_rewards_without_run_ids_have_zero_advantage() -> None:
    advantages = _compute_advantages([0.1] * 512, use_run_ids=False)

    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_without_run_ids_keeps_population_std_normalization() -> None:
    rewards = [0.1, 1.0, 0.1, 1.0]
    advantages = _compute_advantages(rewards, use_run_ids=False)

    rewards_tensor = torch.tensor(rewards, dtype=torch.float64)
    expected_std, expected_mean = torch.std_mean(rewards_tensor, correction=0)
    expected = (rewards_tensor - expected_mean) / (expected_std + 1e-6)

    torch.testing.assert_close(
        advantages,
        expected.to(advantages.dtype),
        rtol=0,
        atol=0,
    )


def test_mixed_rewards_keep_sample_std_normalization() -> None:
    """The joint reduction must preserve the existing correction=1 behavior."""
    rewards = [0.1, 1.0, 0.1, 1.0]
    advantages = _compute_advantages(rewards)

    rewards_tensor = torch.tensor(rewards)
    expected_std, expected_mean = torch.std_mean(rewards_tensor)
    expected = (rewards_tensor - expected_mean) / (expected_std + 1e-6)

    torch.testing.assert_close(advantages, expected, rtol=0, atol=0)


def test_tensor_grpo_mixed_rewards_keep_sample_std_normalization() -> None:
    rewards = [0.1, 1.0, 0.1, 1.0]
    advantages, group_stds = _compute_tensor_advantages(rewards)

    rewards_tensor = torch.tensor(rewards)
    expected_std, expected_mean = torch.std_mean(rewards_tensor)
    expected = (rewards_tensor - expected_mean) / (expected_std + 1e-6)

    torch.testing.assert_close(advantages, expected, rtol=0, atol=0)
    torch.testing.assert_close(group_stds[0], expected_std, rtol=0, atol=0)


def test_single_run_id_guard_does_not_produce_nan() -> None:
    """Multiple trajectories from one run retain the existing finite fallback."""
    advantages = _compute_advantages([0.1] * 16, run_ids=[0] * 16)

    assert torch.isfinite(advantages).all()
    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_single_run_id_guard_preserves_unit_std_fallback() -> None:
    rewards = [0.1, 1.0, 0.1, 1.0]
    advantages = _compute_advantages(rewards, run_ids=[0] * len(rewards))

    rewards_tensor = torch.tensor(rewards)
    per_run_mean = torch.std_mean(rewards_tensor, correction=0)[1]
    expected = (rewards_tensor - per_run_mean) / (torch.tensor(1.0) + 1e-6)

    assert torch.isfinite(advantages).all()
    torch.testing.assert_close(advantages, expected, rtol=0, atol=0)
