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
Core algorithms for GRPO and other RL methods
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from ..nexrl_types import Trajectory


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: int | None = None):
    """Compute mean of values with mask applied"""
    if axis is not None:
        return (values * mask).sum(axis) / (mask.sum(axis) + 1e-8)
    return (values * mask).sum() / (mask.sum() + 1e-8)


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-8):
    """Whiten (normalize) values with mask applied"""
    mean = masked_mean(values, mask)
    var = masked_mean((values - mean) ** 2, mask)
    std = torch.sqrt(var + epsilon)
    return (values - mean) / std


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_reward_coef: float):
        self.value = kl_reward_coef

    def update(self, current_kl: float, n_steps: int):
        pass


def kl_penalty(
    logprob: torch.Tensor, ref_logprob: torch.Tensor, kl_penalty_type: str = "kl"
) -> torch.Tensor:
    """Compute KL divergence given logprob and ref_logprob.

    Args:
        logprob: Current policy log probabilities
        ref_logprob: Reference policy log probabilities
        kl_penalty_type: Type of KL penalty ('kl', 'abs', 'mse', 'low_var_kl')

    Returns:
        KL divergence tensor
    """
    if kl_penalty_type == "kl":
        return logprob - ref_logprob

    if kl_penalty_type == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty_type == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty_type == "low_var_kl":
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-5, max=5)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    raise NotImplementedError(f"KL penalty type {kl_penalty_type} not implemented")


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    run_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: shape (bs, response_length)
        eos_mask: shape (bs, response_length)
        index: Group IDs for each sample, shape (bs,)
        epsilon: Small constant to avoid division by zero
        run_ids: Optional run IDs for each sample, shape (bs,).
                 If provided, uses variant number of run_ids for mean computation.

    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
        id2std: Dictionary mapping group IDs to standard deviations
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2run_id: defaultdict[Any, list[Any]] | None = (
        defaultdict(list) if run_ids is not None else None
    )
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            if run_ids is not None and id2run_id is not None:
                id2run_id[index[i]].append(run_ids[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                # Two-stage mean: first compute mean per run_id, then mean of those means
                if run_ids is not None and id2run_id is not None:
                    # Group scores by run_id
                    run_id_to_scores = defaultdict(list)
                    for score, run_id in zip(id2score[idx], id2run_id[idx]):
                        run_id_key = run_id.item() if hasattr(run_id, "item") else run_id
                        run_id_to_scores[run_id_key].append(score)
                    # Compute mean per run_id, then mean of those means
                    per_run_means = [
                        torch.stack(scores).mean() for scores in run_id_to_scores.values()
                    ]
                    per_run_means_tensor = torch.stack(per_run_means)
                    id2mean[idx] = per_run_means_tensor.mean()
                    # Guard against single-element tensor: torch.std uses Bessel
                    # correction (N-1), which returns nan for N=1.
                    if len(per_run_means) <= 1:
                        id2std[idx] = torch.tensor(1.0, device=scores.device)
                    else:
                        id2std[idx] = torch.std(per_run_means_tensor)
                else:
                    id2mean[idx] = torch.mean(score_tensor)
                    id2std[idx] = torch.std(score_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            idx = index[i]
            scores[i] = (scores[i] - id2mean[idx]) / (id2std[idx] + epsilon)

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores, id2std


def compute_grpo_advantage_for_trajectories(
    trajectories: list[Trajectory],
    epsilon: float = 1e-6,
    logger: logging.Logger | None = None,
    use_run_ids: bool = False,
) -> list[Trajectory]:
    """
    Compute GRPO advantages with std normalization and add 'advantage' to each trajectory.

    Args:
        trajectories: List of trajectories containing rewards and group_ids
        epsilon: Small constant to avoid division by zero
        logger: Optional logger for debugging
        use_run_ids: If True, uses variant number of run_ids for mean computation.
                     Trajectories should contain 'run_id' field when this is True.
    """
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, traj in enumerate(trajectories):
        group_id = traj.get("group_id", str(i))
        group_to_indices[group_id].append(i)

    group_stats: dict[str, tuple[float, float]] = {}
    for group_id, indices in group_to_indices.items():
        rewards = [trajectories[i]["reward"] for i in indices]
        if logger is not None:
            logger.info(f"group_id: {group_id}, rewards: {rewards}")
        if len(rewards) == 1:
            group_stats[group_id] = (0.0, 1.0)
        else:
            # Two-stage mean: first compute mean per run_id, then mean of those means
            if use_run_ids:
                run_ids_list = [trajectories[i].get("run_id") for i in indices]
                # Group rewards by run_id
                run_id_to_rewards = defaultdict(list)
                for reward, run_id in zip(rewards, run_ids_list):
                    run_id_to_rewards[run_id].append(reward)
                # Compute mean per run_id, then mean of those means
                per_run_means = [
                    np.mean(run_rewards).item() for run_rewards in run_id_to_rewards.values()
                ]
                per_run_means_tensor = torch.tensor(per_run_means)
                # mean_reward = np.mean(per_run_means)
                # std_reward = np.std(per_run_means)
                mean_reward = per_run_means_tensor.mean()
                std_reward = torch.std(per_run_means_tensor)
            else:
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
            group_stats[group_id] = (mean_reward, std_reward)
    for i, traj in enumerate(trajectories):
        group_id = traj.get("group_id", str(i))
        mean_reward, std_reward = group_stats[group_id]
        reward = traj["reward"]
        advantage = (reward - mean_reward) / (std_reward + epsilon)
        traj["advantage"] = advantage

    return trajectories
