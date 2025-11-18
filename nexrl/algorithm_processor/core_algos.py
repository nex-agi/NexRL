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

from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None):
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
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty: str = "kl"
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.

    Args:
        logprob: Current policy log probabilities
        ref_logprob: Reference policy log probabilities
        kl_penalty: Type of KL penalty ('kl', 'abs', 'mse', 'low_var_kl')

    Returns:
        KL divergence tensor
    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-5, max=5)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    raise NotImplementedError(f"KL penalty type {kl_penalty} not implemented")


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: shape (bs, response_length)
        eos_mask: shape (bs, response_length)
        index: Group IDs for each sample, shape (bs,)
        epsilon: Small constant to avoid division by zero

    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
        id2std: Dictionary mapping group IDs to standard deviations
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(score_tensor)
                id2std[idx] = torch.std(score_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            idx = index[i]
            scores[i] = (scores[i] - id2mean[idx]) / (id2std[idx] + epsilon)

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores, id2std
