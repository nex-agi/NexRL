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
Tests for core algorithms
"""

import torch

from nexrl.algorithm.core_algos import (
    AdaptiveKLController,
    FixedKLController,
    compute_grpo_outcome_advantage,
    kl_penalty,
    masked_mean,
    masked_whiten,
)


def test_masked_mean():
    """Test masked_mean function"""
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])

    # Compute mean along axis 1
    result = masked_mean(values, mask, axis=1)

    # First row: (1 + 2 + 3) / 3 = 2.0
    # Second row: (5 + 6) / 2 = 5.5
    assert torch.allclose(result[0], torch.tensor(2.0))
    assert torch.allclose(result[1], torch.tensor(5.5))


def test_masked_whiten():
    """Test masked_whiten function"""
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    result = masked_whiten(values, mask)

    # Result should have mean ~0 and std ~1
    assert torch.allclose(masked_mean(result, mask), torch.tensor(0.0), atol=1e-6)


def test_adaptive_kl_controller():
    """Test AdaptiveKLController"""
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0, horizon=10000)

    assert controller.value == 0.1
    assert controller.target == 6.0

    # Update with current KL higher than target
    controller.update(current_kl=8.0, n_steps=100)

    # Value should increase since current_kl > target
    assert controller.value > 0.1


def test_fixed_kl_controller():
    """Test FixedKLController"""
    controller = FixedKLController(kl_reward_coef=0.5)

    assert controller.value == 0.5

    # Update should not change value
    controller.update(current_kl=10.0, n_steps=100)
    assert controller.value == 0.5


def test_kl_penalty_kl():
    """Test kl_penalty with 'kl' type"""
    logprob = torch.tensor([[0.1, 0.2, 0.3]])
    ref_logprob = torch.tensor([[0.15, 0.25, 0.35]])

    result = kl_penalty(logprob, ref_logprob, kl_penalty="kl")

    expected = logprob - ref_logprob
    assert torch.allclose(result, expected)


def test_kl_penalty_abs():
    """Test kl_penalty with 'abs' type"""
    logprob = torch.tensor([[0.1, 0.2, 0.3]])
    ref_logprob = torch.tensor([[0.15, 0.25, 0.35]])

    result = kl_penalty(logprob, ref_logprob, kl_penalty="abs")

    expected = (logprob - ref_logprob).abs()
    assert torch.allclose(result, expected)


def test_kl_penalty_mse():
    """Test kl_penalty with 'mse' type"""
    logprob = torch.tensor([[0.1, 0.2, 0.3]])
    ref_logprob = torch.tensor([[0.15, 0.25, 0.35]])

    result = kl_penalty(logprob, ref_logprob, kl_penalty="mse")

    expected = 0.5 * (logprob - ref_logprob).square()
    assert torch.allclose(result, expected)


def test_compute_grpo_outcome_advantage():
    """Test compute_grpo_outcome_advantage"""
    # Create sample data
    batch_size = 4
    response_length = 5

    # Token-level rewards (will be summed to get scores)
    token_level_rewards = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # score = 1.0
            [2.0, 0.0, 0.0, 0.0, 0.0],  # score = 2.0
            [1.5, 0.0, 0.0, 0.0, 0.0],  # score = 1.5
            [2.5, 0.0, 0.0, 0.0, 0.0],  # score = 2.5
        ]
    )

    # EOS mask (all tokens are valid)
    eos_mask = torch.ones(batch_size, response_length)

    # Group IDs (first two samples in group 0, last two in group 1)
    # Use list of integers instead of tensor to match expected interface
    index = [0, 0, 1, 1]

    advantages, returns, id2std = compute_grpo_outcome_advantage(
        token_level_rewards, eos_mask, index
    )

    # Check shapes
    assert advantages.shape == (batch_size, response_length)
    assert returns.shape == (batch_size, response_length)

    # Check that std dict has correct groups
    assert 0 in id2std
    assert 1 in id2std
