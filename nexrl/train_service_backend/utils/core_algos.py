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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from typing import Dict, Tuple

import torch

from .core_utils import masked_mean


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        loss = torch.mean(seq_losses)
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss_NX_20250515(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict]:
    """Compute PPO policy-gradient loss with detailed clip statistics, advantage monitoring, equality count, and NaN checks."""
    # ratio = π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

    # Determine clip bounds
    if low_clip_range > 0 and high_clip_range > 0:
        lower_bound = 1.0 - low_clip_range
        upper_bound = 1.0 + high_clip_range
    else:
        lower_bound = 1.0 - cliprange
        upper_bound = 1.0 + cliprange

    # PPO clipped surrogate objectives
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, lower_bound, upper_bound)
    clip_pg_losses = torch.max(pg_losses, pg_losses2)

    # ===== clip statistics =====
    clipped_low_mask = ratio < lower_bound
    clipped_high_mask = ratio > upper_bound
    pg_clipfrac = masked_mean((clipped_low_mask | clipped_high_mask).float(), eos_mask)
    pg_clipfrac_low = masked_mean(clipped_low_mask.float(), eos_mask)
    pg_clipfrac_high = masked_mean(clipped_high_mask.float(), eos_mask)
    avg_ratio = masked_mean(ratio, eos_mask)

    # ===== advantage monitoring =====
    adv_pos_mask = (advantages > 0) & eos_mask.bool()
    adv_neg_mask = (advantages < 0) & eos_mask.bool()
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()
    avg_ratio_pos = masked_mean(ratio, adv_pos_mask.float())
    avg_ratio_neg = masked_mean(ratio, adv_neg_mask.float())

    # ===== adv-specific clip statistics =====
    adv_pos_clipfrac = masked_mean(
        (clipped_low_mask | clipped_high_mask).float(), adv_pos_mask.float()
    )
    adv_pos_clipfrac_low = masked_mean(clipped_low_mask.float(), adv_pos_mask.float())
    adv_pos_clipfrac_high = masked_mean(clipped_high_mask.float(), adv_pos_mask.float())
    adv_neg_clipfrac = masked_mean(
        (clipped_low_mask | clipped_high_mask).float(), adv_neg_mask.float()
    )
    adv_neg_clipfrac_low = masked_mean(clipped_low_mask.float(), adv_neg_mask.float())
    adv_neg_clipfrac_high = masked_mean(clipped_high_mask.float(), adv_neg_mask.float())

    # ===== equality monitoring =====
    eq_mask = (old_log_prob == log_prob) & eos_mask.bool()
    equal_count = eq_mask.sum()

    # ===== agg loss =====
    pg_loss = agg_loss(loss_mat=clip_pg_losses, loss_mask=eos_mask, loss_agg_mode=loss_agg_mode)

    # ===== adv-specific counts and losses =====
    pos_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_pos_mask).sum()
    neg_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_neg_mask).sum()
    adv_pos_loss = masked_mean(clip_pg_losses, adv_pos_mask.float())
    adv_neg_loss = masked_mean(clip_pg_losses, adv_neg_mask.float())

    # ===== adv kept counts =====
    kept_mask = (~clipped_low_mask & ~clipped_high_mask) & eos_mask.bool()
    adv_pos_kept_count = (kept_mask & adv_pos_mask).sum()
    adv_neg_kept_count = (kept_mask & adv_neg_mask).sum()

    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_low": pg_clipfrac_low.detach().item(),
        "actor/pg_clipfrac_high": pg_clipfrac_high.detach().item(),
        "actor/avg_ratio": avg_ratio.detach().item(),
        "actor/adv_pos_count": adv_pos_count.detach().item(),
        "actor/adv_neg_count": adv_neg_count.detach().item(),
        "actor/avg_ratio_pos": avg_ratio_pos.detach().item(),
        "actor/avg_ratio_neg": avg_ratio_neg.detach().item(),
        "actor/old_logprob_and_logprob_equal_count": equal_count.detach().item(),
        "actor/adv_pos_clipfrac": adv_pos_clipfrac.detach().item(),
        "actor/adv_pos_clipfrac_low": adv_pos_clipfrac_low.detach().item(),
        "actor/adv_pos_clipfrac_high": adv_pos_clipfrac_high.detach().item(),
        "actor/adv_neg_clipfrac": adv_neg_clipfrac.detach().item(),
        "actor/adv_neg_clipfrac_low": adv_neg_clipfrac_low.detach().item(),
        "actor/adv_neg_clipfrac_high": adv_neg_clipfrac_high.detach().item(),
        "actor/pos_clip_count": pos_clip_count.detach().item(),
        "actor/neg_clip_count": neg_clip_count.detach().item(),
        "actor/adv_pos_loss": adv_pos_loss.detach().item(),
        "actor/adv_neg_loss": adv_neg_loss.detach().item(),
        "actor/adv_pos_kept_count": adv_pos_kept_count.detach().item(),
        "actor/adv_neg_kept_count": adv_neg_kept_count.detach().item(),
    }


def compute_policy_loss_NX_20250515_v2(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
) -> Tuple[torch.Tensor, Dict]:
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122"""
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, {
        "pg_clipfrac": pg_clipfrac.detach().item(),
        "ppo_kl": ppo_kl.detach().item(),
        "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }


def compute_policy_loss(
    old_log_prob, log_prob, advantages, eos_mask, cliprange
) -> Tuple[torch.Tensor, Dict]:
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122"""
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

    return pg_loss, {"pg_clipfrac": pg_clipfrac.detach().item(), "ppo_kl": ppo_kl.detach().item()}


def compute_policy_loss_importance_sampling(
    old_log_prob, log_prob, advantages, eos_mask
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute importance sampling loss (unclipped policy gradient).

    Args:
        old_log_prob: Log probabilities from the sampling policy (π_old)
        log_prob: Log probabilities from the current policy (π_θ)
        advantages: Advantage estimates
        eos_mask: Mask indicating valid positions (1 for valid, 0 for padding)
        cliprange: Clip range (not used in standard importance sampling, kept for API compatibility)

    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Compute importance ratio: π_θ / π_old
    logprob_diff = log_prob - old_log_prob
    # print(f"old_log_prob: {old_log_prob}")
    # print(f"log_prob: {log_prob}")

    prob_ratio = torch.exp(logprob_diff)

    # Importance-weighted advantage

    importance_weighted = prob_ratio * advantages

    # Negative importance-weighted advantage (we want to maximize, so minimize negative)
    elementwise_loss = -importance_weighted
    # print(f"elementwise_loss: {elementwise_loss}")
    # Aggregate loss over valid tokens
    loss = masked_mean(elementwise_loss, eos_mask)
    # print(f"loss: {loss}")
    # Compute KL divergence for monitoring
    approx_kl = masked_mean(-logprob_diff, eos_mask)

    # Compute additional metrics
    avg_ratio = masked_mean(prob_ratio, eos_mask)

    # Advantage-specific metrics
    adv_pos_mask = (advantages > 0) & eos_mask.bool()
    adv_neg_mask = (advantages < 0) & eos_mask.bool()
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()
    avg_ratio_pos = masked_mean(prob_ratio, adv_pos_mask.float())
    avg_ratio_neg = masked_mean(prob_ratio, adv_neg_mask.float())

    metrics = {
        "pg_loss": loss.detach().item(),
        "approx_kl": approx_kl.detach().item(),
        "avg_ratio": avg_ratio.detach().item(),
        "adv_pos_count": adv_pos_count.detach().item(),
        "adv_neg_count": adv_neg_count.detach().item(),
        "avg_ratio_pos": avg_ratio_pos.detach().item(),
        "avg_ratio_neg": avg_ratio_neg.detach().item(),
    }

    return loss, metrics


def compute_policy_loss_impl(
    log_prob: torch.Tensor,
    data: dict,
    loss_func_type: str,
    clip_ratio: float,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_c: float,
    loss_agg_mode: str,
    do_old_log_prob_compute: bool,
) -> Tuple[torch.Tensor, dict]:
    """Compute policy loss based on the specified loss function type."""
    advantages = data["advantages"]
    responses = data["responses"]
    attention_mask = data["attention_mask"]
    response_length = responses.size(1)
    response_mask = attention_mask[:, -response_length:]

    old_log_prob = data["old_log_probs"] if do_old_log_prob_compute else log_prob.clone().detach()

    if loss_func_type == "NX_20250515":
        pg_loss, loss_metrics = compute_policy_loss_NX_20250515(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
            loss_agg_mode=loss_agg_mode,
        )
    elif loss_func_type == "NX_20250515_v2":
        pg_loss, loss_metrics = compute_policy_loss_NX_20250515_v2(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            cliprange=clip_ratio,
            cliprange_low=clip_ratio_low,
            cliprange_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
            loss_agg_mode=loss_agg_mode,
        )
    elif loss_func_type == "NX_20241031":
        pg_loss, loss_metrics = compute_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
        )
    elif loss_func_type == "importance_sampling":
        pg_loss, loss_metrics = compute_policy_loss_importance_sampling(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
        )
    else:
        raise RuntimeError(f"Not support loss_func_type: {loss_func_type}")

    return pg_loss, loss_metrics


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob."""
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-5, max=5)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
