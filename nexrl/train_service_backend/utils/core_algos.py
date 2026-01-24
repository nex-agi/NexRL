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


def compute_reverse_kl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    train_token_start_pct: float = 0.0,
    train_token_end_pct: float = 1.0,
    advantage_start_pct: float = 0.0,
    advantage_end_pct: float = 1.0,
    student_entropy: torch.Tensor = None,
    entropy_start_pct: float = 0.0,
    entropy_end_pct: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    """Compute reverse KL divergence loss from log probabilities (OPTIMIZED VERSION).

    This is the memory-efficient implementation that directly uses log probabilities
    instead of full logits, as described in https://thinkingmachines.ai/blog/on-policy-distillation/

    Reverse KL: KL(π_student || π_teacher) = E_{x ~ π_student} [log π_student(x) - log π_teacher(x)]

    Key advantages over logits-based version:
    - Memory efficient: O(batch_size * seq_len) vs O(batch_size * seq_len * vocab_size)
    - Faster: No need to compute full softmax and gather
    - Mathematically equivalent for sampled tokens
    - ~40,000x memory reduction for typical vocab sizes!

    Args:
        student_log_probs: torch.Tensor
            Student model log probabilities for sampled tokens
            Shape: [batch_size, response_length]
        teacher_log_probs: torch.Tensor
            Teacher model log probabilities for the same sampled tokens
            Shape: [batch_size, response_length]
        response_mask: torch.Tensor
            Mask for valid tokens of shape [batch_size, response_length]
        loss_agg_mode: str
            How to aggregate the loss: "token-mean" or "seq-mean-token-sum" or "seq-mean-token-mean"
        train_token_start_pct: float
            Start percentage of tokens to train (0.0-1.0, default: 0.0)
        train_token_end_pct: float
            End percentage of tokens to train (0.0-1.0, default: 1.0)
            Example: train_token_start_pct=0.0, train_token_end_pct=0.5 trains first 50% of tokens
                     train_token_start_pct=0.5, train_token_end_pct=1.0 trains last 50% of tokens
                     train_token_start_pct=0.25, train_token_end_pct=0.75 trains middle 50% of tokens
        advantage_start_pct: float
            Start percentile of advantage (kl_divergence) to train (0.0-1.0, default: 0.0)
        advantage_end_pct: float
            End percentile of advantage (kl_divergence) to train (0.0-1.0, default: 1.0)
            Example: advantage_start_pct=0.0, advantage_end_pct=0.5 trains tokens with bottom 50% advantage
                     advantage_start_pct=0.5, advantage_end_pct=1.0 trains tokens with top 50% advantage
                     advantage_start_pct=0.25, advantage_end_pct=0.75 trains tokens with middle 50% advantage
        student_entropy: torch.Tensor (optional)
            Student model entropy for each token [batch_size, response_length]
            Required when entropy_start_pct > 0.0 or entropy_end_pct < 1.0
        entropy_start_pct: float
            Start percentile of entropy to train (0.0-1.0, default: 0.0)
        entropy_end_pct: float
            End percentile of entropy to train (0.0-1.0, default: 1.0)
            Example: entropy_start_pct=0.0, entropy_end_pct=0.5 trains tokens with bottom 50% entropy (more confident)
                     entropy_start_pct=0.5, entropy_end_pct=1.0 trains tokens with top 50% entropy (less confident)
                     entropy_start_pct=0.25, entropy_end_pct=0.75 trains tokens with middle 50% entropy

    Returns:
        loss: torch.Tensor
            Scalar loss value
        metrics: Dict
            Dictionary containing loss metrics
    """
    batch_size, response_length = student_log_probs.shape

    # Apply token range selection if specified
    if train_token_start_pct > 0.0 or train_token_end_pct < 1.0:
        # Create a new mask that only includes tokens in the specified range
        response_mask = apply_token_range_mask(
            response_mask, train_token_start_pct, train_token_end_pct
        )

    # Compute reverse KL per token
    # KL(student || teacher) = log student(x) - log teacher(x)
    # This is also the "advantage" in On-Policy Distillation
    token_wise_kl = student_log_probs - teacher_log_probs

    # Apply advantage-based selection if specified
    # Select tokens based on their advantage (kl_divergence) percentile within the batch
    if advantage_start_pct > 0.0 or advantage_end_pct < 1.0:
        # Get valid advantages (masked tokens)
        valid_advantages = token_wise_kl[response_mask.bool()]

        if valid_advantages.numel() > 0:
            # Compute percentiles on the valid advantages
            start_percentile = torch.quantile(valid_advantages, advantage_start_pct)
            end_percentile = torch.quantile(valid_advantages, advantage_end_pct)

            # Create mask for tokens within the percentile range
            advantage_mask = (token_wise_kl >= start_percentile) & (token_wise_kl <= end_percentile)

            # Store original number of tokens before filtering for logging
            tokens_before_filtering = response_mask.sum().item()

            # Combine with existing response_mask (use multiplication for float masks)
            response_mask = response_mask * advantage_mask.float()

            tokens_after_filtering = response_mask.sum().item()

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(f"advantage_start_pct: {advantage_start_pct}")
            print(f"advantage_end_pct: {advantage_end_pct}")
            print(f"start_percentile (advantage): {start_percentile.item():.4f}")
            print(f"end_percentile (advantage): {end_percentile.item():.4f}")
            print(f"tokens_before_filtering: {tokens_before_filtering}")
            print(f"tokens_after_filtering: {tokens_after_filtering}")
            print(f"filtering_ratio: {tokens_after_filtering / tokens_before_filtering:.2%}")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    # Apply entropy-based selection if specified
    # Select tokens based on their entropy percentile within the batch
    if student_entropy is not None and (entropy_start_pct > 0.0 or entropy_end_pct < 1.0):
        # Get valid entropies (masked tokens)
        valid_entropies = student_entropy[response_mask.bool()]

        if valid_entropies.numel() > 0:
            # Compute percentiles on the valid entropies
            start_percentile_entropy = torch.quantile(valid_entropies, entropy_start_pct)
            end_percentile_entropy = torch.quantile(valid_entropies, entropy_end_pct)

            # Create mask for tokens within the percentile range
            entropy_mask = (student_entropy >= start_percentile_entropy) & (
                student_entropy <= end_percentile_entropy
            )

            # Store original number of tokens before filtering for logging
            tokens_before_entropy_filtering = response_mask.sum().item()

            # Combine with existing response_mask (use multiplication for float masks)
            response_mask = response_mask * entropy_mask.float()

            tokens_after_entropy_filtering = response_mask.sum().item()

            print("########################################")
            print(f"entropy_start_pct: {entropy_start_pct}")
            print(f"entropy_end_pct: {entropy_end_pct}")
            print(f"start_percentile (entropy): {start_percentile_entropy.item():.4f}")
            print(f"end_percentile (entropy): {end_percentile_entropy.item():.4f}")
            print(f"tokens_before_entropy_filtering: {tokens_before_entropy_filtering}")
            print(f"tokens_after_entropy_filtering: {tokens_after_entropy_filtering}")
            print(
                f"entropy_filtering_ratio: {tokens_after_entropy_filtering / tokens_before_entropy_filtering:.2%}"
            )
            print("########################################")

    # Aggregate loss
    reverse_kl_loss = agg_loss(
        loss_mat=token_wise_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    # Compute additional metrics (no gradient required)
    with torch.no_grad():
        # Mean KL per token
        mean_kl_per_token = masked_mean(token_wise_kl, response_mask)

        # Student and teacher perplexity
        student_perplexity = torch.exp(masked_mean(-student_log_probs, response_mask))
        teacher_perplexity = torch.exp(masked_mean(-teacher_log_probs, response_mask))

        # Distribution difference metrics
        log_prob_diff = torch.abs(student_log_probs - teacher_log_probs)
        mean_log_prob_diff = masked_mean(log_prob_diff, response_mask)
        max_log_prob_diff = (log_prob_diff * response_mask).max().item()

        # Token range statistics (useful when using partial token training)
        num_trained_tokens = response_mask.sum().item()
        avg_trained_tokens_per_sample = num_trained_tokens / batch_size

        # Advantage statistics (for advantage-based selection)
        if response_mask.sum() > 0:
            # Get valid advantages (only masked tokens)
            valid_advantages = token_wise_kl[response_mask.bool()]
            advantage_mean = valid_advantages.mean().item()
            advantage_std = valid_advantages.std().item()
            advantage_min = valid_advantages.min().item()
            advantage_max = valid_advantages.max().item()
        else:
            advantage_mean = 0.0
            advantage_std = 0.0
            advantage_min = 0.0
            advantage_max = 0.0

        # Entropy statistics (for entropy-based selection)
        if student_entropy is not None and response_mask.sum() > 0:
            valid_entropies = student_entropy[response_mask.bool()]
            entropy_mean = valid_entropies.mean().item()
            entropy_std = valid_entropies.std().item()
            entropy_min = valid_entropies.min().item()
            entropy_max = valid_entropies.max().item()
        else:
            entropy_mean = 0.0
            entropy_std = 0.0
            entropy_min = 0.0
            entropy_max = 0.0

    metrics = {
        "distill/reverse_kl": mean_kl_per_token.item(),
        "distill/student_perplexity": student_perplexity.item(),
        "distill/teacher_perplexity": teacher_perplexity.item(),
        "distill/mean_log_prob_diff": mean_log_prob_diff.item(),
        "distill/max_log_prob_diff": max_log_prob_diff,
        "distill/num_trained_tokens": num_trained_tokens,
        "distill/avg_trained_tokens_per_sample": avg_trained_tokens_per_sample,
        "distill/train_token_start_pct": train_token_start_pct,
        "distill/train_token_end_pct": train_token_end_pct,
        "distill/advantage_start_pct": advantage_start_pct,
        "distill/advantage_end_pct": advantage_end_pct,
        "distill/advantage_mean": advantage_mean,
        "distill/advantage_std": advantage_std,
        "distill/advantage_min": advantage_min,
        "distill/advantage_max": advantage_max,
        "distill/entropy_start_pct": entropy_start_pct,
        "distill/entropy_end_pct": entropy_end_pct,
        "distill/entropy_mean": entropy_mean,
        "distill/entropy_std": entropy_std,
        "distill/entropy_min": entropy_min,
        "distill/entropy_max": entropy_max,
    }

    return reverse_kl_loss, metrics


def compute_forward_kl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    train_token_start_pct: float = 0.0,
    train_token_end_pct: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    """Compute forward KL divergence loss from log probabilities.

    Forward KL: KL(π_teacher || π_student) = E_{x ~ π_teacher} [log π_teacher(x) - log π_student(x)]

    Note: For on-policy distillation, reverse KL is typically preferred because:
    - It's mode-seeking (student focuses on teacher's high-prob regions)
    - More stable training

    But forward KL can be useful for:
    - Mode-covering behavior (student covers all teacher modes)
    - Some specific applications

    Args:
        student_log_probs: Student log probs [batch_size, response_length]
        teacher_log_probs: Teacher log probs [batch_size, response_length]
        response_mask: Mask [batch_size, response_length]
        loss_agg_mode: Aggregation mode
        train_token_start_pct: Start percentage of tokens to train (0.0-1.0, default: 0.0)
        train_token_end_pct: End percentage of tokens to train (0.0-1.0, default: 1.0)

    Returns:
        loss: Scalar loss
        metrics: Dict of metrics
    """
    batch_size, response_length = student_log_probs.shape

    # Apply token range selection if specified
    if train_token_start_pct > 0.0 or train_token_end_pct < 1.0:
        # Create a new mask that only includes tokens in the specified range
        response_mask = apply_token_range_mask(
            response_mask, train_token_start_pct, train_token_end_pct
        )

    # Forward KL is just the reverse of reverse KL
    token_wise_kl = teacher_log_probs - student_log_probs

    forward_kl_loss = agg_loss(
        loss_mat=token_wise_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    with torch.no_grad():
        mean_kl_per_token = masked_mean(token_wise_kl, response_mask)

        # Token range statistics (useful when using partial token training)
        num_trained_tokens = response_mask.sum().item()
        avg_trained_tokens_per_sample = num_trained_tokens / batch_size

    metrics = {
        "distill/forward_kl": mean_kl_per_token.item(),
        "distill/num_trained_tokens": num_trained_tokens,
        "distill/avg_trained_tokens_per_sample": avg_trained_tokens_per_sample,
        "distill/train_token_start_pct": train_token_start_pct,
        "distill/train_token_end_pct": train_token_end_pct,
    }

    return forward_kl_loss, metrics


def apply_token_range_mask(
    response_mask: torch.Tensor,
    start_pct: float,
    end_pct: float,
) -> torch.Tensor:
    """Apply token range masking to only train specific percentage range of tokens.

    For each sample in the batch, this function:
    1. Finds the actual number of valid tokens (based on response_mask)
    2. Calculates which tokens fall in the [start_pct, end_pct) range
    3. Creates a new mask that only includes those tokens

    Args:
        response_mask: Original response mask [batch_size, response_length]
        start_pct: Start percentage (0.0-1.0), inclusive
        end_pct: End percentage (0.0-1.0), exclusive

    Returns:
        new_mask: Modified mask with only the specified token range [batch_size, response_length]

    Example:
        If a sample has 100 valid tokens and start_pct=0.2, end_pct=0.8:
        - Will keep tokens from index 20 to 79 (60 tokens total)
        - Tokens 0-19 and 80-99 will be masked out
    """
    assert 0.0 <= start_pct <= 1.0, f"start_pct must be in [0, 1], got {start_pct}"
    assert 0.0 <= end_pct <= 1.0, f"end_pct must be in [0, 1], got {end_pct}"
    assert start_pct < end_pct, f"start_pct ({start_pct}) must be < end_pct ({end_pct})"

    batch_size, response_length = response_mask.shape
    new_mask = torch.zeros_like(response_mask)

    for i in range(batch_size):
        # Find valid token positions for this sample
        valid_positions = torch.where(response_mask[i] > 0)[0]

        if len(valid_positions) == 0:
            # No valid tokens, keep mask as all zeros
            continue

        num_valid_tokens = len(valid_positions)

        # Calculate start and end indices based on percentages
        start_idx = int(num_valid_tokens * start_pct)
        end_idx = int(num_valid_tokens * end_pct)

        # Ensure we have at least one token to train
        if end_idx <= start_idx:
            end_idx = start_idx + 1

        # Clamp to valid range
        end_idx = min(end_idx, num_valid_tokens)

        # Set mask for the selected token range
        selected_positions = valid_positions[start_idx:end_idx]
        new_mask[i, selected_positions] = 1.0

    return new_mask


def compute_reverse_kl_loss_with_clip(
    student_log_probs: torch.Tensor,
    old_student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange: float = 0.2,
    cliprange_low: float | None = None,
    cliprange_high: float | None = None,
    loss_agg_mode: str = "token-mean",
    train_token_start_pct: float = 0.0,
    train_token_end_pct: float = 1.0,
    advantage_start_pct: float = 0.0,
    advantage_end_pct: float = 1.0,
    student_entropy: torch.Tensor = None,
    entropy_start_pct: float = 0.0,
    entropy_end_pct: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    """Compute reverse KL divergence loss with PPO-like clipping for stable distillation.

    This implements the approach that combines:
    1. Reverse KL divergence loss to match teacher distribution
    2. PPO-style clipping to prevent large policy updates

    The intuition is similar to PPO:
    - We want to move the student toward the teacher (reverse KL)
    - But we also want to limit how much the student changes per update (clipping)
    - This makes training more stable and prevents mode collapse

    Reference:
    - On-policy distillation: https://thinkingmachines.ai/blog/on-policy-distillation/

    The loss combines two objectives:
    1. KL(student || teacher): push student toward teacher
    2. Clip(student, old_student): prevent large changes

    Args:
        student_log_probs: Current student log probs [batch_size, response_length]
        old_student_log_probs: Student log probs from previous iteration [batch_size, response_length]
        teacher_log_probs: Teacher log probs [batch_size, response_length]
        response_mask: Mask [batch_size, response_length]
        cliprange: PPO-style clip range (default: 0.2)
        cliprange_low: Lower clip range (if None, uses cliprange)
        cliprange_high: Upper clip range (if None, uses cliprange)
        loss_agg_mode: Aggregation mode
        train_token_start_pct: Start percentage of tokens to train (0.0-1.0, default: 0.0)
        train_token_end_pct: End percentage of tokens to train (0.0-1.0, default: 1.0)
            Example: train_token_start_pct=0.0, train_token_end_pct=0.5 trains first 50% of tokens
                     train_token_start_pct=0.5, train_token_end_pct=1.0 trains last 50% of tokens
                     train_token_start_pct=0.25, train_token_end_pct=0.75 trains middle 50% of tokens
        advantage_start_pct: Start percentile of advantage (kl_divergence) to train (0.0-1.0, default: 0.0)
        advantage_end_pct: End percentile of advantage (kl_divergence) to train (0.0-1.0, default: 1.0)
            Example: advantage_start_pct=0.0, advantage_end_pct=0.5 trains tokens with bottom 50% advantage
                     advantage_start_pct=0.5, advantage_end_pct=1.0 trains tokens with top 50% advantage
                     advantage_start_pct=0.25, advantage_end_pct=0.75 trains tokens with middle 50% advantage
        student_entropy: Student model entropy for each token [batch_size, response_length] (optional)
        entropy_start_pct: Start percentile of entropy to train (0.0-1.0, default: 0.0)
        entropy_end_pct: End percentile of entropy to train (0.0-1.0, default: 1.0)
            Example: entropy_start_pct=0.0, entropy_end_pct=0.5 trains tokens with bottom 50% entropy
                     entropy_start_pct=0.5, entropy_end_pct=1.0 trains tokens with top 50% entropy

    Returns:
        loss: Scalar loss
        metrics: Dict of metrics
    """
    batch_size, response_length = student_log_probs.shape

    # Apply token range selection if specified
    if train_token_start_pct > 0.0 or train_token_end_pct < 1.0:
        # Create a new mask that only includes tokens in the specified range
        response_mask = apply_token_range_mask(
            response_mask, train_token_start_pct, train_token_end_pct
        )
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print(f"train_token_start_pct: {train_token_start_pct}")
    print(f"train_token_end_pct: {train_token_end_pct}")
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    # Compute the ratio of new vs old student policy (similar to PPO)
    # ratio = π_student_new / π_student_old = exp(log π_new - log π_old)
    log_ratio = student_log_probs - old_student_log_probs
    ratio = torch.exp(log_ratio)

    # Compute reverse KL divergence: KL(student || teacher)
    # This is the "advantage" in our case - we want to move toward teacher
    kl_divergence = student_log_probs - teacher_log_probs

    # Apply advantage-based selection if specified
    # Select tokens based on their advantage (kl_divergence) percentile within the batch
    if advantage_start_pct > 0.0 or advantage_end_pct < 1.0:
        # Get valid advantages (masked tokens)
        valid_advantages = kl_divergence[response_mask.bool()]

        if valid_advantages.numel() > 0:
            # Compute percentiles on the valid advantages
            start_percentile = torch.quantile(valid_advantages, advantage_start_pct)
            end_percentile = torch.quantile(valid_advantages, advantage_end_pct)

            # Create mask for tokens within the percentile range
            advantage_mask = (kl_divergence >= start_percentile) & (kl_divergence <= end_percentile)

            # Store original number of tokens before filtering for logging
            tokens_before_filtering = response_mask.sum().item()

            # Combine with existing response_mask (use multiplication for float masks)
            response_mask = response_mask * advantage_mask.float()

            tokens_after_filtering = response_mask.sum().item()

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(f"advantage_start_pct: {advantage_start_pct}")
            print(f"advantage_end_pct: {advantage_end_pct}")
            print(f"start_percentile (advantage): {start_percentile.item():.4f}")
            print(f"end_percentile (advantage): {end_percentile.item():.4f}")
            print(f"tokens_before_filtering: {tokens_before_filtering}")
            print(f"tokens_after_filtering: {tokens_after_filtering}")
            print(f"filtering_ratio: {tokens_after_filtering / tokens_before_filtering:.2%}")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    # Apply entropy-based selection if specified
    # Select tokens based on their entropy percentile within the batch
    if student_entropy is not None and (entropy_start_pct > 0.0 or entropy_end_pct < 1.0):
        # Get valid entropies (masked tokens)
        valid_entropies = student_entropy[response_mask.bool()]

        if valid_entropies.numel() > 0:
            # Compute percentiles on the valid entropies
            start_percentile_entropy = torch.quantile(valid_entropies, entropy_start_pct)
            end_percentile_entropy = torch.quantile(valid_entropies, entropy_end_pct)

            # Create mask for tokens within the percentile range
            entropy_mask = (student_entropy >= start_percentile_entropy) & (
                student_entropy <= end_percentile_entropy
            )

            # Store original number of tokens before filtering for logging
            tokens_before_entropy_filtering = response_mask.sum().item()

            # Combine with existing response_mask (use multiplication for float masks)
            response_mask = response_mask * entropy_mask.float()

            tokens_after_entropy_filtering = response_mask.sum().item()

            print("########################################")
            print(f"entropy_start_pct: {entropy_start_pct}")
            print(f"entropy_end_pct: {entropy_end_pct}")
            print(f"start_percentile (entropy): {start_percentile_entropy.item():.4f}")
            print(f"end_percentile (entropy): {end_percentile_entropy.item():.4f}")
            print(f"tokens_before_entropy_filtering: {tokens_before_entropy_filtering}")
            print(f"tokens_after_entropy_filtering: {tokens_after_entropy_filtering}")
            print(
                f"entropy_filtering_ratio: {tokens_after_entropy_filtering / tokens_before_entropy_filtering:.2%}"
            )
            print("########################################")

    # Set up clip ranges
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    lower_bound = 1.0 - cliprange_low
    upper_bound = 1.0 + cliprange_high

    # PPO-style clipped loss for distillation
    # Loss 1: Direct KL divergence (unclipped)
    losses1 = kl_divergence

    # Loss 2: KL divergence weighted by clipped ratio
    # This prevents the student from changing too much in a single update
    clipped_ratio = torch.clamp(ratio, lower_bound, upper_bound)

    # Scale the KL loss by the clipped ratio
    # When ratio is clipped, we reduce the gradient signal
    losses2 = kl_divergence * clipped_ratio / ratio.detach()

    # Take the max to get conservative updates (similar to PPO)
    clipped_losses = torch.max(losses1, losses2)

    # Aggregate loss
    distill_loss = agg_loss(
        loss_mat=clipped_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    # Compute metrics
    with torch.no_grad():
        # Clipping statistics
        clipped_low_mask = ratio < lower_bound
        clipped_high_mask = ratio > upper_bound
        print(f"lower_bound: {lower_bound}")
        print(f"upper_bound: {upper_bound}")
        print(f"ratio: {ratio}")
        clipfrac = masked_mean((clipped_low_mask | clipped_high_mask).float(), response_mask)
        clipfrac_low = masked_mean(clipped_low_mask.float(), response_mask)
        clipfrac_high = masked_mean(clipped_high_mask.float(), response_mask)

        # Policy change statistics
        approx_kl = masked_mean(-log_ratio, response_mask)  # KL between old and new student
        avg_ratio = masked_mean(ratio, response_mask)

        # Distribution statistics
        mean_kl_per_token = masked_mean(kl_divergence, response_mask)
        student_perplexity = torch.exp(masked_mean(-student_log_probs, response_mask))
        teacher_perplexity = torch.exp(masked_mean(-teacher_log_probs, response_mask))
        old_student_perplexity = torch.exp(masked_mean(-old_student_log_probs, response_mask))

        # Difference metrics
        log_prob_diff = torch.abs(student_log_probs - teacher_log_probs)
        mean_log_prob_diff = masked_mean(log_prob_diff, response_mask)
        max_log_prob_diff = (log_prob_diff * response_mask).max().item()

        # Token range statistics (useful when using partial token training)
        num_trained_tokens = response_mask.sum().item()
        avg_trained_tokens_per_sample = num_trained_tokens / batch_size

        # Advantage statistics (for advantage-based selection)
        if response_mask.sum() > 0:
            valid_advantages = kl_divergence[response_mask.bool()]
            advantage_mean = valid_advantages.mean().item()
            advantage_std = valid_advantages.std().item()
            advantage_min = valid_advantages.min().item()
            advantage_max = valid_advantages.max().item()
        else:
            advantage_mean = 0.0
            advantage_std = 0.0
            advantage_min = 0.0
            advantage_max = 0.0

        # Entropy statistics (for entropy-based selection)
        if student_entropy is not None and response_mask.sum() > 0:
            valid_entropies = student_entropy[response_mask.bool()]
            entropy_mean = valid_entropies.mean().item()
            entropy_std = valid_entropies.std().item()
            entropy_min = valid_entropies.min().item()
            entropy_max = valid_entropies.max().item()
        else:
            entropy_mean = 0.0
            entropy_std = 0.0
            entropy_min = 0.0
            entropy_max = 0.0

    metrics = {
        "distill/reverse_kl": mean_kl_per_token.item(),
        "distill/clipfrac": clipfrac.item(),
        "distill/clipfrac_low": clipfrac_low.item(),
        "distill/clipfrac_high": clipfrac_high.item(),
        "distill/approx_kl": approx_kl.item(),  # KL(new_student || old_student)
        "distill/avg_ratio": avg_ratio.item(),
        "distill/student_perplexity": student_perplexity.item(),
        "distill/teacher_perplexity": teacher_perplexity.item(),
        "distill/old_student_perplexity": old_student_perplexity.item(),
        "distill/mean_log_prob_diff": mean_log_prob_diff.item(),
        "distill/max_log_prob_diff": max_log_prob_diff,
        "distill/num_trained_tokens": num_trained_tokens,
        "distill/avg_trained_tokens_per_sample": avg_trained_tokens_per_sample,
        "distill/train_token_start_pct": train_token_start_pct,
        "distill/train_token_end_pct": train_token_end_pct,
        "distill/advantage_start_pct": advantage_start_pct,
        "distill/advantage_end_pct": advantage_end_pct,
        "distill/advantage_mean": advantage_mean,
        "distill/advantage_std": advantage_std,
        "distill/advantage_min": advantage_min,
        "distill/advantage_max": advantage_max,
        "distill/entropy_start_pct": entropy_start_pct,
        "distill/entropy_end_pct": entropy_end_pct,
        "distill/entropy_mean": entropy_mean,
        "distill/entropy_std": entropy_std,
        "distill/entropy_min": entropy_min,
        "distill/entropy_max": entropy_max,
    }

    return distill_loss, metrics
