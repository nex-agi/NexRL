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

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .core_utils import masked_mean


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    lbpo_alpha: Optional[float] = None,
) -> torch.Tensor:
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)` shape: (bs, response_length)
        loss_mask: `(torch.Tensor)` shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            - "token-mean"                 → DAPO-style
            - "seq-mean-token-sum"         → GRPO-style
            - "seq-mean-token-mean"        → same as above in practice
            - "max-length-mean"            → seq mean with denom = max_response_length
            - "lbpo"                       → Length-Balanced PO (new)
        lbpo_alpha: (float, optional) only used when loss_agg_mode == "lbpo"
                    Recommended: 0.5 (geometric balance), 0.0 → GRPO, 1.0 → DAPO

    Returns:
        loss: scalar torch.Tensor
    """
    if loss_agg_mode == "token-mean":
        # DAPO: uniform weight per valid token
        loss = masked_mean(loss_mat, loss_mask)

    elif loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-mean"):
        # GRPO: average per sequence, then average sequences
        seq_sum = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_len = torch.sum(loss_mask, dim=-1)
        seq_mean = seq_sum / (seq_len + 1e-8)  # (bs,)
        loss = torch.mean(seq_mean)

    elif loss_agg_mode == "max-length-mean":
        # Dr.GRPO：Normalize each sequence by configured max_response_length (padding length)
        seq_sum = torch.sum(loss_mat * loss_mask, dim=-1)
        denom = torch.tensor(loss_mat.size(-1), dtype=loss_mat.dtype, device=loss_mat.device)
        seq_mean = seq_sum / (denom + 1e-8)
        loss = torch.mean(seq_mean)

    elif loss_agg_mode == "lbpo":
        if lbpo_alpha is None:
            raise ValueError("lbpo_alpha must be provided when loss_agg_mode='lbpo'")

        # Compute effective lengths: L_i = sum(mask_i)
        L = torch.sum(loss_mask, dim=-1)  # (bs,)
        eps = 1e-8
        L = L.clamp(min=eps)

        # Per-token weight: w_{i,t} ∝ L_i^{α - 1}
        token_weights = torch.pow(L, lbpo_alpha - 1.0).unsqueeze(-1)  # (bs, 1)
        seq_weights = torch.pow(L, lbpo_alpha)  # (bs,)
        total_weight = torch.sum(seq_weights)  # scalar

        weighted_loss = loss_mat * loss_mask * token_weights  # (bs, T)
        loss = torch.sum(weighted_loss) / (total_weight + eps)

    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


_ICEPOP_CFG_VALUE_MISSING = object()


def _maybe_get_icepop_cfg_value(cfg: Any, key: str):
    if isinstance(cfg, Mapping) and key in cfg:
        return cfg[key]
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]  # type: ignore[index]
    except Exception:
        return _ICEPOP_CFG_VALUE_MISSING


def _resolve_icepop_cfg_value(cfg: Any, *names: str):
    for name in names:
        value = _maybe_get_icepop_cfg_value(cfg, name)
        if value is not _ICEPOP_CFG_VALUE_MISSING and value is not None:
            return value
    raise KeyError(names[0])


def _has_icepop_cfg_value(cfg: Any, *names: str) -> bool:
    for name in names:
        value = _maybe_get_icepop_cfg_value(cfg, name)
        if value is not _ICEPOP_CFG_VALUE_MISSING and value is not None:
            return True
    return False


def _parse_icepop_cfg(cfg: Any) -> Dict[str, float]:
    if cfg is None:
        raise ValueError("config.icepop must be provided when loss_func_type='icepop'.")
    try:
        alpha = float(_resolve_icepop_cfg_value(cfg, "alpha", "mask_alpha", "lower", "k_min"))
        beta = float(_resolve_icepop_cfg_value(cfg, "beta", "mask_beta", "upper", "k_max"))
    except KeyError as exc:
        if _has_icepop_cfg_value(
            cfg,
            "lambda",
            "lam",
            "lmbda",
            "consistency_lambda",
            "sscr_lambda",
        ):
            raise ValueError(
                "Missing IcePop config value: "
                f"{exc.args[0]}. loss_func_type='icepop' requires alpha/beta; "
                "lambda belongs to loss_func_type='sscr_icepop'."
            ) from exc
        raise ValueError(f"Missing IcePop config value: {exc.args[0]}") from exc

    if alpha > beta:
        raise ValueError(f"Invalid IcePop mask range: alpha ({alpha}) > beta ({beta}).")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError(f"IcePop alpha/beta must be > 0, got alpha={alpha}, beta={beta}.")
    return {"alpha": alpha, "beta": beta}


def _parse_sscr_icepop_cfg(cfg: Any) -> Dict[str, float]:
    if cfg is None:
        raise ValueError("config.icepop must be provided when loss_func_type='sscr_icepop'.")
    try:
        consistency_lambda = float(
            _resolve_icepop_cfg_value(
                cfg,
                "lambda",
                "lam",
                "lmbda",
                "consistency_lambda",
                "sscr_lambda",
            )
        )
    except KeyError as exc:
        raise ValueError(f"Missing S-SCR-IcePop config value: {exc.args[0]}") from exc

    if consistency_lambda < 0.0:
        raise ValueError(f"S-SCR-IcePop lambda must be >= 0, got lambda={consistency_lambda}.")
    return {"lambda": consistency_lambda}


_SCGRPO_CFG_VALUE_MISSING = object()


def _maybe_get_scgrpo_cfg_value(cfg: Any, key: str):
    if isinstance(cfg, Mapping) and key in cfg:
        return cfg[key]
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]  # type: ignore[index]
    except Exception:
        return _SCGRPO_CFG_VALUE_MISSING


def _resolve_scgrpo_cfg_value(cfg: Any, *names: str):
    for name in names:
        value = _maybe_get_scgrpo_cfg_value(cfg, name)
        if value is not _SCGRPO_CFG_VALUE_MISSING and value is not None:
            return value
    raise KeyError(names[0])


def _parse_scgrpo_cfg(cfg: Any) -> Dict[str, float]:
    if cfg is None:
        raise ValueError("config.scgrpo must be provided when loss_func_type='scgrpo'.")
    try:
        alpha_out = float(_resolve_scgrpo_cfg_value(cfg, "alpha_out", "alpha1", "alpha_outer"))
        alpha_in = float(_resolve_scgrpo_cfg_value(cfg, "alpha_in", "alpha2", "alpha_inner"))
    except KeyError as exc:
        raise ValueError(f"Missing SCGRPO config value: {exc.args[0]}") from exc

    if alpha_out <= 0.0 or alpha_in <= 0.0:
        raise ValueError(
            f"SCGRPO alpha_out/alpha_in must be > 0, got alpha_out={alpha_out}, alpha_in={alpha_in}."
        )
    return {"alpha_out": alpha_out, "alpha_in": alpha_in}


_SAPO_CFG_VALUE_MISSING = object()


def _maybe_get_sapo_cfg_value(cfg: Any, key: str):
    if isinstance(cfg, Mapping) and key in cfg:
        return cfg[key]
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]  # type: ignore[index]
    except Exception:
        return _SAPO_CFG_VALUE_MISSING


def _resolve_sapo_cfg_value(cfg: Any, *names: str):
    for name in names:
        value = _maybe_get_sapo_cfg_value(cfg, name)
        if value is not _SAPO_CFG_VALUE_MISSING and value is not None:
            return value
    raise KeyError(names[0])


def _parse_sapo_cfg(cfg: Any) -> Dict[str, float]:
    if cfg is None:
        raise ValueError("config.sapo must be provided when loss_func_type='sapo'.")
    try:
        tau_pos = float(_resolve_sapo_cfg_value(cfg, "tau_pos", "tau_positive", "tau_p"))
        tau_neg = float(_resolve_sapo_cfg_value(cfg, "tau_neg", "tau_negative", "tau_n"))
    except KeyError as exc:
        raise ValueError(f"Missing SAPO config value: {exc.args[0]}") from exc

    if tau_pos <= 0.0 or tau_neg <= 0.0:
        raise ValueError(
            f"SAPO tau_pos/tau_neg must be > 0, got tau_pos={tau_pos}, tau_neg={tau_neg}."
        )
    return {"tau_pos": tau_pos, "tau_neg": tau_neg}


def compute_policy_loss_icepop(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    infer_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    alpha: float,
    beta: float,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict]:
    """IcePop policy-gradient loss.

    Implements (token-wise):
      k = π_train(·; θ_old) / π_infer(·; θ_old)
      M(k) = k if k in [alpha, beta] else 0
      L = - M(k) * min(r * A, clip(r) * A)
      r = π_train(·; θ) / π_train(·; θ_old)
    """
    # ratio r = π_θ / π_θ_old
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

    # PPO clipped surrogate objectives (loss form: -min(...))
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, lower_bound, upper_bound)
    clip_pg_losses = torch.max(pg_losses, pg_losses2)

    # IcePop mask M(k)
    log_k = (old_log_prob - infer_log_prob).to(torch.float32)
    log_k = log_k.clamp(min=-50.0, max=50.0)
    k = torch.exp(log_k).to(log_prob.dtype)
    in_range = (k >= alpha) & (k <= beta)
    m = k * in_range.to(log_prob.dtype)

    icepop_losses = clip_pg_losses * m
    pg_loss = agg_loss(loss_mat=icepop_losses, loss_mask=eos_mask, loss_agg_mode=loss_agg_mode)

    mask_float = eos_mask.to(log_prob.dtype)
    in_range_frac = masked_mean(in_range.to(log_prob.dtype), mask_float)
    m_mean = masked_mean(m, mask_float)
    k_mean = masked_mean(k, mask_float)
    k_mean_in_range = masked_mean(k, (in_range & eos_mask.bool()).to(log_prob.dtype))

    return pg_loss, {
        "actor/pg_clipfrac": masked_mean(
            ((ratio < lower_bound) | (ratio > upper_bound)).to(log_prob.dtype), eos_mask
        )
        .detach()
        .item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/avg_ratio": masked_mean(ratio, eos_mask).detach().item(),
        "actor/icepop_alpha": float(alpha),
        "actor/icepop_beta": float(beta),
        "actor/icepop_in_range_frac": in_range_frac.detach().item(),
        "actor/icepop_m_mean": m_mean.detach().item(),
        "actor/icepop_k_mean": k_mean.detach().item(),
        "actor/icepop_k_mean_in_range": k_mean_in_range.detach().item(),
    }


def compute_policy_loss_sscr_icepop(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    infer_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    consistency_lambda: float,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict]:
    """Sentence-Level Soft Consistency Reweighted IcePop (S-SCR-IcePop).

    Keeps PPO clip structure unchanged (token-wise) but replaces IcePop's token-level hard mask with a
    sentence-level, symmetric, length-normalized soft weight:

      Δ_i = | (1/|y_i|) * Σ_t (log π_train_old - log π_infer_old) |
      w_raw_i = exp(-λ Δ_i)
      w_i = w_raw_i / mean(w_raw)

    Final policy loss (loss form):
      L = mean_i [ w_i * mean_t(clip_pg_loss_{i,t}) ]
    """
    if consistency_lambda < 0.0:
        raise ValueError(f"consistency_lambda must be >= 0, got {consistency_lambda}.")

    eps = 1e-8

    # ratio r = π_θ / π_θ_old
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

    # PPO clipped surrogate objectives (loss form: -min(...))
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, lower_bound, upper_bound)
    clip_pg_losses = torch.max(pg_losses, pg_losses2)  # (bs, T)

    # Sentence-level, symmetric, length-normalized discrepancy Δ_i
    token_mask_f32 = eos_mask.to(torch.float32)
    seq_len = token_mask_f32.sum(dim=-1)  # (bs,)
    train_ll = (old_log_prob.to(torch.float32) * token_mask_f32).sum(dim=-1)
    infer_ll = (infer_log_prob.to(torch.float32) * token_mask_f32).sum(dim=-1)
    delta = torch.abs((train_ll - infer_ll) / (seq_len + eps))  # (bs,)

    # Soft weights with batch-mean normalization (mean weight = 1)
    log_w_raw = (-float(consistency_lambda) * delta).clamp(min=-50.0, max=0.0)
    w_raw = torch.exp(log_w_raw)  # (bs,)
    z = w_raw.mean()
    w = w_raw / (z + eps)  # (bs,)

    # Sentence-mean token loss, then sentence reweight
    token_mask_loss = eos_mask.to(clip_pg_losses.dtype)
    seq_mean_loss = (clip_pg_losses * token_mask_loss).sum(dim=-1) / (
        token_mask_loss.sum(dim=-1) + eps
    )
    pg_loss = torch.mean(seq_mean_loss * w.to(seq_mean_loss.dtype))

    return pg_loss, {
        "actor/pg_clipfrac": masked_mean(
            ((ratio < lower_bound) | (ratio > upper_bound)).to(log_prob.dtype), eos_mask
        )
        .detach()
        .item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/avg_ratio": masked_mean(ratio, eos_mask).detach().item(),
        "actor/sscr_icepop_lambda": float(consistency_lambda),
        "actor/sscr_icepop_delta_mean": delta.mean().detach().item(),
        "actor/sscr_icepop_delta_max": delta.max().detach().item(),
        "actor/sscr_icepop_w_raw_mean": z.detach().item(),
        "actor/sscr_icepop_w_mean": w.mean().detach().item(),
        "actor/sscr_icepop_w_min": w.min().detach().item(),
        "actor/sscr_icepop_w_max": w.max().detach().item(),
    }


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


def compute_policy_loss_scgrpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    alpha_out: float,
    alpha_in: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict]:
    """SCGRPO policy-gradient loss (no hard clip; smooth ratio transform).

    Uses ratio r = exp(log_prob - old_log_prob), then applies:
        g(r) = exp(alpha_out * tanh(log(r) / alpha_in))
    where alpha_out > 0 and alpha_in > 0 are log-space soft-clip parameters.

    Loss (token-wise):
        L = - advantages * g(r)
    """
    if alpha_out <= 0.0:
        raise ValueError("SCGRPO alpha_out must be > 0.")
    if alpha_in <= 0.0:
        raise ValueError("SCGRPO alpha_in must be > 0.")

    # ratio = π_θ / π_θ_old (in log-space first for stability)
    log_ratio = (log_prob - old_log_prob).to(torch.float32)
    ppo_kl = masked_mean(-log_ratio, eos_mask)

    # Smoothly clip log_ratio via tanh in log-space
    g_log_ratio = alpha_out * torch.tanh(log_ratio / alpha_in)
    g_ratio = torch.exp(g_log_ratio).to(log_prob.dtype)
    raw_ratio = torch.exp(log_ratio).to(log_prob.dtype)

    # SCGRPO surrogate (no min/clip)
    pg_losses = -advantages * g_ratio

    avg_ratio = masked_mean(g_ratio, eos_mask)
    avg_ratio_raw = masked_mean(raw_ratio, eos_mask)

    # ===== advantage monitoring =====
    adv_pos_mask = (advantages > 0) & eos_mask.bool()
    adv_neg_mask = (advantages < 0) & eos_mask.bool()
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()
    avg_ratio_pos = masked_mean(g_ratio, adv_pos_mask.to(log_prob.dtype))
    avg_ratio_neg = masked_mean(g_ratio, adv_neg_mask.to(log_prob.dtype))

    # ===== equality monitoring =====
    eq_mask = (old_log_prob == log_prob) & eos_mask.bool()
    equal_count = eq_mask.sum()

    # ===== agg loss =====
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=eos_mask, loss_agg_mode=loss_agg_mode)

    # ===== adv-specific losses =====
    adv_pos_loss = masked_mean(pg_losses, adv_pos_mask.to(log_prob.dtype))
    adv_neg_loss = masked_mean(pg_losses, adv_neg_mask.to(log_prob.dtype))

    return pg_loss, {
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/avg_ratio": avg_ratio.detach().item(),
        "actor/scgrpo_alpha": float(alpha_out),
        "actor/scgrpo_alpha_out": float(alpha_out),
        "actor/scgrpo_alpha_in": float(alpha_in),
        "actor/scgrpo_avg_ratio_raw": avg_ratio_raw.detach().item(),
        "actor/adv_pos_count": adv_pos_count.detach().item(),
        "actor/adv_neg_count": adv_neg_count.detach().item(),
        "actor/avg_ratio_pos": avg_ratio_pos.detach().item(),
        "actor/avg_ratio_neg": avg_ratio_neg.detach().item(),
        "actor/old_logprob_and_logprob_equal_count": equal_count.detach().item(),
        "actor/adv_pos_loss": adv_pos_loss.detach().item(),
        "actor/adv_neg_loss": adv_neg_loss.detach().item(),
    }


def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    tau_pos: float,
    tau_neg: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict]:
    """SAPO policy-gradient loss.

    Objective (token-wise) uses:
      r = π_θ / π_θ_old
      tau = tau_pos if A > 0 else tau_neg
      f(r) = sigmoid(tau * (r - 1)) * (4 / tau)
      J = E[f(r) * A]

    We return the minimization loss: L = -f(r) * A.
    """
    if tau_pos <= 0.0:
        raise ValueError("SAPO tau_pos must be > 0.")
    if tau_neg <= 0.0:
        raise ValueError("SAPO tau_neg must be > 0.")

    log_ratio_fp32 = (log_prob - old_log_prob).to(torch.float32)
    ratio = torch.exp(log_ratio_fp32).to(log_prob.dtype)
    ppo_kl = masked_mean(-log_ratio_fp32, eos_mask)

    tau = torch.where(
        advantages > 0,
        torch.tensor(tau_pos, dtype=log_prob.dtype, device=log_prob.device),
        torch.tensor(tau_neg, dtype=log_prob.dtype, device=log_prob.device),
    )

    # f(r) = sigmoid(tau*(r-1)) * 4/tau
    z = tau * (ratio - 1.0)
    f = torch.sigmoid(z.to(torch.float32)).to(log_prob.dtype) * (4.0 / tau)

    loss_mat = -(f * advantages)
    pg_loss = agg_loss(loss_mat=loss_mat, loss_mask=eos_mask, loss_agg_mode=loss_agg_mode)

    mask_f = eos_mask.to(log_prob.dtype)
    avg_ratio = masked_mean(ratio, eos_mask)
    avg_f = masked_mean(f, mask_f)
    avg_abs_z = masked_mean(z.abs(), mask_f)

    adv_pos_mask = (advantages > 0) & eos_mask.bool()
    adv_neg_mask = (advantages <= 0) & eos_mask.bool()
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()
    avg_f_pos = masked_mean(f, adv_pos_mask.to(log_prob.dtype))
    avg_f_neg = masked_mean(f, adv_neg_mask.to(log_prob.dtype))

    return pg_loss, {
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/avg_ratio": avg_ratio.detach().item(),
        "actor/sapo_tau_pos": float(tau_pos),
        "actor/sapo_tau_neg": float(tau_neg),
        "actor/sapo_f_mean": avg_f.detach().item(),
        "actor/sapo_abs_z_mean": avg_abs_z.detach().item(),
        "actor/sapo_f_mean_pos": avg_f_pos.detach().item(),
        "actor/sapo_f_mean_neg": avg_f_neg.detach().item(),
        "actor/adv_pos_count": adv_pos_count.detach().item(),
        "actor/adv_neg_count": adv_neg_count.detach().item(),
    }


def compute_policy_loss_grpo_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict]:
    """Compute PPO policy-gradient loss with curved clipping bounds."""
    # ratio = π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

    # Curved clip bounds based on current probability x = exp(log_prob):
    # y_lower = (3 - sqrt(9 - 8x)) / 2
    # y_upper = x + 0.5*x*(1-x)
    # ratio = x / y should be in [x / y_upper, x / y_lower]
    cur_prob = torch.exp(log_prob)
    eps = 1e-12
    x = cur_prob.clamp(min=eps, max=1.0)
    upper_y = 1.2 * x
    lower_y = 0.8 * x
    lower_bound = x / upper_y.clamp(min=eps)
    upper_bound = x / lower_y.clamp(min=eps)

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


def compute_policy_loss_lbpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
    loss_agg_mode: str = "token-mean",
    lbpo_alpha: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute PPO policy-gradient loss with detailed monitoring and LBPO support.

    Args:
        old_log_prob, log_prob, advantages, eos_mask: all (bs, seq_len)
        cliprange: fallback clip range
        low_clip_range, high_clip_range: asymmetric clipping bounds
        loss_agg_mode: aggregation mode
        lbpo_alpha: alpha for LBPO (required if loss_agg_mode=='lbpo')

    Returns:
        pg_loss: scalar loss
        metrics: dict of logging scalars
    """
    # --- Core PPO computation ---
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

    if low_clip_range > 0 and high_clip_range > 0:
        lower_bound = 1.0 - low_clip_range
        upper_bound = 1.0 + high_clip_range
    else:
        lower_bound = 1.0 - cliprange
        upper_bound = 1.0 + cliprange

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, lower_bound, upper_bound)
    clip_pg_losses = torch.max(pg_losses, pg_losses2)

    # --- Clipping stats ---
    clipped_low_mask = ratio < lower_bound
    clipped_high_mask = ratio > upper_bound
    pg_clipfrac = masked_mean((clipped_low_mask | clipped_high_mask).float(), eos_mask)
    pg_clipfrac_low = masked_mean(clipped_low_mask.float(), eos_mask)
    pg_clipfrac_high = masked_mean(clipped_high_mask.float(), eos_mask)
    avg_ratio = masked_mean(ratio, eos_mask)

    # --- Advantage-based monitoring ---
    adv_pos_mask = (advantages > 0) & eos_mask.bool()
    adv_neg_mask = (advantages < 0) & eos_mask.bool()
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()
    avg_ratio_pos = masked_mean(ratio, adv_pos_mask.float())
    avg_ratio_neg = masked_mean(ratio, adv_neg_mask.float())

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

    # --- Equality check ---
    eq_mask = (old_log_prob == log_prob) & eos_mask.bool()
    equal_count = eq_mask.sum()

    if loss_agg_mode == "lbpo" and lbpo_alpha is None:
        raise ValueError("lbpo_alpha must be provided when loss_agg_mode='lbpo'")

    # --- Loss aggregation ---
    if loss_agg_mode == "lbpo":
        pg_loss = agg_loss(
            loss_mat=clip_pg_losses,
            loss_mask=eos_mask,
            loss_agg_mode=loss_agg_mode,
            lbpo_alpha=lbpo_alpha,
        )
    else:
        pg_loss = agg_loss(
            loss_mat=clip_pg_losses,
            loss_mask=eos_mask,
            loss_agg_mode=loss_agg_mode,
        )

    # --- Additional advantage loss stats ---
    pos_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_pos_mask).sum()
    neg_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_neg_mask).sum()
    adv_pos_loss = masked_mean(clip_pg_losses, adv_pos_mask.float())
    adv_neg_loss = masked_mean(clip_pg_losses, adv_neg_mask.float())

    kept_mask = (~clipped_low_mask & ~clipped_high_mask) & eos_mask.bool()
    adv_pos_kept_count = (kept_mask & adv_pos_mask).sum()
    adv_neg_kept_count = (kept_mask & adv_neg_mask).sum()

    # --- Final metric dict ---
    metrics = {
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

    if loss_agg_mode == "lbpo":
        assert lbpo_alpha is not None
        metrics["actor/lbpo_alpha"] = float(lbpo_alpha)

    return pg_loss, metrics


def compute_policy_loss_bapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    init_lower_bound: float,
    init_upper_bound: float,
    max_lower_bound: float,
    max_upper_bound: float,
    lower_step: float,
    upper_step: float,
    positive_contribution_threshold: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict]:
    """Balanced Advantage Policy Optimization (BAPO).

    Dynamically expands the asymmetric clipping bounds until the ratio of
    positive token contributions exceeds ``positive_contribution_threshold``.

    The clipping bounds operate directly on the probability ratio ``r``. For
    example ``init_lower_bound=0.8`` corresponds to ``1 - 0.2`` in PPO.

    Args:
        old_log_prob: `(bs, T)` old policy log probs.
        log_prob: `(bs, T)` current policy log probs.
        advantages: `(bs, T)` token advantages.
        eos_mask: `(bs, T)` mask for valid tokens.
        init_lower_bound: initial lower clip bound ``c_low`` (`a⁻`).
        init_upper_bound: initial upper clip bound ``c_high`` (`a⁺`).
        max_lower_bound: max reachable lower bound (`b⁻`).
        max_upper_bound: max reachable upper bound (`b⁺`).
        lower_step: increment for the lower bound (``δ₂`` > 0).
        upper_step: increment for the upper bound (``δ₁`` > 0).
        positive_contribution_threshold: target ratio ``ρ₀`` in [0, 1].
        loss_agg_mode: aggregation strategy for the loss.

    Returns:
        Tuple of `(loss, metrics)`.
    """

    if lower_step <= 0.0:
        raise ValueError("lower_step must be > 0 for BAPO")
    if upper_step <= 0.0:
        raise ValueError("upper_step must be > 0 for BAPO")
    if init_lower_bound > max_lower_bound:
        raise ValueError("init_lower_bound must be <= max_lower_bound")
    if init_upper_bound > max_upper_bound:
        raise ValueError("init_upper_bound must be <= max_upper_bound")
    if not (0.0 <= positive_contribution_threshold <= 1.0):
        raise ValueError("positive_contribution_threshold must be in [0, 1]")

    mask_float = eos_mask.to(log_prob.dtype)
    neg_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(neg_approx_kl)
    ppo_kl = masked_mean(-neg_approx_kl, eos_mask)

    ratio_times_adv = ratio * advantages
    eps = 1e-8

    c_low = float(init_lower_bound)
    c_high = float(init_upper_bound)

    def _positive_ratio_value(low_bound: float, high_bound: float) -> float:
        clipped_ratio = torch.clamp(ratio, min=low_bound, max=high_bound)
        surrogate = torch.min(ratio_times_adv, clipped_ratio * advantages)
        masked_surrogate = surrogate * mask_float
        masked_surrogate_fp32 = masked_surrogate.to(torch.float32)
        total = masked_surrogate_fp32.abs().sum()
        total_value = float(total.detach().cpu())
        if total_value <= eps:
            return 1.0
        positive = torch.clamp(masked_surrogate_fp32, min=0.0).sum()
        ratio_value = positive / (total + eps)
        return float(ratio_value.detach().cpu())

    pos_ratio_value = _positive_ratio_value(c_low, c_high)
    adjust_steps = 0

    while (
        pos_ratio_value < positive_contribution_threshold
        and c_low + lower_step <= max_lower_bound + 1e-12
    ):
        if c_high + upper_step <= max_upper_bound + 1e-12:
            c_high = min(c_high + upper_step, max_upper_bound)
        elif c_low + lower_step <= max_lower_bound + 1e-12:
            c_low = min(c_low + lower_step, max_lower_bound)
        else:
            break
        adjust_steps += 1
        pos_ratio_value = _positive_ratio_value(c_low, c_high)

    clipped_ratio = torch.clamp(ratio, min=c_low, max=c_high)
    clipped_ratio_times_adv = clipped_ratio * advantages
    surrogate = torch.min(ratio_times_adv, clipped_ratio_times_adv)
    clip_pg_losses = -surrogate

    pg_loss = agg_loss(
        loss_mat=clip_pg_losses,
        loss_mask=eos_mask,
        loss_agg_mode=loss_agg_mode,
    )

    clipped_low_mask = ratio < c_low
    clipped_high_mask = ratio > c_high
    pg_clipfrac = masked_mean((clipped_low_mask | clipped_high_mask).float(), eos_mask)
    pg_clipfrac_low = masked_mean(clipped_low_mask.float(), eos_mask)
    pg_clipfrac_high = masked_mean(clipped_high_mask.float(), eos_mask)
    avg_ratio = masked_mean(ratio, eos_mask)

    adv_pos_mask = (advantages > 0) & eos_mask.bool()
    adv_neg_mask = (advantages < 0) & eos_mask.bool()
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()
    avg_ratio_pos = masked_mean(ratio, adv_pos_mask.float())
    avg_ratio_neg = masked_mean(ratio, adv_neg_mask.float())

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

    eq_mask = (old_log_prob == log_prob) & eos_mask.bool()
    equal_count = eq_mask.sum()

    pos_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_pos_mask).sum()
    neg_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_neg_mask).sum()
    adv_pos_loss = masked_mean(clip_pg_losses, adv_pos_mask.float())
    adv_neg_loss = masked_mean(clip_pg_losses, adv_neg_mask.float())

    kept_mask = (~clipped_low_mask & ~clipped_high_mask) & eos_mask.bool()
    adv_pos_kept_count = (kept_mask & adv_pos_mask).sum()
    adv_neg_kept_count = (kept_mask & adv_neg_mask).sum()

    metrics = {
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
        "actor/bapo_positive_contribution": pos_ratio_value,
        "actor/bapo_c_low": c_low,
        "actor/bapo_c_high": c_high,
        "actor/bapo_adjust_steps": float(adjust_steps),
        "actor/bapo_pos_threshold": positive_contribution_threshold,
    }

    return pg_loss, metrics


def compute_policy_loss_ce_gppo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
    beta1: float,
    beta2: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict]:
    """
    Cross-Entropy guided PPO loss.

    Applies ratio scaling outside the clip range with separate weights for
    negative and positive advantages.
    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

    if low_clip_range > 0 and high_clip_range > 0:
        lower_bound = 1.0 - low_clip_range
        upper_bound = 1.0 + high_clip_range
    else:
        lower_bound = 1.0 - cliprange
        upper_bound = 1.0 + cliprange

    ratio_sg = ratio.detach()
    eps = 1e-8

    cond_low = (ratio < lower_bound) & (advantages < 0)
    cond_high = (ratio > upper_bound) & (advantages > 0)

    scale = torch.ones_like(ratio)
    scale = torch.where(
        cond_low,
        beta1 * lower_bound / (ratio_sg + eps),
        scale,
    )
    scale = torch.where(
        cond_high,
        beta2 * upper_bound / (ratio_sg + eps),
        scale,
    )

    l_token = scale * ratio * advantages
    per_token_loss = -l_token
    pg_loss = agg_loss(per_token_loss, eos_mask, loss_agg_mode=loss_agg_mode)

    clipped_low_mask = ratio < lower_bound
    clipped_high_mask = ratio > upper_bound
    pg_clipfrac = masked_mean((clipped_low_mask | clipped_high_mask).float(), eos_mask)
    pg_clipfrac_low = masked_mean(clipped_low_mask.float(), eos_mask)
    pg_clipfrac_high = masked_mean(clipped_high_mask.float(), eos_mask)
    avg_ratio = masked_mean(ratio, eos_mask)

    adv_pos_mask = (advantages > 0) & eos_mask.bool()
    adv_neg_mask = (advantages < 0) & eos_mask.bool()
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()
    avg_ratio_pos = masked_mean(ratio, adv_pos_mask.float())
    avg_ratio_neg = masked_mean(ratio, adv_neg_mask.float())

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

    eq_mask = (old_log_prob == log_prob) & eos_mask.bool()
    equal_count = eq_mask.sum()

    pos_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_pos_mask).sum()
    neg_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_neg_mask).sum()
    adv_pos_loss = masked_mean(per_token_loss, adv_pos_mask.float())
    adv_neg_loss = masked_mean(per_token_loss, adv_neg_mask.float())

    kept_mask = (~clipped_low_mask & ~clipped_high_mask) & eos_mask.bool()
    adv_pos_kept_count = (kept_mask & adv_pos_mask).sum()
    adv_neg_kept_count = (kept_mask & adv_neg_mask).sum()

    def _masked_mean_selected(values: torch.Tensor, selector: torch.Tensor):
        sel = selector.bool()
        if sel.any():
            return values.masked_select(sel).mean()
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)

    ce_loss_low = _masked_mean_selected(l_token, cond_low)
    ce_loss_high = _masked_mean_selected(l_token, cond_high)

    metrics = {
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
        "actor/ce_gppo_l_low": ce_loss_low.detach().item(),
        "actor/ce_gppo_l_high": ce_loss_high.detach().item(),
    }

    return pg_loss, metrics


def compute_policy_loss_ce_gppo_mask(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
    beta1: float,
    beta2: float,
    loss_agg_mode: str = "token-mean",
    adv_mask_delta: float = 0.05,
    return_masks: bool = False,
) -> Tuple[torch.Tensor, Dict, Optional[Dict[str, torch.Tensor]]]:
    """Cross-Entropy guided PPO loss with GRPO-style trace masking.

    Args:
        return_masks: If True, returns mask matrices as third return value.

    Returns:
        pg_loss: Policy gradient loss
        metrics: Dictionary of metrics
        mask_dict: Optional dictionary containing mask matrices:
            - "clipped_low_mask": (bs, seq_len) bool tensor for low-clipped tokens
            - "clipped_high_mask": (bs, seq_len) bool tensor for high-clipped tokens
            - "adv_mask_delta_mask": (bs, seq_len) bool tensor for adv_mask_delta masked tokens
    """

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    rev_log_ratio = -negative_approx_kl

    if low_clip_range > 0 and high_clip_range > 0:
        lower_bound = 1.0 - low_clip_range
        upper_bound = 1.0 + high_clip_range
    else:
        lower_bound = 1.0 - cliprange
        upper_bound = 1.0 + cliprange

    eos_bool_mask = eos_mask.bool()
    seq_rev_log_ratio_sum = torch.sum(rev_log_ratio * eos_mask, dim=-1)
    seq_lengths = torch.sum(eos_mask, dim=-1).clamp(min=1).to(log_prob.dtype)
    seq_rev_log_ratio_mean = seq_rev_log_ratio_sum / seq_lengths
    seq_over_delta = seq_rev_log_ratio_mean > adv_mask_delta
    seq_over_delta_mask = seq_over_delta.unsqueeze(-1)

    neg_adv_mask_all = (advantages < 0) & eos_bool_mask
    zero_mask = neg_adv_mask_all & seq_over_delta_mask

    mask_multiplier = (~zero_mask).to(log_prob.dtype)
    effective_mask = eos_mask * mask_multiplier
    effective_bool_mask = effective_mask.bool()

    ratio_sg = ratio.detach()
    eps = 1e-8

    cond_low = (ratio < lower_bound) & (advantages < 0) & effective_bool_mask
    cond_high = (ratio > upper_bound) & (advantages > 0) & effective_bool_mask

    scale = torch.ones_like(ratio)
    scale = torch.where(
        cond_low,
        beta1 * lower_bound / (ratio_sg + eps),
        scale,
    )
    scale = torch.where(
        cond_high,
        beta2 * upper_bound / (ratio_sg + eps),
        scale,
    )

    l_token = scale * ratio * advantages
    per_token_loss = -l_token
    pg_loss = agg_loss(
        loss_mat=per_token_loss,
        loss_mask=effective_mask,
        loss_agg_mode=loss_agg_mode,
    )

    ppo_kl = masked_mean(-negative_approx_kl, effective_mask)

    clipped_low_mask = ratio < lower_bound
    clipped_high_mask = ratio > upper_bound
    clip_any = clipped_low_mask | clipped_high_mask

    pg_clipfrac = masked_mean(clip_any.float(), effective_mask)
    pg_clipfrac_low = masked_mean(clipped_low_mask.float(), effective_mask)
    pg_clipfrac_high = masked_mean(clipped_high_mask.float(), effective_mask)
    avg_ratio = masked_mean(ratio, effective_mask)

    adv_pos_mask = (advantages > 0) & effective_bool_mask
    adv_neg_mask = (advantages < 0) & effective_bool_mask
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()

    adv_pos_mask_f = adv_pos_mask.float()
    adv_neg_mask_f = adv_neg_mask.float()

    avg_ratio_pos = masked_mean(ratio, adv_pos_mask_f)
    avg_ratio_neg = masked_mean(ratio, adv_neg_mask_f)

    adv_pos_clipfrac = masked_mean(clip_any.float(), adv_pos_mask_f)
    adv_pos_clipfrac_low = masked_mean(clipped_low_mask.float(), adv_pos_mask_f)
    adv_pos_clipfrac_high = masked_mean(clipped_high_mask.float(), adv_pos_mask_f)

    adv_neg_clipfrac = masked_mean(clip_any.float(), adv_neg_mask_f)
    adv_neg_clipfrac_low = masked_mean(clipped_low_mask.float(), adv_neg_mask_f)
    adv_neg_clipfrac_high = masked_mean(clipped_high_mask.float(), adv_neg_mask_f)

    eq_mask = (old_log_prob == log_prob) & effective_bool_mask
    equal_count = eq_mask.sum()

    pos_clip_count = (clip_any & adv_pos_mask).sum()
    neg_clip_count = (clip_any & adv_neg_mask).sum()

    adv_pos_loss = masked_mean(per_token_loss, adv_pos_mask_f)
    adv_neg_loss = masked_mean(per_token_loss, adv_neg_mask_f)

    kept_mask = (~clip_any) & effective_bool_mask
    adv_pos_kept_count = (kept_mask & adv_pos_mask).sum()
    adv_neg_kept_count = (kept_mask & adv_neg_mask).sum()

    def _masked_mean_selected(values: torch.Tensor, selector: torch.Tensor):
        sel = selector.bool()
        if sel.any():
            return values.masked_select(sel).mean()
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)

    ce_loss_low = _masked_mean_selected(l_token, cond_low)
    ce_loss_high = _masked_mean_selected(l_token, cond_high)

    adv_masked_tokens = zero_mask.sum().to(log_prob.dtype)
    total_tokens = eos_bool_mask.sum().clamp(min=1).to(log_prob.dtype)
    adv_mask_frac = adv_masked_tokens / total_tokens
    neg_adv_tokens = neg_adv_mask_all.sum().clamp(min=1).to(log_prob.dtype)
    adv_mask_frac_on_neg = adv_masked_tokens / neg_adv_tokens
    seq_over_delta_frac = seq_over_delta.float().mean()

    masked_rev_log_ratio_for_max = rev_log_ratio.masked_fill(~eos_bool_mask, -1e9)
    token_rev_kl_max = masked_rev_log_ratio_for_max.max()

    metrics = {
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
        "actor/ce_gppo_l_low": ce_loss_low.detach().item(),
        "actor/ce_gppo_l_high": ce_loss_high.detach().item(),
        "actor/adv_mask_frac": adv_mask_frac.detach().item(),
        "actor/adv_mask_frac_on_neg_adv": adv_mask_frac_on_neg.detach().item(),
        "actor/adv_masked_token_count": adv_masked_tokens.detach().item(),
        "actor/neg_adv_token_count": neg_adv_tokens.detach().item(),
        "actor/seq_over_delta_frac": seq_over_delta_frac.detach().item(),
        "actor/seq_rev_log_ratio_mean": seq_rev_log_ratio_mean.mean().detach().item(),
        "actor/token_rev_kl_max": token_rev_kl_max.detach().item(),
    }

    # Prepare mask matrices if requested
    mask_dict = None
    if return_masks:
        mask_dict = {
            "clipped_low_mask": clipped_low_mask.detach().cpu(),
            "clipped_high_mask": clipped_high_mask.detach().cpu(),
            "adv_mask_delta_mask": zero_mask.detach().cpu(),
        }

    return pg_loss, metrics, mask_dict


def compute_policy_loss_fix_advantage(
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
    advantages = torch.where(advantages > 0, advantages * 0.5, advantages)
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


def compute_policy_loss_grpo_min_max(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict]:
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

    # Clamp the ratio for clipping
    ratio_clipped = torch.clamp(ratio, lower_bound, upper_bound)

    # ===== SARS: Sign-Aligned Ratio Suppression =====
    # For positive advantages: use min(ratio, ratio_clipped) → loss = -adv * min(...)
    # For negative advantages: use max(ratio, ratio_clipped) → loss = -adv * max(...)
    # We implement via torch.where on the final surrogate loss
    surrogate_unclipped = -advantages * ratio
    surrogate_clipped = -advantages * ratio_clipped

    # When adv >= 0: we want the SMALLER loss (more conservative update) → min(surrogate_unclipped, surrogate_clipped)
    # But note: since adv >= 0, -adv*ratio is <= 0, and min means "more negative" = stronger update?
    # Actually, standard PPO uses: loss = -min(ratio * A, clip(ratio) * A)
    # So let's reconstruct it correctly:

    # Correct SARS formulation:
    #   if A >= 0: loss = -min(ratio * A, clip(ratio) * A)
    #   if A <  0: loss = -max(ratio * A, clip(ratio) * A)
    #
    # Which is equivalent to:
    #   loss = -torch.where(A >= 0, torch.min(ratio * A, ratio_clipped * A), torch.max(ratio * A, ratio_clipped * A))

    ratio_times_adv = ratio * advantages
    ratio_clipped_times_adv = ratio_clipped * advantages

    sars_surrogate = torch.where(
        advantages >= 0,
        torch.min(ratio_times_adv, ratio_clipped_times_adv),
        torch.max(ratio_times_adv, ratio_clipped_times_adv),
    )
    clip_pg_losses = -sars_surrogate  # because policy gradient loss = -E[surrogate]

    # ===== rest remains unchanged =====
    clipped_low_mask = ratio < lower_bound
    clipped_high_mask = ratio > upper_bound
    pg_clipfrac = masked_mean((clipped_low_mask | clipped_high_mask).float(), eos_mask)
    pg_clipfrac_low = masked_mean(clipped_low_mask.float(), eos_mask)
    pg_clipfrac_high = masked_mean(clipped_high_mask.float(), eos_mask)
    avg_ratio = masked_mean(ratio, eos_mask)

    # Advantage monitoring
    adv_pos_mask = (advantages > 0) & eos_mask.bool()
    adv_neg_mask = (advantages < 0) & eos_mask.bool()
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()
    avg_ratio_pos = masked_mean(ratio, adv_pos_mask.float())
    avg_ratio_neg = masked_mean(ratio, adv_neg_mask.float())

    # Adv-specific clip stats
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

    # Equality monitoring
    eq_mask = (old_log_prob == log_prob) & eos_mask.bool()
    equal_count = eq_mask.sum()

    # Aggregate loss
    pg_loss = agg_loss(loss_mat=clip_pg_losses, loss_mask=eos_mask, loss_agg_mode=loss_agg_mode)

    # Adv-specific counts and losses
    pos_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_pos_mask).sum()
    neg_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_neg_mask).sum()
    adv_pos_loss = masked_mean(clip_pg_losses, adv_pos_mask.float())
    adv_neg_loss = masked_mean(clip_pg_losses, adv_neg_mask.float())

    # Kept counts
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


from typing import Dict, Tuple

import torch


def compute_policy_loss_grpo_mask(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
    loss_agg_mode: str = "token-mean",
    adv_mask_delta: float = 0.05,
) -> Tuple[torch.Tensor, Dict]:
    """
    Custom GRPO-style loss with sequence-level mask M_{i,t} that removes
    negative-advantage tokens when the average reverse KL surpasses delta.
    """

    # ===== basic PPO quantities =====
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    rev_log_ratio = -negative_approx_kl

    # ===== clip bounds =====
    if low_clip_range > 0 and high_clip_range > 0:
        lower_bound = 1.0 - low_clip_range
        upper_bound = 1.0 + high_clip_range
    else:
        lower_bound = 1.0 - cliprange
        upper_bound = 1.0 + cliprange

    # ===== sequence-level reverse KL (only eos_mask tokens) =====
    eos_bool_mask = eos_mask.bool()

    seq_rev_log_ratio_sum = torch.sum(rev_log_ratio * eos_mask, dim=-1)
    seq_lengths = torch.sum(eos_mask, dim=-1)
    seq_lengths = torch.clamp(seq_lengths, min=1).to(log_prob.dtype)

    seq_rev_log_ratio_mean = seq_rev_log_ratio_sum / seq_lengths
    # Removed: seq_rev_log_ratio_mean_max
    seq_over_delta = seq_rev_log_ratio_mean > adv_mask_delta
    seq_over_delta_mask = seq_over_delta.unsqueeze(-1)

    # ===== build adv-based token mask =====
    neg_adv_mask = (advantages < 0) & eos_bool_mask

    # token is masked iff:
    #   - negative advantage
    #   - sequence reverse KL > delta
    #   - originally valid token (eos_mask)
    zero_mask = neg_adv_mask & seq_over_delta_mask

    mask_multiplier = (~zero_mask).to(log_prob.dtype)
    effective_mask = eos_mask * mask_multiplier
    effective_bool_mask = effective_mask.bool()

    # ===== PPO clipped objective (masked tokens contribute zero) =====
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, lower_bound, upper_bound)
    clip_pg_losses = torch.max(pg_losses, pg_losses2)

    ppo_kl = masked_mean(-negative_approx_kl, effective_mask)

    # ===== clip statistics =====
    clipped_low_mask = ratio < lower_bound
    clipped_high_mask = ratio > upper_bound
    clip_any = clipped_low_mask | clipped_high_mask

    pg_clipfrac = masked_mean(clip_any.float(), effective_mask)
    pg_clipfrac_low = masked_mean(clipped_low_mask.float(), effective_mask)
    pg_clipfrac_high = masked_mean(clipped_high_mask.float(), effective_mask)
    avg_ratio = masked_mean(ratio, effective_mask)

    # ===== advantage monitoring =====
    adv_pos_mask = (advantages > 0) & effective_bool_mask
    adv_neg_mask = (advantages < 0) & effective_bool_mask

    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()

    adv_pos_mask_f = adv_pos_mask.float()
    adv_neg_mask_f = adv_neg_mask.float()

    avg_ratio_pos = masked_mean(ratio, adv_pos_mask_f)
    avg_ratio_neg = masked_mean(ratio, adv_neg_mask_f)

    # ===== adv-specific clip statistics =====
    adv_pos_clipfrac = masked_mean(clip_any.float(), adv_pos_mask_f)
    adv_pos_clipfrac_low = masked_mean(clipped_low_mask.float(), adv_pos_mask_f)
    adv_pos_clipfrac_high = masked_mean(clipped_high_mask.float(), adv_pos_mask_f)

    adv_neg_clipfrac = masked_mean(clip_any.float(), adv_neg_mask_f)
    adv_neg_clipfrac_low = masked_mean(clipped_low_mask.float(), adv_neg_mask_f)
    adv_neg_clipfrac_high = masked_mean(clipped_high_mask.float(), adv_neg_mask_f)

    # ===== equality monitoring =====
    eq_mask = (old_log_prob == log_prob) & effective_bool_mask
    equal_count = eq_mask.sum()

    # ===== aggregate policy loss =====
    pg_loss = agg_loss(
        loss_mat=clip_pg_losses,
        loss_mask=effective_mask,
        loss_agg_mode=loss_agg_mode,
    )

    # ===== adv-specific counts and losses =====
    pos_clip_count = (clip_any & adv_pos_mask).sum()
    neg_clip_count = (clip_any & adv_neg_mask).sum()

    adv_pos_loss = masked_mean(clip_pg_losses, adv_pos_mask_f)
    adv_neg_loss = masked_mean(clip_pg_losses, adv_neg_mask_f)

    # ===== adv kept counts =====
    kept_mask = (~clip_any) & effective_bool_mask
    adv_pos_kept_count = (kept_mask & adv_pos_mask).sum()
    adv_neg_kept_count = (kept_mask & adv_neg_mask).sum()

    # ======================================================================
    # ======================= adv mask statistics ===========================
    # ======================================================================

    # 1) token-level: 被 adv_mask_delta 新增 mask 的 token / 原始 eos_mask token
    adv_masked_tokens = zero_mask.sum().to(log_prob.dtype)
    total_tokens = eos_bool_mask.sum().clamp(min=1).to(log_prob.dtype)
    adv_mask_frac = adv_masked_tokens / total_tokens

    # 2) neg-adv-only: 在负 advantage token 中被 mask 的比例
    neg_adv_tokens = neg_adv_mask.sum().clamp(min=1).to(log_prob.dtype)
    adv_mask_frac_on_neg = adv_masked_tokens / neg_adv_tokens

    # 3) sequence-level: 触发 adv_mask_delta 的序列比例
    seq_over_delta_frac = seq_over_delta.float().mean()
    seq_over_delta_count = seq_over_delta.sum().to(log_prob.dtype)
    seq_count = torch.tensor(seq_over_delta.numel(), dtype=log_prob.dtype, device=log_prob.device)

    # ===== token-level max reverse KL (new diagnostic metric) =====
    masked_rev_log_ratio_for_max = rev_log_ratio.masked_fill(~eos_bool_mask, -1e9)
    token_rev_kl_max = masked_rev_log_ratio_for_max.max()

    return pg_loss, {
        # ===== core PPO =====
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_low": pg_clipfrac_low.detach().item(),
        "actor/pg_clipfrac_high": pg_clipfrac_high.detach().item(),
        "actor/avg_ratio": avg_ratio.detach().item(),
        # ===== advantage stats =====
        "actor/adv_pos_count": adv_pos_count.detach().item(),
        "actor/adv_neg_count": adv_neg_count.detach().item(),
        "actor/avg_ratio_pos": avg_ratio_pos.detach().item(),
        "actor/avg_ratio_neg": avg_ratio_neg.detach().item(),
        # ===== equality =====
        "actor/old_logprob_and_logprob_equal_count": equal_count.detach().item(),
        # ===== adv-specific clip stats =====
        "actor/adv_pos_clipfrac": adv_pos_clipfrac.detach().item(),
        "actor/adv_pos_clipfrac_low": adv_pos_clipfrac_low.detach().item(),
        "actor/adv_pos_clipfrac_high": adv_pos_clipfrac_high.detach().item(),
        "actor/adv_neg_clipfrac": adv_neg_clipfrac.detach().item(),
        "actor/adv_neg_clipfrac_low": adv_neg_clipfrac_low.detach().item(),
        "actor/adv_neg_clipfrac_high": adv_neg_clipfrac_high.detach().item(),
        # ===== counts & losses =====
        "actor/pos_clip_count": pos_clip_count.detach().item(),
        "actor/neg_clip_count": neg_clip_count.detach().item(),
        "actor/adv_pos_loss": adv_pos_loss.detach().item(),
        "actor/adv_neg_loss": adv_neg_loss.detach().item(),
        "actor/adv_pos_kept_count": adv_pos_kept_count.detach().item(),
        "actor/adv_neg_kept_count": adv_neg_kept_count.detach().item(),
        # ===== adv mask (NEW) =====
        "actor/adv_mask_frac": adv_mask_frac.detach().item(),
        "actor/adv_mask_frac_on_neg_adv": adv_mask_frac_on_neg.detach().item(),
        "actor/adv_masked_token_count": adv_masked_tokens.detach().item(),
        "actor/neg_adv_token_count": neg_adv_tokens.detach().item(),
        "actor/seq_over_delta_frac": seq_over_delta_frac.detach().item(),
        "actor/seq_over_delta_count": seq_over_delta_count.detach().item(),
        "actor/seq_count": seq_count.detach().item(),
        "actor/seq_rev_log_ratio_mean": seq_rev_log_ratio_mean.mean().detach().item(),
        "actor/token_rev_kl_max": token_rev_kl_max.detach().item(),  # <-- 新增且更有意义的极值指标
    }


def compute_policy_loss_bapo_mask(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    init_lower_bound: float,
    init_upper_bound: float,
    max_lower_bound: float,
    max_upper_bound: float,
    lower_step: float,
    upper_step: float,
    positive_contribution_threshold: float,
    loss_agg_mode: str = "token-mean",
    adv_mask_delta: float = 0.05,
) -> Tuple[torch.Tensor, Dict]:
    """BAPO loss combined with GRPO-style masking.

    Combines:
    1. BAPO: Dynamic clipping bounds adjustment until positive contribution threshold is met
    2. GRPO Mask: Sequence-level masking based on reverse KL divergence

    Args:
        old_log_prob: `(bs, T)` old policy log probs.
        log_prob: `(bs, T)` current policy log probs.
        advantages: `(bs, T)` token advantages.
        eos_mask: `(bs, T)` mask for valid tokens.
        init_lower_bound: initial lower clip bound ``c_low`` (`a⁻`).
        init_upper_bound: initial upper clip bound ``c_high`` (`a⁺`).
        max_lower_bound: max reachable lower bound (`b⁻`).
        max_upper_bound: max reachable upper bound (`b⁺`).
        lower_step: increment for the lower bound (``δ₂`` > 0).
        upper_step: increment for the upper bound (``δ₁`` > 0).
        positive_contribution_threshold: target ratio ``ρ₀`` in [0, 1].
        loss_agg_mode: aggregation strategy for the loss.
        adv_mask_delta: threshold for sequence-level reverse KL masking.

    Returns:
        Tuple of `(loss, metrics)`.
    """
    # Validation (same as BAPO)
    if lower_step <= 0.0:
        raise ValueError("lower_step must be > 0 for BAPO")
    if upper_step <= 0.0:
        raise ValueError("upper_step must be > 0 for BAPO")
    if init_lower_bound > max_lower_bound:
        raise ValueError("init_lower_bound must be <= max_lower_bound")
    if init_upper_bound > max_upper_bound:
        raise ValueError("init_upper_bound must be <= max_upper_bound")
    if not (0.0 <= positive_contribution_threshold <= 1.0):
        raise ValueError("positive_contribution_threshold must be in [0, 1]")

    # ===== GRPO Mask Logic (applied first) =====
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    rev_log_ratio = -negative_approx_kl

    eos_bool_mask = eos_mask.bool()

    # Sequence-level reverse KL
    seq_rev_log_ratio_sum = torch.sum(rev_log_ratio * eos_mask, dim=-1)
    seq_lengths = torch.sum(eos_mask, dim=-1).clamp(min=1).to(log_prob.dtype)
    seq_rev_log_ratio_mean = seq_rev_log_ratio_sum / seq_lengths
    seq_over_delta = seq_rev_log_ratio_mean > adv_mask_delta
    seq_over_delta_mask = seq_over_delta.unsqueeze(-1)

    # Build mask: zero out negative-advantage tokens when seq_rev_kl > delta
    neg_adv_mask = (advantages < 0) & eos_bool_mask
    zero_mask = neg_adv_mask & seq_over_delta_mask

    mask_multiplier = (~zero_mask).to(log_prob.dtype)
    effective_mask = eos_mask * mask_multiplier
    effective_bool_mask = effective_mask.bool()

    # ===== BAPO Logic (applied on effective_mask) =====
    mask_float = effective_mask.to(log_prob.dtype)
    ppo_kl = masked_mean(-negative_approx_kl, effective_mask)

    ratio_times_adv = ratio * advantages
    eps = 1e-8

    c_low = float(init_lower_bound)
    c_high = float(init_upper_bound)

    def _positive_ratio_value(low_bound: float, high_bound: float) -> float:
        clipped_ratio = torch.clamp(ratio, min=low_bound, max=high_bound)
        surrogate = torch.min(ratio_times_adv, clipped_ratio * advantages)
        masked_surrogate = surrogate * mask_float
        masked_surrogate_fp32 = masked_surrogate.to(torch.float32)
        total = masked_surrogate_fp32.abs().sum()
        total_value = float(total.detach().cpu())
        if total_value <= eps:
            return 1.0
        positive = torch.clamp(masked_surrogate_fp32, min=0.0).sum()
        ratio_value = positive / (total + eps)
        return float(ratio_value.detach().cpu())

    pos_ratio_value = _positive_ratio_value(c_low, c_high)
    adjust_steps = 0

    while (
        pos_ratio_value < positive_contribution_threshold
        and c_low + lower_step <= max_lower_bound + 1e-12
    ):
        if c_high + upper_step <= max_upper_bound + 1e-12:
            c_high = min(c_high + upper_step, max_upper_bound)
        elif c_low + lower_step <= max_lower_bound + 1e-12:
            c_low = min(c_low + lower_step, max_lower_bound)
        else:
            break
        adjust_steps += 1
        pos_ratio_value = _positive_ratio_value(c_low, c_high)

    clipped_ratio = torch.clamp(ratio, min=c_low, max=c_high)
    clipped_ratio_times_adv = clipped_ratio * advantages
    surrogate = torch.min(ratio_times_adv, clipped_ratio_times_adv)
    clip_pg_losses = -surrogate

    # Aggregate loss using effective_mask
    pg_loss = agg_loss(
        loss_mat=clip_pg_losses,
        loss_mask=effective_mask,
        loss_agg_mode=loss_agg_mode,
    )

    # ===== Metrics (similar to both BAPO and GRPO mask) =====
    clipped_low_mask = ratio < c_low
    clipped_high_mask = ratio > c_high
    pg_clipfrac = masked_mean((clipped_low_mask | clipped_high_mask).float(), effective_mask)
    pg_clipfrac_low = masked_mean(clipped_low_mask.float(), effective_mask)
    pg_clipfrac_high = masked_mean(clipped_high_mask.float(), effective_mask)
    avg_ratio = masked_mean(ratio, effective_mask)

    adv_pos_mask = (advantages > 0) & effective_bool_mask
    adv_neg_mask = (advantages < 0) & effective_bool_mask
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()
    avg_ratio_pos = masked_mean(ratio, adv_pos_mask.float())
    avg_ratio_neg = masked_mean(ratio, adv_neg_mask.float())

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

    eq_mask = (old_log_prob == log_prob) & effective_bool_mask
    equal_count = eq_mask.sum()

    pos_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_pos_mask).sum()
    neg_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_neg_mask).sum()
    adv_pos_loss = masked_mean(clip_pg_losses, adv_pos_mask.float())
    adv_neg_loss = masked_mean(clip_pg_losses, adv_neg_mask.float())

    kept_mask = (~clipped_low_mask & ~clipped_high_mask) & effective_bool_mask
    adv_pos_kept_count = (kept_mask & adv_pos_mask).sum()
    adv_neg_kept_count = (kept_mask & adv_neg_mask).sum()

    # GRPO mask metrics
    adv_masked_tokens = zero_mask.sum().to(log_prob.dtype)
    total_tokens = eos_bool_mask.sum().clamp(min=1).to(log_prob.dtype)
    adv_mask_frac = adv_masked_tokens / total_tokens
    neg_adv_tokens = neg_adv_mask.sum().clamp(min=1).to(log_prob.dtype)
    adv_mask_frac_on_neg = adv_masked_tokens / neg_adv_tokens
    seq_over_delta_frac = seq_over_delta.float().mean()
    masked_rev_log_ratio_for_max = rev_log_ratio.masked_fill(~eos_bool_mask, -1e9)
    token_rev_kl_max = masked_rev_log_ratio_for_max.max()

    metrics = {
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
        # BAPO metrics
        "actor/bapo_positive_contribution": pos_ratio_value,
        "actor/bapo_c_low": c_low,
        "actor/bapo_c_high": c_high,
        "actor/bapo_adjust_steps": float(adjust_steps),
        "actor/bapo_pos_threshold": positive_contribution_threshold,
        # GRPO mask metrics
        "actor/adv_mask_frac": adv_mask_frac.detach().item(),
        "actor/adv_mask_frac_on_neg_adv": adv_mask_frac_on_neg.detach().item(),
        "actor/adv_masked_token_count": adv_masked_tokens.detach().item(),
        "actor/neg_adv_token_count": neg_adv_tokens.detach().item(),
        "actor/seq_over_delta_frac": seq_over_delta_frac.detach().item(),
        "actor/seq_rev_log_ratio_mean": seq_rev_log_ratio_mean.mean().detach().item(),
        "actor/token_rev_kl_max": token_rev_kl_max.detach().item(),
    }

    return pg_loss, metrics


def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
) -> Tuple[torch.Tensor, Dict]:
    """
    GSPO (Group-based Stepwise Policy Optimization) loss function.

    Key difference from DAPO/PPO:
    - Uses sequence-level ratio: s_i = exp(mean(log_ratio_t)) = (π_θ(y|x) / π_old(y|x))^(1/|y|)
    - Computes loss per sequence, then averages over sequences (1/G)

    Formula:
        J_GSPO = (1/G) * Σ_i min(s_i * A_i, clip(s_i, 1-ε, 1+ε) * A_i)

        where:
        - s_i = exp((1/|y_i|) * Σ_t log(π_θ(y_t|x,y_<t) / π_old(y_t|x,y_<t)))
        - A_i = (r_i - mean(r)) / std(r)  (group-normalized advantage, computed externally)

    Args:
        old_log_prob: (bs, response_length) - log probabilities from old policy
        log_prob: (bs, response_length) - log probabilities from current policy
        advantages: (bs, response_length) - advantages (same value across tokens for each sequence)
        eos_mask: (bs, response_length) - mask for valid tokens
        cliprange: float - symmetric clip range (used if low/high not specified)
        low_clip_range: float - lower clip bound (1 - low_clip_range)
        high_clip_range: float - upper clip bound (1 + high_clip_range)
        loss_agg_mode: str - ignored for GSPO (always uses sequence-mean)

    Returns:
        pg_loss: scalar tensor - the policy gradient loss
        metrics: dict - training statistics
    """
    # ===== Step 1: Compute token-level log ratio =====
    # log(π_θ / π_old) = log_prob - old_log_prob
    token_log_ratio = log_prob - old_log_prob  # (bs, response_length)

    # ===== Step 2: Compute sequence-level ratio s_i =====
    # s_i = exp((1/|y_i|) * Σ_t log_ratio_t)
    # This is equivalent to: s_i = (π_θ(y|x) / π_old(y|x))^(1/|y|)

    # Sum of log ratios per sequence (masked)
    seq_log_ratio_sum = torch.sum(token_log_ratio * eos_mask, dim=-1)  # (bs,)
    # Length of each sequence
    seq_lengths = torch.sum(eos_mask, dim=-1).clamp(min=1.0)  # (bs,)
    # Mean log ratio per sequence
    seq_mean_log_ratio = seq_log_ratio_sum / seq_lengths  # (bs,)
    # Sequence-level ratio: s_i = exp(mean_log_ratio)
    seq_ratio = torch.exp(seq_mean_log_ratio)  # (bs,)

    # ===== Step 3: Get sequence-level advantage =====
    # Advantage is assumed to be the same across all tokens in a sequence
    # Take the advantage from first valid token position
    # Note: In GRPO, advantages are already computed as sequence-level and broadcast to all tokens
    seq_advantages = advantages[:, 0]  # (bs,) - take first token's advantage as sequence advantage

    # ===== Step 4: Compute clipped surrogate objective =====
    # Determine clip bounds
    if low_clip_range > 0 and high_clip_range > 0:
        lower_bound = 1.0 - low_clip_range
        upper_bound = 1.0 + high_clip_range
    else:
        lower_bound = 1.0 - cliprange
        upper_bound = 1.0 + cliprange

    # Clipped ratio
    clipped_seq_ratio = torch.clamp(seq_ratio, lower_bound, upper_bound)  # (bs,)

    # GSPO objective (to maximize): min(s_i * A_i, clip(s_i) * A_i)
    # Loss (to minimize): -min(s_i * A_i, clip(s_i) * A_i) = max(-s_i * A_i, -clip(s_i) * A_i)
    pg_losses1 = -seq_advantages * seq_ratio  # (bs,)
    pg_losses2 = -seq_advantages * clipped_seq_ratio  # (bs,)
    seq_pg_losses = torch.max(pg_losses1, pg_losses2)  # (bs,)

    # ===== Step 5: Aggregate loss =====
    # GSPO uses (1/G) * Σ_i loss_i, i.e., mean over sequences
    pg_loss = torch.mean(seq_pg_losses)

    # ===== Compute metrics =====
    # Clip statistics (sequence-level)
    clipped_low_mask = seq_ratio < lower_bound  # (bs,)
    clipped_high_mask = seq_ratio > upper_bound  # (bs,)
    pg_clipfrac = (clipped_low_mask | clipped_high_mask).float().mean()
    pg_clipfrac_low = clipped_low_mask.float().mean()
    pg_clipfrac_high = clipped_high_mask.float().mean()
    avg_seq_ratio = seq_ratio.mean()

    # KL divergence (token-level for reference)
    ppo_kl = masked_mean(-token_log_ratio, eos_mask)

    # Advantage monitoring (sequence-level)
    adv_pos_mask = seq_advantages > 0  # (bs,)
    adv_neg_mask = seq_advantages < 0  # (bs,)
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()

    # Average ratio by advantage sign
    avg_ratio_pos = (
        seq_ratio[adv_pos_mask].mean()
        if adv_pos_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    avg_ratio_neg = (
        seq_ratio[adv_neg_mask].mean()
        if adv_neg_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )

    # Advantage-specific clip fractions
    adv_pos_clipfrac = (
        (clipped_low_mask | clipped_high_mask)[adv_pos_mask].float().mean()
        if adv_pos_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_pos_clipfrac_low = (
        clipped_low_mask[adv_pos_mask].float().mean()
        if adv_pos_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_pos_clipfrac_high = (
        clipped_high_mask[adv_pos_mask].float().mean()
        if adv_pos_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_neg_clipfrac = (
        (clipped_low_mask | clipped_high_mask)[adv_neg_mask].float().mean()
        if adv_neg_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_neg_clipfrac_low = (
        clipped_low_mask[adv_neg_mask].float().mean()
        if adv_neg_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_neg_clipfrac_high = (
        clipped_high_mask[adv_neg_mask].float().mean()
        if adv_neg_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )

    # Clip counts
    pos_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_pos_mask).sum()
    neg_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_neg_mask).sum()

    # Advantage-specific losses
    adv_pos_loss = (
        seq_pg_losses[adv_pos_mask].mean()
        if adv_pos_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_neg_loss = (
        seq_pg_losses[adv_neg_mask].mean()
        if adv_neg_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )

    # Kept counts (not clipped)
    kept_mask = ~clipped_low_mask & ~clipped_high_mask
    adv_pos_kept_count = (kept_mask & adv_pos_mask).sum()
    adv_neg_kept_count = (kept_mask & adv_neg_mask).sum()

    # Equality monitoring
    eq_mask = (old_log_prob == log_prob) & eos_mask.bool()
    equal_count = eq_mask.sum()

    # GSPO-specific metrics
    avg_seq_length = seq_lengths.mean()
    std_seq_ratio = (
        seq_ratio.std() if seq_ratio.numel() > 1 else torch.tensor(0.0, device=seq_ratio.device)
    )

    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_low": pg_clipfrac_low.detach().item(),
        "actor/pg_clipfrac_high": pg_clipfrac_high.detach().item(),
        "actor/avg_ratio": avg_seq_ratio.detach().item(),
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
        # GSPO-specific metrics
        "actor/gspo_avg_seq_length": avg_seq_length.detach().item(),
        "actor/gspo_std_seq_ratio": std_seq_ratio.detach().item(),
    }


def compute_policy_loss_mgspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
) -> Tuple[torch.Tensor, Dict]:
    """
    MGSPO (Modified Group-based Stepwise Policy Optimization) loss function.

    Key difference from GSPO:
    - Uses modified sequence-level ratio: min(s_i, 1/s_i) instead of s_i
    - This ensures the weight is always <= 1, providing more conservative updates

    Formula:
        J_MGSPO = (1/G) * Σ_i min(s'_i * A_i, clip(s'_i, 1-ε, 1+ε) * A_i)

        where:
        - s_i = exp((1/|y_i|) * Σ_t log(π_θ(y_t|x,y_<t) / π_old(y_t|x,y_<t)))
        - s'_i = min(s_i, 1/s_i)  # Modified weight
        - A_i = (r_i - mean(r)) / std(r)  (group-normalized advantage, computed externally)

    Args:
        old_log_prob: (bs, response_length) - log probabilities from old policy
        log_prob: (bs, response_length) - log probabilities from current policy
        advantages: (bs, response_length) - advantages (same value across tokens for each sequence)
        eos_mask: (bs, response_length) - mask for valid tokens
        cliprange: float - symmetric clip range (used if low/high not specified)
        low_clip_range: float - lower clip bound (1 - low_clip_range)
        high_clip_range: float - upper clip bound (1 + high_clip_range)

    Returns:
        pg_loss: scalar tensor - the policy gradient loss
        metrics: dict - training statistics
    """
    # ===== Step 1: Compute token-level log ratio =====
    # log(π_θ / π_old) = log_prob - old_log_prob
    token_log_ratio = log_prob - old_log_prob  # (bs, response_length)

    # ===== Step 2: Compute sequence-level ratio s_i =====
    # s_i = exp((1/|y_i|) * Σ_t log_ratio_t)
    # This is equivalent to: s_i = (π_θ(y|x) / π_old(y|x))^(1/|y|)

    # Sum of log ratios per sequence (masked)
    seq_log_ratio_sum = torch.sum(token_log_ratio * eos_mask, dim=-1)  # (bs,)
    # Length of each sequence
    seq_lengths = torch.sum(eos_mask, dim=-1).clamp(min=1.0)  # (bs,)
    # Mean log ratio per sequence
    seq_mean_log_ratio = seq_log_ratio_sum / seq_lengths  # (bs,)
    # Sequence-level ratio: s_i = exp(mean_log_ratio)
    seq_ratio = torch.exp(seq_mean_log_ratio)  # (bs,)

    # ===== Step 3: Compute modified ratio s'_i = min(s_i, 1/s_i) =====
    # This ensures the weight is always <= 1
    seq_ratio_inv = 1.0 / (seq_ratio + 1e-8)  # (bs,) - add eps for numerical stability
    modified_seq_ratio = torch.minimum(seq_ratio, seq_ratio_inv)  # (bs,)

    # ===== Step 4: Get sequence-level advantage =====
    # Advantage is assumed to be the same across all tokens in a sequence
    # Take the advantage from first valid token position
    # Note: In GRPO, advantages are already computed as sequence-level and broadcast to all tokens
    seq_advantages = advantages[:, 0]  # (bs,) - take first token's advantage as sequence advantage

    # ===== Step 5: Compute clipped surrogate objective =====
    # Determine clip bounds
    if low_clip_range > 0 and high_clip_range > 0:
        lower_bound = 1.0 - low_clip_range
        upper_bound = 1.0 + high_clip_range
    else:
        lower_bound = 1.0 - cliprange
        upper_bound = 1.0 + cliprange

    # Clipped modified ratio
    clipped_modified_seq_ratio = torch.clamp(modified_seq_ratio, lower_bound, upper_bound)  # (bs,)

    # MGSPO objective (to maximize): min(s'_i * A_i, clip(s'_i) * A_i)
    # Loss (to minimize): -min(s'_i * A_i, clip(s'_i) * A_i) = max(-s'_i * A_i, -clip(s'_i) * A_i)
    pg_losses1 = -seq_advantages * modified_seq_ratio  # (bs,)
    pg_losses2 = -seq_advantages * clipped_modified_seq_ratio  # (bs,)
    seq_pg_losses = torch.max(pg_losses1, pg_losses2)  # (bs,)

    # ===== Step 6: Aggregate loss =====
    # MGSPO uses (1/G) * Σ_i loss_i, i.e., mean over sequences
    pg_loss = torch.mean(seq_pg_losses)

    # ===== Compute metrics =====
    # Clip statistics (sequence-level, using modified ratio)
    clipped_low_mask = modified_seq_ratio < lower_bound  # (bs,)
    clipped_high_mask = modified_seq_ratio > upper_bound  # (bs,)
    pg_clipfrac = (clipped_low_mask | clipped_high_mask).float().mean()
    pg_clipfrac_low = clipped_low_mask.float().mean()
    pg_clipfrac_high = clipped_high_mask.float().mean()
    avg_modified_seq_ratio = modified_seq_ratio.mean()
    avg_seq_ratio = seq_ratio.mean()

    # KL divergence (token-level for reference)
    ppo_kl = masked_mean(-token_log_ratio, eos_mask)

    # Advantage monitoring (sequence-level)
    adv_pos_mask = seq_advantages > 0  # (bs,)
    adv_neg_mask = seq_advantages < 0  # (bs,)
    adv_pos_count = adv_pos_mask.sum()
    adv_neg_count = adv_neg_mask.sum()

    # Average ratio by advantage sign (using modified ratio)
    avg_ratio_pos = (
        modified_seq_ratio[adv_pos_mask].mean()
        if adv_pos_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    avg_ratio_neg = (
        modified_seq_ratio[adv_neg_mask].mean()
        if adv_neg_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )

    # Advantage-specific clip fractions
    adv_pos_clipfrac = (
        (clipped_low_mask | clipped_high_mask)[adv_pos_mask].float().mean()
        if adv_pos_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_pos_clipfrac_low = (
        clipped_low_mask[adv_pos_mask].float().mean()
        if adv_pos_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_pos_clipfrac_high = (
        clipped_high_mask[adv_pos_mask].float().mean()
        if adv_pos_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_neg_clipfrac = (
        (clipped_low_mask | clipped_high_mask)[adv_neg_mask].float().mean()
        if adv_neg_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_neg_clipfrac_low = (
        clipped_low_mask[adv_neg_mask].float().mean()
        if adv_neg_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_neg_clipfrac_high = (
        clipped_high_mask[adv_neg_mask].float().mean()
        if adv_neg_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )

    # Clip counts
    pos_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_pos_mask).sum()
    neg_clip_count = ((clipped_low_mask | clipped_high_mask) & adv_neg_mask).sum()

    # Advantage-specific losses
    adv_pos_loss = (
        seq_pg_losses[adv_pos_mask].mean()
        if adv_pos_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )
    adv_neg_loss = (
        seq_pg_losses[adv_neg_mask].mean()
        if adv_neg_mask.any()
        else torch.tensor(0.0, device=seq_ratio.device)
    )

    # Kept counts (not clipped)
    kept_mask = ~clipped_low_mask & ~clipped_high_mask
    adv_pos_kept_count = (kept_mask & adv_pos_mask).sum()
    adv_neg_kept_count = (kept_mask & adv_neg_mask).sum()

    # Equality monitoring
    eq_mask = (old_log_prob == log_prob) & eos_mask.bool()
    equal_count = eq_mask.sum()

    # MGSPO-specific metrics
    avg_seq_length = seq_lengths.mean()
    std_seq_ratio = (
        seq_ratio.std() if seq_ratio.numel() > 1 else torch.tensor(0.0, device=seq_ratio.device)
    )
    std_modified_seq_ratio = (
        modified_seq_ratio.std()
        if modified_seq_ratio.numel() > 1
        else torch.tensor(0.0, device=modified_seq_ratio.device)
    )

    # Count how many sequences use s_i vs 1/s_i
    use_original_ratio_count = (seq_ratio <= seq_ratio_inv).sum()
    use_inverse_ratio_count = (seq_ratio > seq_ratio_inv).sum()

    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_low": pg_clipfrac_low.detach().item(),
        "actor/pg_clipfrac_high": pg_clipfrac_high.detach().item(),
        "actor/avg_ratio": avg_modified_seq_ratio.detach().item(),
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
        # MGSPO-specific metrics
        "actor/mgspo_avg_seq_length": avg_seq_length.detach().item(),
        "actor/mgspo_std_seq_ratio": std_seq_ratio.detach().item(),
        "actor/mgspo_avg_original_seq_ratio": avg_seq_ratio.detach().item(),
        "actor/mgspo_std_modified_seq_ratio": std_modified_seq_ratio.detach().item(),
        "actor/mgspo_use_original_ratio_count": use_original_ratio_count.detach().item(),
        "actor/mgspo_use_inverse_ratio_count": use_inverse_ratio_count.detach().item(),
    }


def compute_policy_loss_eb_grpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    low_clip_range: float,
    high_clip_range: float,
    alpha: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict]:
    r"""
    EB-GRPO (alpha-weighted token objective with per-sequence normalization).

    Objective (to maximize):
        J_EB-GRPO = E[ (1/G) Σ_i (1/E_i) Σ_t min(w_{i,t} * Â_i * α_{i,t},
                                                 clip(w_{i,t}) * Â_i * α_{i,t}) ]

    This function returns the *loss to minimize*:
        L = -J_EB-GRPO

    Notes:
    - w_{i,t} is the PPO ratio exp(log_prob - old_log_prob).
    - Â_i is treated as sequence-level advantage (computed as masked mean over tokens).
    - α_{i,t} is provided via `alpha` (shape (bs, T) or (bs,)); if None, defaults to 1 on valid tokens.
    - E_i is computed as Σ_t α_{i,t} over valid tokens (eos_mask).
    """
    token_log_ratio = log_prob - old_log_prob
    ratio = torch.exp(token_log_ratio)

    if low_clip_range > 0 and high_clip_range > 0:
        lower_bound = 1.0 - low_clip_range
        upper_bound = 1.0 + high_clip_range
    else:
        lower_bound = 1.0 - cliprange
        upper_bound = 1.0 + cliprange

    ratio_clipped = torch.clamp(ratio, lower_bound, upper_bound)

    eos_mask_f = eos_mask.to(dtype=log_prob.dtype)
    eos_len = torch.sum(eos_mask_f, dim=-1).clamp(min=1.0)
    seq_adv = torch.sum(advantages * eos_mask_f, dim=-1) / (eos_len + eps)  # (bs,)

    if alpha is None:
        alpha_t = torch.ones_like(ratio)
    else:
        alpha_t = alpha.to(device=ratio.device, dtype=ratio.dtype)
        if alpha_t.ndim == 1:
            alpha_t = alpha_t.unsqueeze(-1).expand_as(ratio)
        if alpha_t.shape != ratio.shape:
            raise ValueError(
                f"alpha must have shape {tuple(ratio.shape)} (or (bs,) broadcastable), got {tuple(alpha_t.shape)}"
            )

    # Effective token weights and per-sequence normalizer E_i
    eff_alpha = alpha_t * eos_mask_f
    E = torch.sum(eff_alpha, dim=-1).clamp(min=eps)  # (bs,)

    # PPO-style clipped surrogate with α_{i,t} included
    seq_adv_t = seq_adv.unsqueeze(-1)  # (bs, 1)
    pg_losses1 = -(seq_adv_t * alpha_t) * ratio
    pg_losses2 = -(seq_adv_t * alpha_t) * ratio_clipped
    per_token_loss = torch.max(pg_losses1, pg_losses2)  # = -min(surrogate1, surrogate2)

    seq_loss = torch.sum(per_token_loss * eos_mask_f, dim=-1) / E
    pg_loss = torch.mean(seq_loss)

    # Diagnostics (token-level, eos-masked unless otherwise noted)
    ppo_kl = masked_mean(-token_log_ratio, eos_mask_f)
    clipped_low_mask = ratio < lower_bound
    clipped_high_mask = ratio > upper_bound
    clip_any = clipped_low_mask | clipped_high_mask

    pg_clipfrac = masked_mean(clip_any.float(), eos_mask_f)
    pg_clipfrac_low = masked_mean(clipped_low_mask.float(), eos_mask_f)
    pg_clipfrac_high = masked_mean(clipped_high_mask.float(), eos_mask_f)
    avg_ratio = masked_mean(ratio, eos_mask_f)

    # α-weighted clip fraction (matches E_i weighting)
    alpha_clipfrac = torch.sum(clip_any.to(ratio.dtype) * eff_alpha) / (torch.sum(eff_alpha) + eps)

    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_low": pg_clipfrac_low.detach().item(),
        "actor/pg_clipfrac_high": pg_clipfrac_high.detach().item(),
        "actor/avg_ratio": avg_ratio.detach().item(),
        "actor/eb_grpo_E_mean": E.mean().detach().item(),
        "actor/eb_grpo_alpha_mean": masked_mean(alpha_t, eos_mask_f).detach().item(),
        "actor/eb_grpo_alpha_clipfrac": alpha_clipfrac.detach().item(),
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


_BAPO_CFG_VALUE_MISSING = object()


def _maybe_get_bapo_cfg_value(cfg: Any, key: str):
    if isinstance(cfg, Mapping) and key in cfg:
        return cfg[key]
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]  # type: ignore[index]
    except Exception:
        return _BAPO_CFG_VALUE_MISSING


def _resolve_bapo_cfg_value(cfg: Any, *names: str):
    for name in names:
        value = _maybe_get_bapo_cfg_value(cfg, name)
        if value is not _BAPO_CFG_VALUE_MISSING and value is not None:
            return value
    raise KeyError(names[0])


def _parse_bapo_cfg(cfg: Any) -> Dict[str, float]:
    if cfg is None:
        raise ValueError("config.bapo must be provided when loss_func_type='bapo'.")
    try:
        params = {
            "init_lower_bound": float(
                _resolve_bapo_cfg_value(cfg, "init_lower_bound", "lower_bound_init", "a_minus")
            ),
            "init_upper_bound": float(
                _resolve_bapo_cfg_value(cfg, "init_upper_bound", "upper_bound_init", "a_plus")
            ),
            "max_lower_bound": float(
                _resolve_bapo_cfg_value(cfg, "max_lower_bound", "lower_bound_max", "b_minus")
            ),
            "max_upper_bound": float(
                _resolve_bapo_cfg_value(cfg, "max_upper_bound", "upper_bound_max", "b_plus")
            ),
            "lower_step": float(
                _resolve_bapo_cfg_value(cfg, "lower_step", "delta_lower", "delta2", "delta_low")
            ),
            "upper_step": float(
                _resolve_bapo_cfg_value(cfg, "upper_step", "delta_upper", "delta1", "delta_high")
            ),
            "positive_contribution_threshold": float(
                _resolve_bapo_cfg_value(
                    cfg,
                    "positive_contribution_threshold",
                    "positive_token_contribution_threshold",
                    "pos_contribution_threshold",
                    "rho0",
                    "rho",
                )
            ),
        }
    except KeyError as exc:
        raise ValueError(f"Missing BAPO config value: {exc.args[0]}") from exc

    return params


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
    adv_mask_delta: float = 0.05,
    lbpo_alpha: Optional[float] = None,
    ce_gppo_beta1: float = 1.0,
    ce_gppo_beta2: float = 1.0,
    bapo_cfg: Optional[Any] = None,
    icepop_cfg: Optional[Any] = None,
    scgrpo_cfg: Optional[Any] = None,
    sapo_cfg: Optional[Any] = None,
    return_masks: bool = False,
) -> Tuple[torch.Tensor, dict, Optional[Dict[str, torch.Tensor]]]:
    """Compute policy loss based on the specified loss function type.

    Args:
        return_masks: If True and loss_func_type is "ce_gppo_mask", returns mask matrices.

    Returns:
        pg_loss: Policy gradient loss
        loss_metrics: Dictionary of metrics
        mask_dict: Optional dictionary containing mask matrices (only for ce_gppo_mask)
    """
    advantages = data["advantages"]
    responses = data["responses"]
    attention_mask = data["attention_mask"]
    response_length = responses.size(1)
    response_mask = attention_mask[:, -response_length:]

    if loss_func_type != "lbpo" and loss_agg_mode == "lbpo":
        raise ValueError("Set loss_func_type='lbpo' when using loss_agg_mode='lbpo'.")
    if loss_func_type == "lbpo" and loss_agg_mode != "lbpo":
        raise ValueError("loss_func_type='lbpo' requires loss_agg_mode='lbpo'.")

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
    elif loss_func_type in ("SCGRPO", "scgrpo"):
        scgrpo_params = _parse_scgrpo_cfg(scgrpo_cfg)
        pg_loss, loss_metrics = compute_policy_loss_scgrpo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            alpha_out=scgrpo_params["alpha_out"],
            alpha_in=scgrpo_params["alpha_in"],
            loss_agg_mode=loss_agg_mode,
        )
    elif loss_func_type == "sapo":
        sapo_params = _parse_sapo_cfg(sapo_cfg)
        pg_loss, loss_metrics = compute_policy_loss_sapo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            tau_pos=sapo_params["tau_pos"],
            tau_neg=sapo_params["tau_neg"],
            loss_agg_mode=loss_agg_mode,
        )
    elif loss_func_type == "icepop":
        if not do_old_log_prob_compute:
            raise ValueError(
                "IcePop requires do_old_log_prob_compute=True (needs π_train(·; θ_old))."
            )
        icepop_params = _parse_icepop_cfg(icepop_cfg)
        infer_log_prob = data.get("sglang_log_probs", None)
        if infer_log_prob is None:
            raise ValueError("IcePop requires 'sglang_log_probs' in the training batch.")
        if infer_log_prob.shape != old_log_prob.shape:
            min_len = min(infer_log_prob.size(1), old_log_prob.size(1))
            infer_log_prob = infer_log_prob[:, :min_len]
            old_log_prob = old_log_prob[:, :min_len]
            log_prob = log_prob[:, :min_len]
            advantages = advantages[:, :min_len]
            response_mask = response_mask[:, :min_len]
        pg_loss, loss_metrics = compute_policy_loss_icepop(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            infer_log_prob=infer_log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            alpha=icepop_params["alpha"],
            beta=icepop_params["beta"],
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
            loss_agg_mode=loss_agg_mode,
        )
    elif loss_func_type == "sscr_icepop":
        if not do_old_log_prob_compute:
            raise ValueError(
                "S-SCR-IcePop requires do_old_log_prob_compute=True (needs π_train(·; θ_old))."
            )
        sscr_params = _parse_sscr_icepop_cfg(icepop_cfg)
        infer_log_prob = data.get("sglang_log_probs", None)
        if infer_log_prob is None:
            raise ValueError("S-SCR-IcePop requires 'sglang_log_probs' in the training batch.")
        if infer_log_prob.shape != old_log_prob.shape:
            min_len = min(infer_log_prob.size(1), old_log_prob.size(1))
            infer_log_prob = infer_log_prob[:, :min_len]
            old_log_prob = old_log_prob[:, :min_len]
            log_prob = log_prob[:, :min_len]
            advantages = advantages[:, :min_len]
            response_mask = response_mask[:, :min_len]
        pg_loss, loss_metrics = compute_policy_loss_sscr_icepop(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            infer_log_prob=infer_log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            consistency_lambda=sscr_params["lambda"],
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
            loss_agg_mode=loss_agg_mode,
        )
    elif loss_func_type == "grpo_clip":
        pg_loss, loss_metrics = compute_policy_loss_grpo_clip(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
            loss_agg_mode=loss_agg_mode,
        )
    elif loss_func_type == "lbpo":
        pg_loss, loss_metrics = compute_policy_loss_lbpo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
            loss_agg_mode=loss_agg_mode,
            lbpo_alpha=lbpo_alpha,
        )
    elif loss_func_type == "bapo":
        bapo_params = _parse_bapo_cfg(bapo_cfg)
        pg_loss, loss_metrics = compute_policy_loss_bapo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            init_lower_bound=bapo_params["init_lower_bound"],
            init_upper_bound=bapo_params["init_upper_bound"],
            max_lower_bound=bapo_params["max_lower_bound"],
            max_upper_bound=bapo_params["max_upper_bound"],
            lower_step=bapo_params["lower_step"],
            upper_step=bapo_params["upper_step"],
            positive_contribution_threshold=bapo_params["positive_contribution_threshold"],
            loss_agg_mode=loss_agg_mode,
        )
        mask_dict = None
    elif loss_func_type == "bapo_mask":
        bapo_params = _parse_bapo_cfg(bapo_cfg)
        pg_loss, loss_metrics = compute_policy_loss_bapo_mask(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            init_lower_bound=bapo_params["init_lower_bound"],
            init_upper_bound=bapo_params["init_upper_bound"],
            max_lower_bound=bapo_params["max_lower_bound"],
            max_upper_bound=bapo_params["max_upper_bound"],
            lower_step=bapo_params["lower_step"],
            upper_step=bapo_params["upper_step"],
            positive_contribution_threshold=bapo_params["positive_contribution_threshold"],
            loss_agg_mode=loss_agg_mode,
            adv_mask_delta=adv_mask_delta,
        )
        mask_dict = None
    elif loss_func_type == "ce-gppo":
        pg_loss, loss_metrics = compute_policy_loss_ce_gppo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
            beta1=ce_gppo_beta1,
            beta2=ce_gppo_beta2,
            loss_agg_mode=loss_agg_mode,
        )
    elif loss_func_type == "ce_gppo_mask":
        pg_loss, loss_metrics, mask_dict = compute_policy_loss_ce_gppo_mask(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
            beta1=ce_gppo_beta1,
            beta2=ce_gppo_beta2,
            loss_agg_mode=loss_agg_mode,
            adv_mask_delta=adv_mask_delta,
            return_masks=return_masks,
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
    elif loss_func_type == "gspo":
        pg_loss, loss_metrics = compute_policy_loss_gspo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
        )
    elif loss_func_type == "mgspo":
        pg_loss, loss_metrics = compute_policy_loss_mgspo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
        )
    elif loss_func_type == "grpo_mask":
        pg_loss, loss_metrics = compute_policy_loss_grpo_mask(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
            loss_agg_mode=loss_agg_mode,
            adv_mask_delta=adv_mask_delta,
        )
    elif loss_func_type == "grpo_min_max":
        pg_loss, loss_metrics = compute_policy_loss_grpo_min_max(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
            loss_agg_mode=loss_agg_mode,
        )
    elif loss_func_type in ("eb_grpo", "eb-grpo"):
        alpha = None
        for key in ("eb_grpo_alpha", "alpha"):
            try:
                alpha = data[key]
                break
            except Exception:
                continue
        pg_loss, loss_metrics = compute_policy_loss_eb_grpo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
            alpha=alpha,
        )
    elif loss_func_type == "fix_advantage":
        pg_loss, loss_metrics = compute_policy_loss_fix_advantage(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio,
            low_clip_range=clip_ratio_low,
            high_clip_range=clip_ratio_high,
            loss_agg_mode=loss_agg_mode,
        )
        mask_dict = None
    else:
        raise RuntimeError(f"Not support loss_func_type: {loss_func_type}")

    # Ensure all branches return mask_dict (None for non-ce_gppo_mask cases)
    if loss_func_type != "ce_gppo_mask":
        mask_dict = None

    return pg_loss, loss_metrics, mask_dict


def kl_penalty(
    logprob: torch.FloatTensor,
    ref_logprob: torch.FloatTensor,
    kl_penalty_type: str = "kl",
    old_logprob: torch.FloatTensor = None,
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob."""
    if kl_penalty_type == "kl":
        return logprob - ref_logprob

    if kl_penalty_type == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty_type == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty_type == "low_var_kl":
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-5, max=5)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty_type == "unbiased_k3_estimate":
        if old_logprob is None:
            raise ValueError("old_logprob must be provided for unbiased_k3_estimate")
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-5, max=5)
        ratio = torch.exp(kl)
        importance_log_ratio = torch.clamp(logprob - old_logprob, min=-5, max=5)
        importance_weight = torch.exp(importance_log_ratio)
        kld = importance_weight * (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty_type == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
