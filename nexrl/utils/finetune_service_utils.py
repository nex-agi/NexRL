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

from __future__ import annotations

from ..nexrl_types import Trajectory  # pylint: disable=relative-beyond-top-level


def convert_trajectories_to_datums(
    trajectories: list[Trajectory],
    use_old_logprobs: bool = False,
) -> list[dict]:
    """
    Convert trajectories to serializable datum dictionaries for the importance_sampling loss.

    Handles both scalar-advantage (standard RL) and per-token-advantage (OPD) trajectories
    automatically:
    - If ``traj.extra_fields["token_advantages"]`` is present and non-empty, those
      per-token advantages are used (OPD path).
    - Otherwise the scalar ``traj["advantage"]`` is broadcast to every response token
      (standard GRPO path).

    The returned dicts always contain a ``loss_mask`` key so the weaver backend can
    use it.  The tinker backend strips it before sending to the server (the tinker
    server does not accept extra fields in loss_fn_inputs).

    Args:
        trajectories: List of Trajectory objects.
        use_old_logprobs: When True, use old_log_probs (from rollout server)
            as the sampling logprobs for the IS ratio instead of the default
            logprobs. Falls back to logprobs if old_log_probs is absent.
    """
    datums_data = []
    for traj in trajectories:
        tokens = traj["tokens"]
        logprobs = traj["logprobs"]
        loss_mask = traj["loss_mask"]
        sampling_logprobs = logprobs
        if use_old_logprobs:
            old_lp = traj.get("old_log_probs")
            if old_lp is not None:
                sampling_logprobs = old_lp

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        adjusted_logprobs = (
            sampling_logprobs[1:] if len(sampling_logprobs) == len(tokens) else sampling_logprobs
        )
        adjusted_loss_mask = loss_mask[1:]

        # --- decide per-token advantages ---
        per_token_adv = traj.extra_fields.get("token_advantages") if traj.extra_fields else None
        if per_token_adv:
            # OPD path: expand response-only advantages back to full shifted length
            token_advantages = []
            response_idx = 0
            for mask in adjusted_loss_mask:
                if mask == 1 and response_idx < len(per_token_adv):
                    token_advantages.append(float(per_token_adv[response_idx]))
                    response_idx += 1
                else:
                    token_advantages.append(0.0)
        else:
            # Standard path: broadcast scalar advantage with mask
            advantage = traj["advantage"]
            token_advantages = [float(advantage * m) for m in adjusted_loss_mask]

        if not any(a != 0.0 for a in token_advantages):
            continue

        loss_fn_inputs = {
            "target_tokens": target_tokens,
            "logprobs": [float(lp) for lp in adjusted_logprobs],
            "advantages": token_advantages,
            "loss_mask": [float(m) for m in adjusted_loss_mask],
        }

        datums_data.append(
            {
                "input_tokens": input_tokens,
                "loss_fn_inputs": loss_fn_inputs,
            }
        )

    return datums_data


def convert_trajectories_to_datums_cross_entropy(trajectories: list[Trajectory]) -> list[dict]:
    """
    Convert trajectories to serializable datum dictionaries for cross-entropy loss (SFT).

    Uses a minimal protocol: only ``target_tokens`` and ``loss_mask`` are required —
    no ``logprobs`` or ``advantages``.
    """
    datums_data = []
    for traj in trajectories:
        tokens = traj["tokens"]
        loss_mask = traj["loss_mask"]

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        adjusted_loss_mask = loss_mask[1:]

        datum_data = {
            "input_tokens": input_tokens,
            "loss_fn_inputs": {
                "target_tokens": target_tokens,
                "loss_mask": [int(m) for m in adjusted_loss_mask],
            },
        }
        datums_data.append(datum_data)

    return datums_data
