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


def convert_trajectories_to_datums(trajectories: list[Trajectory]) -> list[dict]:
    """
    Convert trajectories to serializable datum dictionaries for IS loss.
    """
    datums_data = []
    for traj in trajectories:
        tokens = traj["tokens"]
        logprobs = traj["logprobs"]
        loss_mask = traj["loss_mask"]
        advantage = traj["advantage"]
        # if advantage == 0.0:
        #     continue

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        adjusted_logprobs = logprobs[1:] if len(logprobs) == len(tokens) else logprobs
        adjusted_loss_mask = loss_mask[1:]
        token_advantages = [float(advantage * mask) for mask in adjusted_loss_mask]
        loss_fn_inputs = {
            "target_tokens": target_tokens,
            "logprobs": [float(lp) for lp in adjusted_logprobs],
            "advantages": token_advantages,
            # Token-level mask aligned with target_tokens (0 for prompt / ignored tokens, 1 for response tokens).
            # This is used by the trainer to compute loss/entropy over response tokens only.
            "loss_mask": [int(m) for m in adjusted_loss_mask],
        }

        datum_data = {
            "input_tokens": input_tokens,
            "loss_fn_inputs": loss_fn_inputs,
        }
        datums_data.append(datum_data)

    return datums_data
