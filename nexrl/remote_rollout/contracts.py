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

"""Data exchanged between the Driver and an E2B rollout."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

_MODEL_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _validate_rollout_id(value: str) -> str:
    if not value.strip():
        raise ValueError("rollout_id must be a non-empty string")
    return value


class RemoteRolloutRequest(BaseModel):
    """Stable per-run data sent from the Driver to E2B."""

    model_config = _MODEL_CONFIG

    rollout_id: str
    task: dict[str, JsonValue]

    _rollout_id_must_not_be_empty = field_validator("rollout_id")(_validate_rollout_id)


class RemoteTrajectory(BaseModel):
    """One token-faithful training trajectory produced inside E2B."""

    model_config = _MODEL_CONFIG

    tokens: list[int] = Field(min_length=1)
    loss_mask: list[int]
    old_log_probs: list[float]
    sampling_mask: list[list[int]] | None = None

    @model_validator(mode="after")
    def _validate_alignment(self) -> Self:
        token_count = len(self.tokens)
        if len(self.loss_mask) != token_count or len(self.old_log_probs) != token_count:
            raise ValueError("tokens, loss_mask, and old_log_probs must have equal length")
        if any(token < 0 for token in self.tokens):
            raise ValueError("tokens must contain non-negative token IDs")
        if any(value not in (0, 1) for value in self.loss_mask):
            raise ValueError("loss_mask must contain only 0 or 1")
        if self.loss_mask[0] != 0:
            raise ValueError("loss_mask[0] must be 0; the first token has no target logprob")
        if not any(self.loss_mask):
            raise ValueError("a trajectory must contain at least one response token")
        if self.old_log_probs[0] != 0.0:
            raise ValueError("old_log_probs[0] must be the 0.0 placeholder")
        if any(
            log_prob != 0.0
            for selected, log_prob in zip(self.loss_mask, self.old_log_probs)
            if selected == 0
        ):
            raise ValueError("context token old_log_probs must use the 0.0 placeholder")

        if self.sampling_mask is None:
            return self

        sampled_tokens = [
            token for token, selected in zip(self.tokens, self.loss_mask) if selected == 1
        ]
        if len(self.sampling_mask) != len(sampled_tokens):
            raise ValueError("sampling_mask must have one entry for each response token")
        for index, (candidates, sampled_token) in enumerate(
            zip(self.sampling_mask, sampled_tokens)
        ):
            if not candidates:
                raise ValueError(f"sampling_mask[{index}] must not be empty")
            if any(candidate < 0 for candidate in candidates):
                raise ValueError(f"sampling_mask[{index}] must contain non-negative token IDs")
            if sampled_token not in candidates:
                raise ValueError(
                    f"sampling_mask[{index}] must contain sampled token {sampled_token}"
                )
        return self


class RemoteRolloutResult(BaseModel):
    """Successful E2B rollout result returned to the Driver."""

    model_config = _MODEL_CONFIG

    rollout_id: str
    trajectories: list[RemoteTrajectory] = Field(min_length=1)
    reward: float
    metrics: dict[str, float] = Field(default_factory=dict)

    _rollout_id_must_not_be_empty = field_validator("rollout_id")(_validate_rollout_id)
