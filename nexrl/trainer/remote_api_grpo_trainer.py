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
RemoteApiGrpoTrainer - Remote API trainer with GRPO algorithm
"""

import logging
from typing import Any

from omegaconf import DictConfig

from ..nexrl_types import Trajectory
from .remote_api_trainer import RemoteApiTrainer

logger = logging.getLogger(__name__)


class RemoteApiGrpoTrainer(RemoteApiTrainer):
    """
    Remote API trainer with GRPO (Group Relative Policy Optimization) advantage computation.

    Extends RemoteApiTrainer with GRPO-specific trajectory preparation.
    Works with any remote API backend (Tinker, Weaver, etc.).
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the Remote API GRPO trainer.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        logger.info("RemoteApiGrpoTrainer initialized")

    def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """
        Prepare trajectories with GRPO advantage computation.

        Args:
            trajectories: List of Trajectory dataclasses
            metrics: Dictionary to store GRPO metrics

        Returns:
            Trajectories with computed GRPO advantages
        """
        from ..algorithm.core_algos import compute_grpo_advantage_for_trajectories
        from ..utils.logging_utils import log_grpo_metrics

        # Deterministically order trajectories so GRPO grouping is stable across runs.
        # Rollouts can arrive asynchronously; this prevents nondeterministic group/sample order
        # from affecting logged tensors/metrics.
        def _to_py(v: Any) -> Any:
            if hasattr(v, "item"):
                try:
                    return v.item()
                except Exception:
                    return v
            return v

        def _stable_sort_key(x: Any) -> tuple[int, Any]:
            x_py = _to_py(x)
            if isinstance(x_py, int):
                return (0, x_py)
            return (1, str(x_py))

        def _traj_key(traj: Trajectory) -> tuple[tuple[int, Any], tuple[int, Any], int]:
            group = traj.get("uid") or traj.get("group_id") or ""
            run = traj.get("run_id", 0)
            try:
                run_i = int(_to_py(run))
            except Exception:
                run_i = 0

            # Prefer stable task ordering keys if present (dataloader may inject these).
            primary = traj.get("task_id", None)
            if primary is None:
                primary = traj.get("index", None)
            if primary is None:
                primary = group

            return (_stable_sort_key(primary), _stable_sort_key(group), run_i)

        trajectories = sorted(trajectories, key=_traj_key)

        # Compute GRPO advantages
        trajectories = compute_grpo_advantage_for_trajectories(
            trajectories, logger=logger, use_run_ids=True
        )

        # Log GRPO statistics
        log_grpo_metrics(trajectories, metrics)

        # Compute and log response length statistics
        self._log_response_length_metrics(trajectories, metrics)

        return trajectories

    def _log_response_length_metrics(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> None:
        """
        Compute and log response length statistics from trajectories.

        Args:
            trajectories: List of Trajectory dataclasses
            metrics: Dictionary to store response length metrics
        """
        import numpy as np

        # Compute response length from loss_mask (sum of 1s indicates response tokens)
        response_lengths = []
        prompt_lengths = []

        for traj in trajectories:
            # Response length is the sum of loss_mask
            response_length = sum(traj.loss_mask)
            response_lengths.append(response_length)

            # Prompt length is total tokens minus response tokens
            prompt_length = len(traj.tokens) - response_length
            prompt_lengths.append(prompt_length)

        if response_lengths:
            # Response length metrics
            metrics["response_length/mean"] = float(np.mean(response_lengths))
            metrics["response_length/max"] = float(np.max(response_lengths))
            metrics["response_length/min"] = float(np.min(response_lengths))

            # Clip ratio: fraction of responses at max length
            max_response_length = max(response_lengths)
            metrics["response_length/clip_ratio"] = float(
                np.mean([1.0 if r == max_response_length else 0.0 for r in response_lengths])
            )

        if prompt_lengths:
            # Prompt length metrics
            metrics["prompt_length/mean"] = float(np.mean(prompt_lengths))
            metrics["prompt_length/max"] = float(np.max(prompt_lengths))
            metrics["prompt_length/min"] = float(np.min(prompt_lengths))

            # Clip ratio: fraction of prompts at max length
            max_prompt_length = max(prompt_lengths)
            metrics["prompt_length/clip_ratio"] = float(
                np.mean([1.0 if p == max_prompt_length else 0.0 for p in prompt_lengths])
            )
