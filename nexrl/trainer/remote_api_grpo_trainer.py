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

        # Compute GRPO advantages
        trajectories = compute_grpo_advantage_for_trajectories(
            trajectories, logger=logger, use_run_ids=True
        )

        # Log GRPO statistics
        log_grpo_metrics(trajectories, metrics)

        return trajectories
