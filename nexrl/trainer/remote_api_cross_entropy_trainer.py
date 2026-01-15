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
RemoteApiCrossEntropyTrainer - Remote API trainer with Cross Entropy loss
"""

import logging
from typing import Any

from omegaconf import DictConfig

from ..nexrl_types import Trajectory
from .remote_api_trainer import RemoteApiTrainer

logger = logging.getLogger(__name__)


class RemoteApiCrossEntropyTrainer(RemoteApiTrainer):
    """
    Remote API trainer with Cross Entropy loss for supervised learning.

    This trainer extends RemoteApiTrainer for supervised fine-tuning tasks where:
    - Trajectories contain ground truth labels (via loss_mask)
    - No advantage computation is needed (unlike GRPO)
    - Uses cross_entropy loss function directly

    Works with any remote API backend (Tinker, Weaver, etc.).
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the Remote API Cross Entropy trainer.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Override loss function to cross_entropy for supervised learning
        train_config = config.get("train_service", {}).get("config", {})
        self._loss_fn = train_config.get("loss_fn", "cross_entropy")

        logger.info(f"RemoteApiCrossEntropyTrainer initialized with loss_fn={self._loss_fn}")

    def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """
        Prepare trajectories for cross entropy training.

        For supervised learning with cross entropy, no trajectory preparation
        (like advantage computation) is needed. The loss_mask already indicates
        which tokens should contribute to the loss.

        Args:
            trajectories: List of Trajectory dataclasses with tokens and loss_mask
            metrics: Dictionary to store preparation metrics

        Returns:
            Trajectories unchanged (no preparation needed for cross entropy)
        """
        return trajectories
