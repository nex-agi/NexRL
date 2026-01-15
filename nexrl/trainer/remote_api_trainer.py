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
RemoteApiTrainer - Base trainer for remote API backends (Tinker/Weaver)
Works directly with trajectories, inline GRPO advantage computation.
"""

import logging
from abc import abstractmethod
from typing import Any

from omegaconf import DictConfig

from ..nexrl_types import Trajectory
from ..tinker.tinker_service_holder import TinkerServiceHolder
from ..utils.finetune_service_utils import convert_trajectories_to_datums
from ..utils.logging_utils import log_rollout_metrics
from ..weaver.weaver_service_holder import WeaverServiceHolder
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class RemoteApiTrainer(BaseTrainer):
    """
    Base trainer implementation for remote API backends.

    This class provides common training logic for API-backed trainers
    (Tinker, Weaver, etc.) that use remote API services for distributed training.

    Key features:
    - Works directly with trajectories (no Batch conversion)
    - Inline GRPO advantage computation with std normalization
    - Uses service holder pattern for forward_backward and optim_step
    - No padding removal needed (service handles variable-length sequences)

    Derived classes should:
    - Set _service_holder in their set_service_holder() method
    - Optionally override methods for custom behavior
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the remote API trainer.

        Args:
            config: Configuration dictionary containing:
                - train_service.config: Training hyperparameters (lr, beta1, etc.)
        """
        super().__init__(config)

        # Service holder will be set via set_service_holder() in derived class
        self._service_holder = None

        # Training hyperparameters from config
        train_config = config.get("train_service", {}).get("config", {})
        self._loss_fn = train_config.get("loss_fn", "importance_sampling")
        self._learning_rate = train_config.get("learning_rate", 2e-6)
        self._beta1 = train_config.get("beta1", 0.9)
        self._beta2 = train_config.get("beta2", 0.95)
        self._eps = train_config.get("eps", 1e-8)

        # Step counter for weight saving
        self._step_counter = 0

        logger.info(
            f"RemoteApiTrainer initialized: loss_fn={self._loss_fn}, "
            f"lr={self._learning_rate}, beta1={self._beta1}, beta2={self._beta2}"
        )

    def set_service_holder(self, service_holder: TinkerServiceHolder | WeaverServiceHolder) -> None:
        """
        Set the service holder.

        Args:
            service_holder: Service holder instance (TinkerServiceHolder, WeaverServiceHolder, etc.)
        """
        self._service_holder = service_holder  # type: ignore

    @abstractmethod
    def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """
        Prepare trajectories for training (e.g., compute advantages).

        Default implementation is a no-op. Override in derived classes
        to perform algorithm-specific trajectory preparation.

        Args:
            trajectories: List of Trajectory dataclasses with tokens, loss_mask, and logprobs
            metrics: Dictionary to store preparation metrics

        Returns:
            Prepared trajectories with computed advantages/rewards
        """

    def train(self, trajectories: list[Trajectory]) -> dict:
        """
        Train on a list of trajectories.

        Steps:
        1. Log rollout metrics
        2. Compute GRPO advantages (std-normalized)
        3. Convert trajectories to service Datum format
        4. Execute forward_backward + optim_step
        5. Save weights and update sampling client

        Args:
            trajectories: List of Trajectory dataclasses with tokens, loss_mask, and logprobs

        Returns:
            Dictionary of training metrics
        """
        assert self._service_holder is not None, "ServiceHolder not set"

        metrics: dict[str, Any] = {}

        # Step 1: Log rollout metrics
        log_rollout_metrics(trajectories, metrics)

        logger.info(f"rollout metrics: {metrics}, trajectories: {len(trajectories)}")

        # Step 2: Prepare trajectories (e.g., compute advantages)
        trajectories = self._prepare_trajectories(trajectories, metrics)

        # Step 3: Convert to service Datum format
        datums_data = convert_trajectories_to_datums(trajectories)

        if len(datums_data) == 0:
            logger.warning("No valid datums after filtering, skipping training step")
            metrics["training/skipped"] = 1
            return metrics

        # Step 4: Execute forward_backward and optim_step together
        from ..executor import execute

        training_metrics = execute(
            self._service_holder.forward_backward_and_optim_step,
            datums_data=datums_data,
            loss_fn=self._loss_fn,
            learning_rate=self._learning_rate,
            beta1=self._beta1,
            beta2=self._beta2,
            eps=self._eps,
        )

        # Step 5: Save weights and update sampling client
        self._step_counter += 1
        new_sampling_path = execute(
            self._service_holder.save_weights_for_sampler,
            name=f"step_{self._step_counter:06d}",
        )

        # Update sampling client path (actual sync is handled by WeightSyncController)
        execute(
            self._service_holder.set_current_sampling_path,
            new_sampling_path,
        )

        logger.info(
            f"Training step {self._train_step} completed, "
            f"updated sampling path: {new_sampling_path}"
        )

        # Add training metrics
        metrics.update(
            {
                "training/global_step": self._train_step,
                "training/batch_size": len(trajectories),
                "training/num_datums": len(datums_data),
            }
        )

        # Add all metrics from forward_backward result (with training/ prefix)
        for key, value in training_metrics.items():
            metrics[f"training/{key}"] = value

        return metrics
