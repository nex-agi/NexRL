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
RemoteApiGrpoKlTrainer - Remote API GRPO trainer with KL divergence from a reference model.
"""

import logging
from typing import Any

from omegaconf import DictConfig

from ..nexrl_types import Trajectory
from ..train_service_client import TrainServiceClient
from ..utils.config_utils import get_train_service_config_by_role
from ..utils.init_utils import create_train_service_client
from .remote_api_grpo_trainer import RemoteApiGrpoTrainer

logger = logging.getLogger(__name__)


class RemoteApiGrpoKlTrainer(RemoteApiGrpoTrainer):
    """
    Remote API GRPO trainer with KL divergence from a reference/teacher model.

    Extends RemoteApiGrpoTrainer to add reference model log-prob computation
    via a teacher service, enabling KL-constrained policy optimization
    in remote API training mode.

    The teacher service client is initialized eagerly in initialize_workers().
    Trajectories are tagged with ``use_kl=True`` so that downstream datum
    converters can activate the KL loss path.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the Remote API GRPO KL trainer.

        Args:
            config: Configuration with both actor and teacher train service settings
        """
        super().__init__(config)

        # Extract teacher service configuration
        train_service = config.get("train_service")
        if not train_service:
            raise ValueError("train_service must be specified")

        teacher_train_service_config = get_train_service_config_by_role(train_service, "teacher")
        self._teacher_identifier = teacher_train_service_config.get("identifier", "teacher")
        self._teacher_backend = teacher_train_service_config.get("backend", "mock")
        self._teacher_config = teacher_train_service_config

        # Clients are created lazily in initialize_workers()
        self._teacher_service_client: TrainServiceClient | None = None
        self._teacher_training_client = None  # For Weaver backend

        logger.info(
            "RemoteApiGrpoKlTrainer initialized: teacher_identifier=%s, teacher_backend=%s",
            self._teacher_identifier,
            self._teacher_backend,
        )

    # ========================================================================
    # Initialization Methods
    # ========================================================================

    def initialize_workers(self) -> None:
        """Initialize both actor and teacher worker groups."""
        super().initialize_workers()

        if self._teacher_backend == "mock":
            logger.info("[KL] Mock backend — skipping teacher worker initialization")
            return

        if self._teacher_backend == "weaver":
            # Create a new training instance through the Weaver service holder.
            # Teacher model config can be specified explicitly in the teacher's
            # train_service config; falls back to the actor's values via the
            # shared WeaverServiceHolder.
            assert (
                self._service_holder is not None
            ), "Service holder must be set before initialize_workers"
            service_client = self._service_holder.get_service_client()

            holder_base_model = self._service_holder._base_model  # pylint: disable=protected-access
            holder_lora_rank = self._service_holder._lora_rank  # pylint: disable=protected-access
            base_model = self._teacher_config.get("base_model", holder_base_model)
            training_mode = self._teacher_config.get("training_mode", "lora")
            lora_rank = self._teacher_config.get("lora_rank", holder_lora_rank)

            self._teacher_training_client = service_client.create_model(
                base_model=base_model,
                training_mode=training_mode,
                lora_config={"rank": lora_rank},
            )
            logger.info(
                "[KL] Teacher training client created via Weaver service: "
                "identifier=%s, base_model=%s, training_mode=%s, lora_rank=%s",
                self._teacher_identifier,
                base_model,
                training_mode,
                lora_rank,
            )
            return

        # Legacy: URL-based approach for direct-zmq backend
        teacher_url = self._teacher_config.get("url")
        if not teacher_url:
            raise ValueError(
                f"Teacher train service config must have 'url' for backend '{self._teacher_backend}'"
            )
        self._teacher_service_client = create_train_service_client(
            backend=self._teacher_backend,
            url=teacher_url,
            identifier=self._teacher_identifier,
        )
        logger.info(
            "[KL] Teacher service client created: url=%s, identifier=%s",
            teacher_url,
            self._teacher_identifier,
        )

    # ========================================================================
    # Trajectory Preparation Override
    # ========================================================================

    def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """
        Prepare trajectories with GRPO advantages and reference model info.

        Extends parent to tag trajectories with ``use_kl=True`` so that
        downstream datum converters activate the KL loss path.

        Args:
            trajectories: List of Trajectory dicts
            metrics: Dictionary to store metrics

        Returns:
            Trajectories with computed advantages and KL training tag
        """
        # Run standard GRPO preparation (advantage computation, sorting, metrics)
        trajectories = super()._prepare_trajectories(trajectories, metrics)

        # Tag trajectories to indicate KL training mode
        for traj in trajectories:
            traj["use_kl"] = True

        return trajectories
