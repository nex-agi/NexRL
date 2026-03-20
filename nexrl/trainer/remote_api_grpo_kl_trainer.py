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
        self._teacher_url = teacher_train_service_config.url
        self._teacher_backend = teacher_train_service_config.backend

        # Client is created lazily in initialize_workers()
        self._teacher_service_client: TrainServiceClient | None = None

        logger.info(
            "RemoteApiGrpoKlTrainer initialized: teacher_identifier=%s",
            self._teacher_identifier,
        )

        # Student-to-teacher weight transfer frequency (<= 0 means disabled)
        algorithm_config = config.get("algorithm", {})
        self._student_transfer_freq = int(algorithm_config.get("student_transfer_freq", 0))

        # Warn if transfer enabled but model_path differs (likely different model size)
        if self._student_transfer_freq > 0:
            actor_train_service_config = get_train_service_config_by_role(train_service, "actor")
            actor_model_path = actor_train_service_config.get("model_path", "")
            teacher_model_path = self._teacher_config.get("model_path", "")
            if actor_model_path != teacher_model_path:
                logger.warning(
                    "[KL] student_transfer_freq=%d but actor and teacher have different "
                    "model_path: actor=%s, teacher=%s. "
                    "Make sure they are the same model size, otherwise "
                    "load_checkpoint will fail at runtime.",
                    self._student_transfer_freq,
                    actor_model_path,
                    teacher_model_path,
                )
            logger.info(
                "[KL] Student-to-teacher weight transfer enabled: freq=%d",
                self._student_transfer_freq,
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

        self._teacher_service_client = create_train_service_client(
            backend=self._teacher_backend,
            url=self._teacher_url,
            identifier=self._teacher_identifier,
        )
        logger.info(
            "[KL] Teacher service client created: url=%s, identifier=%s",
            self._teacher_url,
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

    # ========================================================================
    # Training Override (Student-to-Teacher Sync)
    # ========================================================================

    def _has_teacher(self) -> bool:
        """Check if a teacher client is available (either weaver or legacy backend)."""
        return self._teacher_training_client is not None or self._teacher_service_client is not None

    def train(self, trajectories: list[Trajectory]) -> dict:
        """Train with periodic student-to-teacher weight sync."""
        metrics = super().train(trajectories)

        # self._train_step is incremented in BaseTrainer._main_loop() AFTER
        # train() returns, so use (self._train_step + 1) for the post-increment value.
        if self._student_transfer_freq > 0 and self._has_teacher():
            next_step = self._train_step + 1
            if next_step % self._student_transfer_freq == 0:
                self._sync_student_to_teacher(next_step, metrics)

        return metrics

    def _sync_student_to_teacher(self, step: int, metrics: dict) -> None:
        """
        Save actor (student) weights, then load them into teacher (ref model).
        Synchronous — blocks until the teacher finishes loading.
        """
        import time

        from ..executor import execute

        logger.info(
            "[KL] Syncing student weights to teacher at step %d (student_transfer_freq=%d)",
            step,
            self._student_transfer_freq,
        )

        sync_start = time.time()

        # Step 1: Save actor weights to a checkpoint path
        assert self._service_holder is not None, "Service holder must be set before sync"
        checkpoint_path = execute(
            self._service_holder.save_state,
            name=f"ref_sync_step_{step}",
        )
        logger.info("[KL] Actor checkpoint saved to: %s", checkpoint_path)

        # Step 2: Load checkpoint into teacher (blocking)
        if self._teacher_training_client is not None:
            # Weaver backend: teacher is a weaver TrainingClient
            self._teacher_training_client.load_state(checkpoint_path)
        else:
            # Legacy direct-zmq backend
            assert self._teacher_service_client is not None
            self._teacher_service_client.load_checkpoint(
                path=checkpoint_path,
                del_local_after_load=False,
                load_weight_only=True,
            )

        sync_time = time.time() - sync_start
        logger.info(
            "[KL] Student-to-teacher sync completed in %.2fs at step %d",
            sync_time,
            step,
        )
        metrics["kl/student_transfer_time"] = sync_time
        metrics["kl/student_transfer_step"] = step
