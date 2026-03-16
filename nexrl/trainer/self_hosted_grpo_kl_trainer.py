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
SelfHostedGrpoKlTrainer - Self-hosted GRPO trainer with KL divergence from a reference model.
"""

import logging
import os

import torch
from omegaconf import DictConfig, OmegaConf

from ..nexrl_types import Batch
from ..train_service_client import TrainServiceClient
from ..utils.config_utils import get_train_service_config_by_role
from ..utils.init_utils import create_train_service_client
from .self_hosted_grpo_trainer import SelfHostedGrpoTrainer

logger = logging.getLogger(__name__)


class SelfHostedGrpoKlTrainer(SelfHostedGrpoTrainer):
    """
    Self-hosted GRPO trainer with KL divergence from a reference/teacher model.

    Extends SelfHostedGrpoTrainer to compute reference model log probabilities
    using a separate teacher service, enabling KL-constrained policy optimization.

    Key features:
    - Dual world size support (actor + teacher may run on different GPU counts)
    - Batch padding/unpadding for both actor and teacher
    - Reference log-prob computation via teacher service client
    - scoring_attention_mask support for teacher
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the self-hosted GRPO KL trainer.

        Args:
            config: Configuration with both actor and teacher train service settings
        """
        # Extract teacher config before calling super().__init__() so that
        # all attributes are present before any method that might reference them.
        train_service = config.get("train_service")
        if not train_service:
            raise ValueError("train_service must be specified")

        teacher_train_service_config = get_train_service_config_by_role(train_service, "teacher")
        self._teacher_identifier = teacher_train_service_config.get("identifier", "teacher")
        self._teacher_url = teacher_train_service_config.url
        self._teacher_backend = teacher_train_service_config.backend
        self._teacher_init_config = teacher_train_service_config
        self._teacher_initialized = False

        # Teacher client is created lazily in initialize_workers()
        self._teacher_service_client: TrainServiceClient | None = None

        # Call parent __init__ — uses the actor train service config internally
        super().__init__(config)

        # Compute teacher world size using the same formula as SelfHostedTrainer
        teacher_node_world_size = teacher_train_service_config.resource.get("world_size", None)
        if teacher_node_world_size is None:
            raise ValueError(
                "world_size must be specified in teacher train_service.resource config"
            )
        teacher_gpus_per_pod = teacher_train_service_config.resource.get("gpus_per_pod", None)
        if teacher_gpus_per_pod is None:
            raise ValueError(
                "gpus_per_pod must be specified in teacher train_service.resource config"
            )
        self._teacher_world_size = int(teacher_node_world_size * teacher_gpus_per_pod)

        logger.info(
            "SelfHostedGrpoKlTrainer initialized with teacher world_size=%d, "
            "teacher_identifier=%s",
            self._teacher_world_size,
            self._teacher_identifier,
        )

    # ========================================================================
    # Initialization Methods
    # ========================================================================

    def initialize_workers(self) -> None:
        """Initialize both actor and teacher worker groups."""
        super().initialize_workers()
        self._initialize_teacher_workers()

    def _initialize_teacher_workers(self) -> None:
        """Initialize teacher workers and load model onto GPU."""
        if self._teacher_initialized:
            return

        backend = self._teacher_backend

        if backend == "direct-zmq":
            logger.info("[KL] Initializing teacher workers...")
            try:
                # Create teacher client on first use
                if self._teacher_service_client is None:
                    self._teacher_service_client = create_train_service_client(
                        backend=self._teacher_backend,
                        url=self._teacher_url,
                        identifier=self._teacher_identifier,
                    )
                    logger.info(
                        "[KL] Created teacher client with URL: %s, identifier: %s",
                        self._teacher_url,
                        self._teacher_identifier,
                    )

                config_dict = OmegaConf.to_container(self._teacher_init_config, resolve=True)

                teacher_role = self._teacher_init_config.get("role", "")
                assert teacher_role == "teacher", "Teacher role must be 'teacher'"

                try:
                    result = self._teacher_service_client.initialize_worker(
                        config_dict=config_dict, role=teacher_role
                    )
                    logger.info("[KL] Teacher workers initialized: %s", result)
                except Exception as e:
                    experiment_path = os.environ.get("EXPERIMENT_PATH", "EXPERIMENT_PATH")
                    api_server_log = f"{experiment_path}/api_server.log"
                    workers_log = f"{experiment_path}/workers-{self._teacher_identifier}-rank*.log"
                    logger.error(
                        "[KL] Failed to initialize teacher workers: %s. "
                        "Check %s and %s for details.",
                        e,
                        api_server_log,
                        workers_log,
                    )
                    raise RuntimeError(
                        f"Teacher worker initialization failed: {e}. "
                        f"Check {api_server_log} and {workers_log} for details."
                    ) from e

                logger.info("[KL] Initializing teacher model on workers...")
                result = self._teacher_service_client.init_model()
                logger.info("[KL] Teacher model initialized: %s", result)

                self._teacher_initialized = True

            except Exception as e:
                logger.error("[KL] Failed to initialize teacher workers: %s", e)
                raise RuntimeError(f"Teacher worker initialization failed: {e}") from e

        elif backend == "mock":
            logger.info("[KL] Mock backend — skipping teacher worker initialization")
            self._teacher_initialized = True
        else:
            raise ValueError(
                f"[KL] Unknown teacher backend '{backend}' — cannot initialize workers"
            )

    # ========================================================================
    # Batch Preparation Override
    # ========================================================================

    def _compute_ref_log_probs(self, batch: Batch) -> torch.Tensor:
        """
        Compute reference model log probabilities using the teacher service.

        Handles padding for the teacher's world size if different from the actor's.

        Args:
            batch: Batch to compute reference log probs for

        Returns:
            Reference log probabilities tensor (batch_size, response_length)
        """
        assert (
            self._teacher_service_client is not None
        ), "Teacher service client not initialized — call initialize_workers() first"

        original_batch_size = len(batch)

        # Pad batch up to teacher world size if needed
        teacher_batch = batch.pad_to_world_size(self._teacher_world_size)

        with self._teacher_service_client.actor_context():
            trimmed = teacher_batch.trim_for_backend(self._BACKEND_COMPUTE_LOG_PROB_KEYS)
            nextrainer_batch = trimmed.to_nextrainer_batch()
            ret = self._teacher_service_client.compute_log_prob(nextrainer_batch)

        ref_log_probs: torch.Tensor = ret["batch"]["old_log_probs"]

        # Strip padding rows introduced above
        if ref_log_probs.shape[0] > original_batch_size:
            ref_log_probs = ref_log_probs[:original_batch_size]

        return ref_log_probs

    def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
        """
        Prepare batch with GRPO advantages and reference model log probs.

        Extends the parent's _prepare_batch to additionally compute
        reference model log probabilities for KL penalty.

        Args:
            batch: Batch of trajectory data from rollout

        Returns:
            Tuple of (prepared_batch, metrics_dict)
        """
        # Run standard GRPO preparation (advantage computation, KL stats, etc.)
        batch, metrics = super()._prepare_batch(batch)

        # Compute reference log probs from teacher model
        logger.info("[KL] Computing reference model log probabilities")
        ref_log_probs = self._compute_ref_log_probs(batch)
        batch.values["ref_log_prob"] = ref_log_probs
        logger.debug("[KL] Reference log probs shape: %s", ref_log_probs.shape)

        return batch, metrics
