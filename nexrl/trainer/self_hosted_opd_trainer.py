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
SelfHostedOpdTrainer - Self-hosted trainer for On-Policy Distillation

This trainer implements on-policy distillation where:
1. Student model generates trajectories (rollout)
2. Teacher model provides LOG PROBABILITIES for those trajectories
3. Student is trained to minimize reverse KL: KL(student || teacher)

Memory-efficient implementation using log probabilities instead of full logits:
- ~40,000x memory reduction for teacher signal
- Mathematically equivalent for sampled tokens
- Faster computation (no need for softmax + gather)

Reference: https://thinkingmachines.ai/blog/on-policy-distillation/
"""

import copy
import logging
import os
import time
from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ..nexrl_types import Batch
from ..train_service_client import TrainServiceClient
from ..utils.config_utils import get_train_service_config_by_role
from ..utils.init_utils import create_train_service_client
from .self_hosted_trainer import SelfHostedTrainer

logger = logging.getLogger(__name__)


class SelfHostedOpdTrainer(SelfHostedTrainer):
    """
    Self-hosted trainer for On-Policy Distillation (OPD).

    Extends SelfHostedTrainer with OPD-specific functionality:
    1. Converts trajectories to batch format (inherited)
    2. Gets teacher log probabilities for student trajectories
    3. Prepares batch for distillation training
    4. Sends the processed batch to the train service (inherited)
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the self-hosted OPD trainer

        Args:
            config: Configuration dictionary with training settings and OPD parameters
        """
        # OPD requires explicit identifiers for multi-worker-group setup
        # We need to set up both student (actor) and teacher service configs

        # Get train_service configs
        train_service = config.get("train_service")

        if not train_service:
            raise ValueError("train_service must be specified")

        # Get student (actor) service config
        student_service_config = get_train_service_config_by_role(train_service, "actor")
        self._student_identifier = student_service_config.get("identifier", "student")

        # Get teacher service config
        teacher_service_config = get_train_service_config_by_role(train_service, "teacher")
        self._teacher_identifier = teacher_service_config.get("identifier", "teacher")
        self._teacher_url = teacher_service_config.url
        self._teacher_backend = teacher_service_config.backend
        self._teacher_init_config = teacher_service_config
        self._teacher_initialized = False

        # Teacher client will be created lazily on first use
        self._teacher_client: TrainServiceClient | None = None

        # Call parent __init__ - it will use actor_train_service to create the student client
        super().__init__(config)

        # setup teacher world size
        teacher_node_world_size = self._teacher_init_config.resource.get("world_size", None)
        if teacher_node_world_size is None:
            raise ValueError(
                "world_size must be specified in teacher train_service.resource config"
            )
        gpus_per_pod = self._teacher_init_config.resource.get("gpus_per_pod", None)
        if gpus_per_pod is None:
            raise ValueError(
                "gpus_per_pod must be specified in teacher train_service.resource config"
            )
        self._teacher_world_size = int(teacher_node_world_size * gpus_per_pod)
        logger.info(
            f"SelfHostedOpdTrainer [teacher] initialized with world size: {self._teacher_world_size}"
        )

        # OPD algorithm configuration
        self._algorithm_config = config.algorithm
        self._pad_token_id = self.tokenizer.pad_token_id

        # Distillation hyperparameters
        self._distillation_coeff = self._algorithm_config.get("distillation_coeff", 1.0)
        self._entropy_coeff = self._algorithm_config.get("entropy_coeff", 0.0)
        self._temperature = self._algorithm_config.get("temperature", 1.0)
        self._loss_agg_mode = self._algorithm_config.get("loss_agg_mode", "token-mean")
        self._distillation_epochs = self._algorithm_config.get("distillation_epochs", 1)
        self._do_old_student_log_prob_compute = self._algorithm_config.get(
            "do_old_student_log_prob_compute", False
        )

        logger.info("SelfHostedOpdTrainer initialized with:")
        logger.info(f"  student_identifier={self._student_identifier}")
        logger.info(f"  teacher_identifier={self._teacher_identifier}")
        logger.info(f"  distillation_coeff={self._distillation_coeff}")
        logger.info(f"  entropy_coeff={self._entropy_coeff}")
        logger.info(f"  temperature={self._temperature}")
        logger.info(f"  loss_agg_mode={self._loss_agg_mode}")
        logger.info(f"  distillation_epochs={self._distillation_epochs}")
        logger.info(f"  teacher_url={self._teacher_url}")

    def initialize_workers(self) -> None:
        """
        Initialize both student and teacher workers.
        This overrides the parent method to also initialize teacher workers.
        """
        # Initialize student workers (via parent class)
        super().initialize_workers()

        # Initialize teacher workers
        self._initialize_teacher_workers()

    def _initialize_teacher_workers(self) -> None:
        """Initialize teacher workers and load model."""
        if self._teacher_initialized:
            return

        backend = self._teacher_backend

        if backend in ("direct-zmq"):
            logger.info("[OPD] Initializing teacher workers...")
            try:
                # Create teacher client if not already created
                if self._teacher_client is None:
                    self._teacher_client = create_train_service_client(
                        backend=self._teacher_backend,
                        url=self._teacher_url,
                        identifier=self._teacher_identifier,
                    )
                    logger.info(
                        f"[OPD] Created teacher client with URL: {self._teacher_url}, "
                        f"identifier: {self._teacher_identifier}"
                    )

                config_dict = OmegaConf.to_container(self._teacher_init_config, resolve=True)

                # Get the role from teacher config (defaults to "actor" for backward compatibility)
                teacher_role = self._teacher_init_config.get("role", "")
                assert teacher_role == "teacher", "Teacher role must be 'teacher'"

                # Initialize teacher workers with config
                try:
                    result = self._teacher_client.initialize_worker(
                        config_dict=config_dict, role=teacher_role
                    )
                    logger.info(f"[OPD] Teacher workers initialized: {result}")
                except Exception as e:
                    logger.error(f"[OPD] Failed to initialize teacher workers: {e}")
                    experiment_path = os.environ.get("EXPERIMENT_PATH", "EXPERIMENT_PATH")
                    api_server_log = f"{experiment_path}/api_server.log"
                    workers_log = f"{experiment_path}/workers-{self._teacher_identifier}-rank*.log"
                    logger.error(
                        f"Please check **{api_server_log}** and **{workers_log}** for details."
                    )
                    raise RuntimeError(
                        f"Teacher worker initialization failed: {e}. "
                        f"Please check **{api_server_log}** and **{workers_log}** for details."
                    ) from e

                # Initialize model (load to GPU)
                logger.info("[OPD] Initializing teacher model on workers...")
                result = self._teacher_client.init_model()
                logger.info(f"[OPD] Teacher model initialized: {result}")

                self._teacher_initialized = True

            except Exception as e:
                logger.error(f"[OPD] Failed to initialize teacher workers: {e}")
                raise RuntimeError(f"Teacher worker initialization failed: {e}") from e
        elif backend == "mock":
            logger.info("[OPD] Mock backend - skipping teacher worker initialization")
            self._teacher_initialized = True
        else:
            logger.error(
                f"[OPD] Unknown backend {backend} - skipping teacher worker initialization"
            )
            raise ValueError(f"Unknown backend {backend} - skipping teacher worker initialization")

    # ========================================================================
    # Batch Preparation (Override from SelfHostedTrainer)
    # ========================================================================

    def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
        """
        Prepare batch for On-Policy Distillation training.

        Implements the abstract _prepare_batch method with OPD-specific processing:
        1. Log rollout metrics
        2. Remove redundant padding
        3. Get teacher log probabilities
        4. Add distillation config to metadata
        5. Create dummy advantages/returns for compatibility
        6. Compute and log metrics

        Args:
            batch: Batch of trajectory data from rollout

        Returns:
            Tuple of (prepared_batch, metrics_dict)
        """
        logger.info("[OPD] Begin distillation batch preparation")

        # Step 0: Log rollout metrics
        self._log_rollout_metrics(batch)

        def pre_process_batch(batch: Batch) -> Batch:
            batch.metadata["global_token_num"] = torch.sum(
                batch.values["attention_mask"]
                * batch.values.get("loss_mask", batch.values["attention_mask"]),
                dim=-1,
            ).tolist()

            if "loss_mask" in batch.values:
                batch.values["scoring_attention_mask"] = (
                    batch.values["attention_mask"] * batch.values["loss_mask"]
                )
            else:
                batch.values["scoring_attention_mask"] = batch.values["attention_mask"].clone()

            # Remove left and right padding
            batch = Batch.remove_redundant_left_padding(
                batch,
                pad_token_id=self._pad_token_id,
                anchor_field="input_ids",
                fields=[
                    "input_ids",
                    "prompts",
                    "attention_mask",
                    "position_ids",
                    "scoring_attention_mask",
                    "loss_mask",
                ],
            )

            batch = Batch.remove_redundant_right_padding(
                batch,
                pad_token_id=self._pad_token_id,
                anchor_field="input_ids",
                fields=[
                    "input_ids",
                    "responses",
                    "attention_mask",
                    "position_ids",
                    "scoring_attention_mask",
                    "loss_mask",
                ],
            )
            return batch

        # Step 1: Pre-process batch
        batch = batch.pad_to_world_size(world_size=self.world_size)  # pad to student world size
        batch = pre_process_batch(batch)

        # Step 2: Get teacher log probabilities for student trajectories
        logger.info("[OPD] Fetching teacher log probabilities...")
        student_batch_size = batch.metadata["batch_size"]
        # Build teacher batch from the (already padded) student batch to guarantee alignment,
        # then pad extra samples only for teacher world-size divisibility.
        teacher_batch = copy.deepcopy(batch).pad_to_world_size(world_size=self._teacher_world_size)
        teacher_batch = pre_process_batch(teacher_batch)
        teacher_batch_size = teacher_batch.metadata["batch_size"]
        teacher_log_probs = self._get_teacher_log_probs(teacher_batch)
        if teacher_batch_size != student_batch_size:
            logger.info(
                f"[OPD] Teacher batch padded from {student_batch_size} to {teacher_batch_size}, "
                f"teacher_log_probs shape(before slicing): {teacher_log_probs.shape}"
            )
            teacher_log_probs = teacher_log_probs[:student_batch_size]

        logger.info(f"[OPD] teacher_log_probs shape: {teacher_log_probs.shape}")

        # Step 3: Add teacher log probs to batch
        batch.values["teacher_log_probs"] = teacher_log_probs

        # Step 4: Add distillation config to metadata
        batch.metadata["distillation_coeff"] = self._distillation_coeff
        batch.metadata["entropy_coeff"] = self._entropy_coeff
        batch.metadata["temperature"] = self._temperature
        batch.metadata["loss_agg_mode"] = self._loss_agg_mode
        batch.metadata["distillation_epochs"] = self._distillation_epochs

        # Step 5: Create dummy advantages/returns for compatibility with base metrics
        response_length = batch.values["responses"].shape[-1]
        batch_size = batch.metadata["batch_size"]

        # For distillation, we don't use RL rewards, but we need these for metric computation
        batch.values["token_level_scores"] = torch.zeros(
            batch_size, response_length, dtype=torch.float32
        )
        batch.values["token_level_rewards"] = torch.zeros(
            batch_size, response_length, dtype=torch.float32
        )
        batch.values["advantages"] = torch.zeros(batch_size, response_length, dtype=torch.float32)
        batch.values["returns"] = torch.zeros(batch_size, response_length, dtype=torch.float32)

        # Step 6: Optionally compute old student log probs (for tracking)
        if self._do_old_student_log_prob_compute:
            old_log_probs = self._compute_old_log_probs(batch)
            batch.values["old_student_log_probs"] = old_log_probs
        else:
            bsz = batch.metadata["batch_size"]
            old_log_probs = torch.tensor([[0.0]] * bsz, dtype=torch.float32)
            batch.values["old_student_log_probs"] = old_log_probs

        # Step 7: Set update function for distillation training
        batch.metadata["update_fn"] = "update_actor_with_distillation"

        # Step 8: Compute metrics
        metrics = self._compute_distillation_metrics(batch)

        # Log metrics
        self._activity_tracker.experiment_logger_post(backend="wandb", data=metrics)

        logger.info("[OPD] Batch preparation completed")

        return batch, metrics

    def _get_teacher_log_probs(self, batch: Batch) -> torch.Tensor:
        """
        Get teacher model LOG PROBABILITIES for student trajectories.

        This is a memory-efficient version that uses compute_log_prob instead of compute_logits.

        Args:
            batch: Batch containing student trajectories

        Returns:
            torch.Tensor: Teacher log probabilities of shape [batch_size, response_length]
        """
        # Create teacher client on first use (in case not initialized in initialize_workers)
        if self._teacher_client is None:
            self._teacher_client = create_train_service_client(
                backend=self._teacher_backend,
                url=self._teacher_url,
                identifier=self._teacher_identifier,
            )
            logger.info(
                f"[OPD] Created teacher client with URL: {self._teacher_url}, "
                f"identifier: {self._teacher_identifier}"
            )

        # Ensure teacher is initialized
        if not self._teacher_initialized:
            self._initialize_teacher_workers()

        # Trim batch to only include fields required by the backend's compute_log_prob
        _BACKEND_COMPUTE_LOG_PROB_KEYS = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "scoring_attention_mask",
        ]
        teacher_batch = batch.trim_for_backend(_BACKEND_COMPUTE_LOG_PROB_KEYS).to_nextrainer_batch()

        # Get teacher LOG PROBABILITIES using compute_log_prob API
        with self._teacher_client.actor_context():
            result = self._teacher_client.compute_log_prob(teacher_batch)

        teacher_log_probs = result["batch"]["old_log_probs"]  # [batch_size, response_length]

        return teacher_log_probs

    def _compute_old_log_probs(self, batch: Batch) -> torch.Tensor:
        """
        Recompute old log probs for student model (for tracking purposes).

        Args:
            batch: Batch to compute log probs for

        Returns:
            Old log probabilities tensor
        """
        _BACKEND_COMPUTE_LOG_PROB_KEYS = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "scoring_attention_mask",
        ]
        trimmed = batch.trim_for_backend(_BACKEND_COMPUTE_LOG_PROB_KEYS)
        with self._train_service_client.actor_context():
            ret = self._train_service_client.compute_log_prob(trimmed.to_nextrainer_batch())
        old_log_probs: torch.Tensor = ret["batch"]["old_log_probs"]
        return old_log_probs

    # ========================================================================
    # Metrics and Logging Methods
    # ========================================================================

    def _log_rollout_metrics(self, batch: Batch) -> None:
        """
        Log rollout metrics from the batch to wandb.

        Args:
            batch: Batch containing rollout data
        """
        metrics = {}

        # Log basic stats
        if "reward" in batch.values:
            rewards = batch.values["reward"]
            if isinstance(rewards, torch.Tensor):
                metrics["rollout/reward_mean"] = rewards.mean().item()
                metrics["rollout/reward_std"] = rewards.std().item()
                metrics["rollout/reward_max"] = rewards.max().item()
                metrics["rollout/reward_min"] = rewards.min().item()
            elif isinstance(rewards, list) and len(rewards) > 0:
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                metrics["rollout/reward_mean"] = rewards_tensor.mean().item()
                metrics["rollout/reward_std"] = rewards_tensor.std().item()
                metrics["rollout/reward_max"] = rewards_tensor.max().item()
                metrics["rollout/reward_min"] = rewards_tensor.min().item()

        # Log sequence lengths
        if "attention_mask" in batch.values:
            seq_lens = batch.values["attention_mask"].sum(dim=1)
            metrics["rollout/seq_length_mean"] = seq_lens.float().mean().item()
            metrics["rollout/seq_length_max"] = seq_lens.max().item()
            metrics["rollout/seq_length_min"] = seq_lens.min().item()

        # Log score dict metrics if available
        if "score" in batch.values:
            scores = batch.values["score"]
            if scores and len(scores) > 0:
                # Collect all keys from score dicts
                all_keys: set[str] = set()
                for score_dict in scores:
                    if isinstance(score_dict, dict):
                        all_keys.update(score_dict.keys())

                # Compute mean for each key
                for key in all_keys:
                    values = []
                    for score_dict in scores:
                        if isinstance(score_dict, dict) and key in score_dict:
                            val = score_dict[key]
                            if isinstance(val, bool):
                                val = float(val)
                            elif isinstance(val, (int, float)):
                                val = float(val)
                            else:
                                continue
                            values.append(val)

                    if values:
                        metrics[f"rollout/{key}"] = np.mean(values)

        if metrics:
            self._activity_tracker.experiment_logger_post(backend="wandb", data=metrics)

    def _compute_distillation_metrics(self, batch: Batch) -> Dict:
        """
        Compute distillation-specific metrics.

        Args:
            batch: Batch containing all computed values

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic batch info
        metrics["distill/batch_size"] = batch.metadata["batch_size"]
        metrics["distill/seq_length"] = batch.values["input_ids"].shape[1]

        # Response length stats
        if "attention_mask" in batch.values:
            seq_lens = batch.values["attention_mask"].sum(dim=1)
            metrics["distill/seq_length_mean"] = seq_lens.float().mean().item()
            metrics["distill/seq_length_max"] = seq_lens.max().item()
            metrics["distill/seq_length_min"] = seq_lens.min().item()

        # Teacher log probs stats
        if "teacher_log_probs" in batch.values:
            teacher_log_probs = batch.values["teacher_log_probs"]
            if isinstance(teacher_log_probs, torch.Tensor):
                # Mask out padding
                response_length = batch.values["responses"].shape[-1]
                if "scoring_attention_mask" in batch.values:
                    mask = batch.values["scoring_attention_mask"][:, -response_length:]
                else:
                    mask = batch.values["attention_mask"][:, -response_length:]

                # Handle shape mismatch - teacher_log_probs may be [batch, seq] or [batch, response]
                if teacher_log_probs.shape[-1] == response_length:
                    valid_log_probs = teacher_log_probs * mask
                    valid_count = mask.sum()
                    if valid_count > 0:
                        metrics["distill/teacher_log_prob_mean"] = (
                            valid_log_probs.sum() / valid_count
                        ).item()
                        metrics["distill/teacher_log_prob_min"] = (
                            teacher_log_probs[mask.bool()].min().item()
                        )
                        metrics["distill/teacher_log_prob_max"] = (
                            teacher_log_probs[mask.bool()].max().item()
                        )

        # Prompt and response length breakdown based on loss_mask
        attention_mask = batch.values["attention_mask"]
        loss_mask = batch.values.get("loss_mask", attention_mask)
        actual_tokens = attention_mask.sum(-1).float()
        response_length_values = (attention_mask * loss_mask).sum(-1).float()
        prompt_length = actual_tokens - response_length_values

        metrics["distill/prompt_length_mean"] = prompt_length.mean().item()
        metrics["distill/prompt_length_max"] = prompt_length.max().item()
        metrics["distill/prompt_length_min"] = prompt_length.min().item()
        metrics["distill/response_length_mean"] = response_length_values.mean().item()
        metrics["distill/response_length_max"] = response_length_values.max().item()
        metrics["distill/response_length_min"] = response_length_values.min().item()

        return metrics

    @staticmethod
    def _compute_response_info(batch: Batch) -> dict:
        """
        Compute response length and mask information.

        Uses loss_mask and attention_mask to compute *actual* prompt/response
        lengths rather than relying on the tensor split position.

        Args:
            batch: Batch containing attention masks and loss_mask

        Returns:
            Dictionary with response_mask, prompt_length, and response_length
        """
        attention_mask = batch.values["attention_mask"]
        loss_mask = batch.values.get("loss_mask", attention_mask)

        actual_tokens = attention_mask.sum(-1).float()
        response_length = (attention_mask * loss_mask).sum(-1).float()
        prompt_length = actual_tokens - response_length

        max_response_length = batch.values["responses"].shape[-1]
        if "scoring_attention_mask" in batch.values:
            response_mask = batch.values["scoring_attention_mask"][:, -max_response_length:]
        else:
            response_mask = attention_mask[:, -max_response_length:]

        return {
            "response_mask": response_mask,
            "prompt_length": prompt_length,
            "response_length": response_length,
        }

    # ========================================================================
    # Override train() to use update_actor_with_distillation
    # ========================================================================

    def train(self, trajectories: list) -> dict:
        """
        Train on a list of trajectories using on-policy distillation.

        This overrides the base train() method to use update_actor_with_distillation
        instead of update_actor.

        1. Process trajectories (add padding and tensor fields)
        2. Convert trajectories to batch
        3. Prepare batch (OPD-specific: get teacher log probs)
        4. Execute training step with distillation
        5. Return metrics

        Args:
            trajectories: List of trajectories to train on

        Returns:
            Dictionary of training metrics
        """
        logger.info("SelfHostedOpdTrainer begin")

        # Step 1: Process trajectories to add padding and tensor fields
        trajectories = self._process_trajectories(trajectories)

        # Step 2: Convert trajectories to batch
        # identifier serves as model_tag for batch tracking
        batch = Batch.from_trajectories(trajectories, model_tag=self._identifier)

        # Step 3: Prepare batch
        # (OPD-specific: get teacher log probs, also having padding logic)
        logger.info("Trainer begin batch preparation!")
        preparation_start = time.time()
        batch, preparation_metrics = self._prepare_batch(batch)
        preparation_time = time.time() - preparation_start
        logger.info("Trainer batch preparation completed!")

        # Step 4: Execute training step with distillation
        batch_start_time = time.time()
        batch_size = len(batch)

        with self._train_service_client.actor_context():
            logger.info(f"[OPD] Begin training batch with size: {batch_size} in actor context")

            # Update actor with distillation (key difference from base class)
            update_actor_start_time = time.time()

            # Trim batch to only include fields required by the backend to reduce
            # network traffic and GPU memory.
            _BACKEND_DISTILLATION_KEYS = [
                "input_ids",
                "attention_mask",
                "position_ids",
                "responses",
                "teacher_log_probs",
                "old_student_log_probs",
            ]
            backend_batch = batch.trim_for_backend(_BACKEND_DISTILLATION_KEYS)
            actor_output = self._train_service_client.update_actor_with_distillation(
                backend_batch.to_nextrainer_batch()
            )
            update_actor_time = time.time() - update_actor_start_time
            train_metrics = self._reduce_metrics(actor_output["meta_info"]["metrics"])

            self._training_stats["batches_processed"] += 1
            self._training_stats["total_samples"] += batch_size

            # Save checkpoint for weight sync
            save_checkpoint_start_time = time.time()
            self._train_service_client.save_checkpoint(
                local_path=self._config.sync_weight_path,
                global_step=0,
                saved_fully_shared_ckpt=False,
                save_weight_only=True,
                remove_previous_ckpt=False,
            )
            save_checkpoint_time = time.time() - save_checkpoint_start_time

            # Store timing data for metrics
            train_metrics["timing/training/update_actor_distillation"] = update_actor_time
            train_metrics["timing/training/save_checkpoint"] = save_checkpoint_time

            # Periodic full checkpoint save
            save_freq = self._config.save_freq
            if save_freq > 0 and self._train_step % save_freq == 0:
                logger.info(f"[OPD] Saving checkpoint for step {self._train_step}...")
                local_global_step_folder = os.path.join(
                    self._config.checkpoint_path, f"global_step_{self._train_step}"
                )
                self._train_service_client.save_checkpoint(
                    local_path=local_global_step_folder,
                    global_step=self._train_step,
                    saved_fully_shared_ckpt=True,
                    save_weight_only=False,
                    remove_previous_ckpt=self._config.remove_previous_ckpt,
                )
                logger.info(f"[OPD] Checkpoint save completed for step {self._train_step}")

        # Calculate timing metrics outside context
        batch_end_time = time.time()
        training_step_time = batch_end_time - batch_start_time

        # Calculate step timing (time between consecutive batch completions)
        step_time = None
        if self._last_batch_finish_time is not None:
            step_time = batch_end_time - self._last_batch_finish_time
        self._last_batch_finish_time = batch_end_time

        # Update metrics with all timing information
        train_metrics.update(preparation_metrics)
        train_metrics.update(
            {
                "timing/batch_processing": preparation_time,
                "timing/training/training_step": training_step_time,
                "training/global_step": self._train_step,
                "training/batch_size": batch_size,
            }
        )

        # Add step timing if available (not available for first batch)
        if step_time is not None:
            train_metrics["timing/step"] = step_time

        logger.info("[OPD] Trainer training step completed!")

        return train_metrics
