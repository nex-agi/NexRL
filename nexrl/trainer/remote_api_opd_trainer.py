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
RemoteApiOpdTrainer - Remote API trainer with On-Policy Distillation

This trainer implements on-policy distillation (OPD) using remote API backends
(Weaver/Tinker). It creates and owns teacher sampling client(s), computes reverse
KL divergence between student and teacher models, and adjusts per-token advantages
before training with the standard importance_sampling loss.

Key features:
- Owns teacher client(s) - no service holder modifications needed
- Computes teacher logprobs in parallel via ThreadPoolExecutor
- Computes per-token KL advantages (not scalar like GRPO)
- Supports multiple teachers (one per dataset)
- Works with both Weaver and Tinker backends
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from omegaconf import DictConfig

from ..nexrl_types import Trajectory
from ..utils.config_utils import get_train_service_config_by_role
from ..utils.logging_utils import log_rollout_metrics
from .remote_api_trainer import RemoteApiTrainer
from .teacher_client import TeacherClient

logger = logging.getLogger(__name__)


def _discounted_future_sum(advantages: np.ndarray, discount: float) -> np.ndarray:
    """Apply discount factor to future advantages (for optional KL discounting)."""
    if discount == 0:
        return advantages

    result = np.zeros_like(advantages)
    future_sum = 0.0
    for t in reversed(range(len(advantages))):
        future_sum = advantages[t] + discount * future_sum
        result[t] = future_sum
    return result


class RemoteApiOpdTrainer(RemoteApiTrainer):
    """
    Remote API trainer with On-Policy Distillation (OPD) algorithm.

    Extends RemoteApiTrainer with OPD-specific trajectory preparation:
    - Creates and owns teacher client(s) from config
    - Computes teacher logprobs for student trajectories (parallel)
    - Computes per-token reverse KL: KL(student || teacher)
    - Stores per-token advantages in trajectory.extra_fields
    - Uses custom datum conversion for per-token advantages

    Works with any remote API backend (Tinker, Weaver, etc.).
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the Remote API OPD trainer.

        Args:
            config: Configuration dictionary containing:
                - train_service.config: Training hyperparameters (lr, kl_penalty_coef, etc.)
                - trainer.teachers: List of teacher configs
        """
        super().__init__(config)

        train_service = config.get("train_service")
        if not train_service:
            raise ValueError("train_service must be specified")
        actor_train_service_config = get_train_service_config_by_role(train_service, "actor")

        # OPD hyperparameters from config
        train_config = actor_train_service_config.get("config", {})
        self._kl_penalty_coef = train_config.get("kl_penalty_coef", 1.0)
        self._kl_discount_factor = train_config.get("kl_discount_factor", 0.0)

        # Teacher configurations (list of dicts)
        # config is already the trainer sub-config (self._config.trainer), so read teachers directly
        self._teacher_configs = config.get("teachers", [])
        if not self._teacher_configs:
            raise ValueError("teachers must be specified")

        # Teacher clients will be created in set_service_holder
        self._teacher_clients: list[TeacherClient] = []

        # Executor for parallel teacher logprob computation
        self._executor = ThreadPoolExecutor(max_workers=32)

        logger.info("RemoteApiOpdTrainer initialized with:")
        logger.info(f"  kl_penalty_coef={self._kl_penalty_coef}")
        logger.info(f"  kl_discount_factor={self._kl_discount_factor}")
        logger.info(f"  num_teachers={len(self._teacher_configs)}")

    def set_service_holder(self, service_holder) -> None:
        """
        Override: also create teacher clients from config.

        Args:
            service_holder: Service holder instance (TinkerServiceHolder, WeaverServiceHolder, etc.)
        """
        super().set_service_holder(service_holder)

        # Create teacher clients
        logger.info(f"Creating {len(self._teacher_configs)} teacher client(s)...")
        for i, teacher_cfg in enumerate(self._teacher_configs):
            try:
                client = TeacherClient.from_config(teacher_cfg, service_holder)
                self._teacher_clients.append(client)
                logger.info(
                    f"Teacher {i+1}/{len(self._teacher_configs)} created: "
                    f"{teacher_cfg.get('base_model')}"
                )
            except Exception as e:
                logger.error(f"Failed to create teacher client {i}: {e}")
                raise RuntimeError(f"Teacher client creation failed: {e}") from e

        logger.info(f"All {len(self._teacher_clients)} teacher client(s) created successfully")

    def _compute_teacher_logprobs_parallel(
        self, trajectories: list[Trajectory]
    ) -> list[list[float | None]]:
        """
        Compute teacher logprobs for all trajectories in parallel.

        Args:
            trajectories: List of trajectories with tokens

        Returns:
            List of teacher logprob lists, one per trajectory
        """
        if not self._teacher_clients:
            logger.warning("No teacher clients available, returning empty logprobs")
            return [[] for _ in trajectories]

        # For now, use first teacher for all trajectories (single-teacher case)
        teacher_client = self._teacher_clients[0]

        # Submit all trajectories for parallel processing
        futures = {}
        for i, traj in enumerate(trajectories):
            # Full sequence = all tokens (prompt + response)
            full_tokens = traj.tokens
            future = self._executor.submit(teacher_client.compute_logprobs, full_tokens)
            futures[future] = i

        # Collect results in original order
        results: list[list[float | None]] = [[] for _ in trajectories]
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Teacher logprob computation failed for trajectory {idx}: {e}")
                # Return empty logprobs for this trajectory
                results[idx] = []

        return results

    def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """
        Prepare trajectories with OPD: compute teacher logprobs and per-token KL advantages.

        Steps:
        1. Compute teacher logprobs for all trajectories (parallel)
        2. Compute per-token reverse KL: student_logprobs - teacher_logprobs
        3. Compute per-token KL advantages: -kl_coef * reverse_kl * loss_mask
        4. Optionally apply discount factor
        5. Store in trajectory.extra_fields["token_advantages"]

        Args:
            trajectories: List of Trajectory dataclasses with tokens, loss_mask, and logprobs
            metrics: Dictionary to store OPD metrics

        Returns:
            Trajectories with per-token advantages in extra_fields
        """
        logger.info(f"[OPD] Preparing {len(trajectories)} trajectories with teacher logprobs")

        if self._kl_penalty_coef == 0:
            logger.warning(
                "kl_penalty_coef is 0, skipping teacher logprob computation. "
                "This is equivalent to no distillation."
            )
            # Store zero advantages for compatibility
            for traj in trajectories:
                response_length = sum(traj.loss_mask)
                traj.extra_fields["token_advantages"] = [0.0] * response_length
            return trajectories

        # Step 1: Compute teacher logprobs in parallel
        teacher_logprobs_batch = self._compute_teacher_logprobs_parallel(trajectories)

        # Step 2-4: Compute per-token KL advantages for each trajectory
        reverse_kl_sums = []
        valid_token_counts = []

        for traj, teacher_logprobs in zip(trajectories, teacher_logprobs_batch):
            # Get student logprobs (stored during rollout)
            # Note: student_logprobs might be list of tensors or list of floats
            student_logprobs = traj.get("logprobs", [])
            loss_mask = traj.loss_mask

            if not teacher_logprobs or not student_logprobs:
                logger.warning("Missing logprobs, setting zero advantages")
                traj.extra_fields["token_advantages"] = [0.0] * sum(loss_mask)
                continue

            # Convert student logprobs to floats, handling tensors and Nones
            if hasattr(student_logprobs[0], "item"):
                student_logprobs = [
                    float(lp.item()) if lp is not None else 0.0 for lp in student_logprobs
                ]
            else:
                student_logprobs = [float(lp) if lp is not None else 0.0 for lp in student_logprobs]

            # Convert teacher logprobs to floats, handling Nones at any position
            teacher_logprobs = [float(lp) if lp is not None else 0.0 for lp in teacher_logprobs]

            # Align both logprob arrays to target positions (tokens[1:]).
            # Teacher logprobs are always full-length (position-aligned with tokens),
            # with first element = None (no context for first token), so skip it.
            # Student logprobs may be full-length OR already shifted to target positions.
            teacher_lp = teacher_logprobs[1:]
            if len(student_logprobs) == len(traj.tokens):
                student_lp = student_logprobs[1:]
            else:
                student_lp = student_logprobs

            min_len = min(len(student_lp), len(teacher_lp))

            # Compute reverse KL per token: log p_student - log p_teacher
            reverse_kl = np.array([student_lp[i] - teacher_lp[i] for i in range(min_len)])

            # Apply loss mask (only compute KL for response tokens)
            # loss_mask aligns with tokens; shift by 1 to align with target positions
            loss_mask_shifted = loss_mask[1 : min_len + 1]
            reverse_kl_masked = reverse_kl * np.array(loss_mask_shifted, dtype=float)

            # Compute KL advantages: -kl_coef * reverse_kl
            kl_advantages = -self._kl_penalty_coef * reverse_kl_masked

            # Optionally apply discount factor for future KL
            if self._kl_discount_factor > 0:
                kl_advantages = _discounted_future_sum(kl_advantages, self._kl_discount_factor)

            # Extract only response token advantages (where loss_mask=1)
            # This matches the format expected by convert_trajectories_to_datums
            response_advantages = [
                float(kl_advantages[i])
                for i in range(len(kl_advantages))
                if loss_mask_shifted[i] == 1
            ]

            # Store per-token advantages in extra_fields
            traj.extra_fields["token_advantages"] = response_advantages

            # Track metrics
            valid_tokens = np.sum(loss_mask_shifted)
            if valid_tokens > 0:
                reverse_kl_sums.append(float(np.sum(reverse_kl_masked)))
                valid_token_counts.append(int(valid_tokens))

        # Compute and log average reverse KL
        if valid_token_counts:
            total_kl = sum(reverse_kl_sums)
            total_tokens = sum(valid_token_counts)
            avg_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
            metrics["opd/teacher_kl"] = avg_kl
            metrics["opd/valid_tokens"] = total_tokens
            logger.info(f"[OPD] Average teacher KL: {avg_kl:.4f} over {total_tokens} tokens")
        else:
            logger.warning("[OPD] No valid tokens for KL computation")

        logger.info(f"[OPD] Trajectory preparation completed")
        return trajectories

    def train(self, trajectories: list[Trajectory]) -> dict:
        """
        Train on a list of trajectories using OPD.

        This overrides the base train() method to use OPD-specific datum conversion
        that handles per-token advantages.

        Steps:
        1. Log rollout metrics
        2. Prepare trajectories (compute teacher logprobs and KL advantages)
        3. Convert to service Datum format (using OPD-specific conversion)
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
        logger.info(f"[OPD] rollout metrics: {metrics}, trajectories: {len(trajectories)}")

        # Step 2: Prepare trajectories (compute teacher logprobs and KL advantages)
        trajectories = self._prepare_trajectories(trajectories, metrics)

        # Dump prepared trajectories if enabled (DataDumper)
        if self._data_dumper.should_dump("prepared_trajectories", self._train_step):
            # Extract advantages from extra_fields
            advantages = [t.extra_fields.get("token_advantages", []) for t in trajectories]
            self._data_dumper.dump_prepared_trajectories(
                step=self._train_step,
                trajectories=trajectories,
                advantages=advantages,
            )

        # Step 3: Convert to service Datum format
        from ..utils.finetune_service_utils import convert_trajectories_to_datums

        datums_data = convert_trajectories_to_datums(trajectories)

        # Dump datums if enabled (DataDumper)
        if self._data_dumper.should_dump("datums", self._train_step):
            self._data_dumper.dump_datums(
                step=self._train_step,
                datums_data=datums_data,
            )

        if len(datums_data) == 0:
            logger.warning("[OPD] No valid datums after filtering, skipping training step")
            metrics["training/skipped"] = 1
            return metrics

        # Step 4: Execute forward_backward and optim_step together
        from ..executor import execute

        loss_fn_config = {
            "entropy_coeff": self._entropy_coeff,
            "enable_debug_dump": self._enable_debug_dump,
        }

        training_metrics = execute(
            self._service_holder.forward_backward_and_optim_step,
            datums_data=datums_data,
            loss_fn=self._loss_fn,
            loss_fn_config=loss_fn_config,
            learning_rate=self._learning_rate,
            beta1=self._beta1,
            beta2=self._beta2,
            weight_decay=self._weight_decay,
            eps=self._eps,
            grad_clip_norm=self._grad_clip_norm,
        )

        # Dump training metrics if enabled (DataDumper)
        if self._data_dumper.should_dump("training_metrics", self._train_step):
            self._data_dumper.dump_training_metrics(
                step=self._train_step,
                metrics=training_metrics,
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
            f"[OPD] Training step {self._train_step} completed, "
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

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
