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
SelfHostedTrainer - Trainer implementation for self-hosted training backend
"""

import logging
import os
import threading
import time
from abc import abstractmethod

import numpy as np
import torch
from omegaconf import DictConfig

from ..inference_service_client import hf_tokenizer
from ..nexrl_types import Batch, Trajectory
from ..utils.config_utils import get_train_service_config_by_role
from ..utils.data_dumper import get_data_dumper
from ..utils.init_utils import create_train_service_client
from ..utils.torch_functional import compute_position_id_with_mask, padding_data
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SelfHostedTrainer(BaseTrainer):
    """
    Base self-hosted trainer implementation with extensible batch preparation.

    This trainer provides the core training loop for self-hosted backends:
    1. Converts trajectories to batch format
    2. Prepares batch (algorithm-specific, override _prepare_batch)
    3. Executes training step
    4. Handles checkpointing

    Subclasses should override _prepare_batch() to implement algorithm-specific logic.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the self-hosted trainer

        Args:
            config: Configuration dictionary with training settings
        """
        super().__init__(config)

        # Get the actor train service config
        train_service = config.get("train_service")
        if not train_service:
            raise ValueError("train_service must be specified")
        self._actor_train_service_config = get_train_service_config_by_role(train_service, "actor")

        # Train service client (using actor train service)
        self._train_service_client = create_train_service_client(
            self._actor_train_service_config.backend,
            self._actor_train_service_config.url,
            self._actor_train_service_config.get("identifier", None),
        )
        # self.world_size = self._actor_train_service_config.resource.get("world_size", None)
        # if self.world_size is None:
        #     raise ValueError("world_size must be specified in actor train_service.resource config")
        # gpus_per_pod = self._actor_train_service_config.resource.get("gpus_per_pod", -1)
        # self.world_size = int(self.world_size * gpus_per_pod)
        node_world_size = self._actor_train_service_config.resource.get("world_size", None)
        if node_world_size is None:
            raise ValueError("world_size must be specified in actor train_service.resource config")
        gpus_per_pod = self._actor_train_service_config.resource.get("gpus_per_pod", None)
        if gpus_per_pod is None:
            raise ValueError(
                "gpus_per_pod must be specified in actor train_service.resource config"
            )
        self.world_size = int(node_world_size * gpus_per_pod)
        logger.info(f"SelfHostedTrainer [actor] initialized with world size: {self.world_size}")

        # Initialize tokenizer for trajectory processing
        tokenizer_path = config.algorithm.inference_service.get(
            "tokenizer", config.algorithm.inference_service.model
        )
        self.tokenizer = hf_tokenizer(tokenizer_path)

        # Configuration for trajectory processing
        self._max_prompt_length = config.get("max_prompt_length", 4096)
        self._max_response_length = config.get("max_response_length", 2048)

        # Training statistics
        self._training_stats = {
            "batches_processed": 0,
            "total_samples": 0,
            "total_training_time": 0.0,
            "average_batch_size": 0.0,
            "last_batch_time": 0.0,
        }

        # Track last batch finish time for step timing
        self._last_batch_finish_time: float | None = None

        # Background checkpoint saving
        self._save_thread: threading.Thread | None = None

        # Initialize data dumper for debug data collection
        self._data_dumper = get_data_dumper(config, rank=0, key="trainer")

        logger.info("SelfHostedTrainer initialized")

    def initialize_workers(self) -> None:
        """
        Initialize the train service backend workers.
        This should be called before starting training.
        """
        backend = self._actor_train_service_config.backend

        if backend in ("nextrainer", "http"):
            logger.info("Initializing self-hosted workers with final config...")
            try:
                from omegaconf import OmegaConf

                # Use the actor train service config
                config_dict = OmegaConf.to_container(self._actor_train_service_config, resolve=True)

                # Merge debug config into actor config for DataDumper initialization
                # NOTE: self._config is a sub-config (config.trainer), so we need to check
                # the parent chain to find debug config at the root level.
                debug_config = self._config.get("debug", None)
                if debug_config is None:
                    parent = getattr(self._config, "_parent", None)
                    if parent is not None:
                        debug_config = parent.get("debug", None)

                if debug_config is not None and "actor" in config_dict:
                    config_dict["actor"]["debug"] = OmegaConf.to_container(
                        debug_config, resolve=True
                    )
                    logger.info(
                        f"[DEBUG] Added debug config to actor: {config_dict['actor'].get('debug')}"
                    )
                else:
                    logger.info(
                        f"[DEBUG] debug_config={debug_config}, 'actor' in config_dict={'actor' in config_dict}"
                    )

                # Initialize workers with config
                try:
                    result = self._train_service_client.initialize_worker(
                        config_dict=config_dict, role="actor"
                    )
                    logger.info(f"Workers initialized: {result}")
                except Exception as e:
                    logger.error(f"Failed to initialize workers: {e}")
                    experiment_path = os.environ.get("EXPERIMENT_PATH", "EXPERIMENT_PATH")
                    identifier = self._config.train_service.get("identifier", "default")
                    api_server_log = f"{experiment_path}/api_server.log"
                    workers_log = f"{experiment_path}/workers-{identifier}-rank*.log"
                    logger.error(
                        f"Please check **{api_server_log}** and **{workers_log}** for more detailed error information."
                    )
                    raise RuntimeError(
                        f"Worker initialization failed: {e}. "
                        f"Please check **{api_server_log}** and **{workers_log}** for more detailed error information."
                    ) from e

                # Initialize model (load to GPU)
                logger.info("Initializing model on workers...")
                result = self._train_service_client.init_model()
                logger.info(f"Model initialized: {result}")

            except Exception as e:
                logger.error(f"Failed to initialize workers: {e}")
                raise RuntimeError(f"Worker initialization failed: {e}") from e
        elif backend == "mock":
            logger.info("Mock backend - skipping worker initialization")
        else:
            logger.warning(f"Unknown backend {backend} - skipping worker initialization")

    def stop(self) -> None:
        """Stop the trainer with cleanup for checkpoint saving."""
        super().stop()

        # Wait for any ongoing checkpoint save to complete
        if self._save_thread is not None and self._save_thread.is_alive():
            logger.info("Waiting for checkpoint save to complete...")
            self._save_thread.join()

    @abstractmethod
    def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
        """
        Prepare batch for training (algorithm-specific processing).

        This method should be overridden by subclasses to implement
        algorithm-specific batch preparation (e.g., GRPO advantage computation).

        Args:
            batch: Batch of trajectory data from rollout

        Returns:
            Tuple of (prepared_batch, metrics_dict)
        """

    def train(self, trajectories: list[Trajectory]) -> dict:
        """
        Train on a list of trajectories.

        1. Process trajectories (add padding and tensor fields)
        2. Convert trajectories to batch
        3. Prepare batch (algorithm-specific, calls _prepare_batch)
        4. Execute training step
        5. Return metrics

        Args:
            trajectories: List of trajectories to train on

        Returns:
            Dictionary of training metrics
        """
        logger.info("SelfHostedTrainer begin")

        # Dump trajectories before processing (for debug)
        if self._data_dumper.should_dump("trajectory", self._train_step):
            self._data_dumper.dump_trajectory(self._train_step, trajectories)

        # Step 1: Process trajectories to add padding and tensor fields
        trajectories = self._process_trajectories(trajectories)

        # Step 2: Convert trajectories to batch
        # identifier serves as model_tag for batch tracking
        batch = Batch.from_trajectories(trajectories, model_tag=self._identifier)
        batch = batch.pad_to_world_size(world_size=self.world_size)

        # Step 3: Prepare batch (algorithm-specific)
        logger.info("Trainer begin batch preparation!")
        preparation_start = time.time()
        batch, preparation_metrics = self._prepare_batch(batch)
        preparation_time = time.time() - preparation_start
        logger.info("Trainer batch preparation completed!")

        # Step 4: Execute training step
        batch_start_time = time.time()
        batch_size = len(batch)

        # Propagate trainer-side global step to workers (used for easy_dump filenames, etc.)
        batch.metadata["global_step"] = self._train_step

        with self._train_service_client.actor_context():
            logger.info(f"Begin training batch with size: {batch_size} in actor context")

            # Update actor
            update_actor_start_time = time.time()
            actor_output = self._train_service_client.update_actor(batch.to_nextrainer_batch())
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
            train_metrics["timing/training/update_actor"] = update_actor_time
            train_metrics["timing/training/save_checkpoint"] = save_checkpoint_time

            # Periodic full checkpoint save
            save_freq = self._config.save_freq
            if save_freq > 0 and self._train_step % save_freq == 0:
                logger.info(f"Saving checkpoint for step {self._train_step}...")
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
                logger.info(f"Checkpoint save completed for step {self._train_step}")

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

        logger.info("Trainer training step completed!")

        return train_metrics

    def _process_trajectories(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """
        Process trajectories to add padding and tensor fields for NexTrainer.

        Converts tokens and loss_mask into padded tensors
        (input_ids, attention_mask, position_ids, loss_mask, prompts, responses).

        The loss_mask is used to determine which tokens are prompt vs response.
        We find the first contiguous zeros (before the first 1) as the prompt.

        Args:
            trajectories: List of Trajectory dataclasses with tokens and loss_mask

        Returns:
            Processed trajectories with added tensor fields
        """
        pad_token_id = self.tokenizer.pad_token_id

        # Calculate the maximum response length among all trajectories
        # This is needed because responses with tool-call can be longer than the configured max response length
        # Tool-call result length cannot be restricted
        max_response_len = 0
        max_prompt_len = 0
        for traj in trajectories:
            loss_mask_values = traj.loss_mask
            try:
                first_one_idx = loss_mask_values.index(1)
            except ValueError:
                # No 1s found, entire sequence is prompt, no response
                first_one_idx = len(loss_mask_values)

            prompt_length = first_one_idx
            response_length = len(traj.tokens) - first_one_idx
            max_prompt_len = max(max_prompt_len, prompt_length)
            max_response_len = max(max_response_len, response_length)

        # Use the larger of the actual max lengths or the configured maxes
        max_prompt_length = max(max_prompt_len, self._max_prompt_length)
        max_response_length = max(max_response_len, self._max_response_length)

        for traj in trajectories:
            tokens = traj.tokens
            loss_mask_values = traj.loss_mask

            # Extract prompt_tokens and response_tokens based on loss_mask
            # The loss mask can be like 0011000111 - we want the first zeros until a 1
            # Find the index of the first 1 in the loss mask
            try:
                first_one_idx = loss_mask_values.index(1)
            except ValueError:
                # No 1s found, entire sequence is prompt
                first_one_idx = len(loss_mask_values)

            prompt_tokens = tokens[:first_one_idx]
            response_tokens = tokens[first_one_idx:]
            response_loss_mask_values = loss_mask_values[first_one_idx:]

            # Pad prompt tokens and masks (left padding)
            prompt_input_ids = padding_data(
                prompt_tokens,
                max_length=max_prompt_length,
                pad_token_id=pad_token_id,
                left_pad=True,
                truncation="error",
            )
            prompt_attention_mask = padding_data(
                torch.ones((1, len(prompt_tokens)), dtype=torch.int),
                max_length=max_prompt_length,
                pad_token_id=0,
                left_pad=True,
                truncation="error",
            )
            prompt_loss_mask = torch.zeros_like(prompt_input_ids, dtype=torch.int)

            # Pad response tokens and masks (right padding)
            response_input_ids = padding_data(
                response_tokens,
                max_length=max_response_length,
                pad_token_id=pad_token_id,
                left_pad=False,
                truncation="error",
            )
            response_attention_mask = padding_data(
                torch.ones((1, len(response_tokens)), dtype=torch.int),
                max_length=max_response_length,
                pad_token_id=0,
                left_pad=False,
                truncation="error",
            )
            # Use the actual loss_mask values from the trajectory for response
            response_loss_mask = padding_data(
                torch.tensor([response_loss_mask_values], dtype=torch.int),
                max_length=max_response_length,
                pad_token_id=0,
                left_pad=False,
                truncation="error",
            )

            # Concatenate prompt and response
            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1)
            attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
            loss_mask = torch.cat([prompt_loss_mask, response_loss_mask], dim=1)

            # Compute position_ids from attention_mask
            position_ids = compute_position_id_with_mask(attention_mask)

            # Add processed fields to trajectory
            traj["input_ids"] = input_ids.squeeze(0)
            traj["attention_mask"] = attention_mask.squeeze(0)
            traj["position_ids"] = position_ids.squeeze(0)
            traj["loss_mask"] = loss_mask.squeeze(0)
            traj["prompts"] = prompt_input_ids.squeeze(0)
            traj["responses"] = response_input_ids.squeeze(0)

        return trajectories

    @staticmethod
    def _reduce_metrics(metrics: dict) -> dict:
        """Reduce metrics by computing mean of each value."""
        for key, val in metrics.items():
            if val is None:
                metrics[key] = None
            elif isinstance(val, list):
                # Filter out None values before computing mean
                filtered = [v for v in val if v is not None]
                metrics[key] = np.mean(filtered) if filtered else None
            else:
                metrics[key] = np.mean(val)
        return metrics
