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
Train Worker for NexRL framework
"""

import logging
import threading
import time

import numpy as np
from omegaconf import DictConfig

from .activity_tracker import ActivityTrackerProxy
from .base_module import NexRLModule
from .executor import execute
from .nexrl_types import Batch
from .train_batch_pool import TrainBatchPool
from .utils.init_utils import create_train_service_client
from .weight_sync.weight_sync_controller import WeightSyncController

logger = logging.getLogger(__name__)


class TrainWorker(NexRLModule):
    """
    Train Worker - Handles training of models using batches from TrainBatchPool
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the train worker

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self._config = config
        self._stop_event: threading.Event = threading.Event()
        self._train_step: int = 0
        self._total_train_steps: int = config.total_train_steps
        self._train_batch_pool: TrainBatchPool = None  # type: ignore  #  Set via set_module_references(), will be asserted in run()

        self._weight_sync_controller: WeightSyncController = None  # type: ignore  # Set via set_module_references(), will be asserted in run()
        self._model_tag = "default"  # TODO: should be configurable

        self._activity_tracker: ActivityTrackerProxy = None  # type: ignore  # Set via set_activity_tracker(), will be asserted in run()

        self._train_service_client = create_train_service_client(
            config.train_service.backend,
            config.train_service.url,
            config.train_service.get("identifier", None),
        )

        # Training statistics
        self.training_stats = {
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

        logger.info("TrainWorker initialized")

    def set_module_references(
        self, train_batch_pool: TrainBatchPool, weight_sync_controller: WeightSyncController
    ):
        """
        Set the module references for the train worker.

        Args:
            _train_batch_pool: Reference to the train batch pool
        """
        self._train_batch_pool = train_batch_pool
        self._weight_sync_controller = weight_sync_controller

    def initialize_workers(self) -> None:
        """
        Initialize workers with the final, composed configuration.
        This is called AFTER all config overrides are applied.

        For NexTrainer backend, this sends the actor config to workers via the API.
        """
        backend = self._config.train_service.backend

        if backend == "nextrainer":
            logger.info("Initializing NexTrainer workers with final config...")
            try:
                # Extract actor config and convert to dictionary
                from omegaconf import OmegaConf

                train_service_config = self._config.train_service
                config_dict = OmegaConf.to_container(train_service_config, resolve=True)

                # Initialize workers with config
                result = self._train_service_client.initialize_worker(
                    config_dict=config_dict, role="actor"
                )
                logger.info(f"Workers initialized: {result}")

                # Initialize model (load to GPU)
                logger.info("Initializing model on workers...")
                result = self._train_service_client.init_model()
                logger.info(f"Model initialized: {result}")

            except Exception as e:
                logger.error(f"Failed to initialize workers: {e}")
                raise RuntimeError(f"Worker initialization failed: {e}")
        elif backend == "mock":
            logger.info("Mock backend - skipping worker initialization")
        else:
            logger.warning(f"Unknown backend {backend} - skipping worker initialization")

    def run(self) -> None:
        """
        It contains a while loop. Each loop will call `_get_batch` and call `_fit` to train the batch.
        """
        assert self._train_batch_pool is not None, "TrainBatchPool is not set"
        assert self._weight_sync_controller is not None, "WeightSyncController is not set"

        self._stop_event.clear()
        self._thread: threading.Thread = threading.Thread(target=self._main_loop)
        self._thread.start()

    def stop(self) -> None:
        """
        Set the running to False and end the execution.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

        # Wait for any ongoing checkpoint save to complete
        if self._save_thread is not None and self._save_thread.is_alive():
            logger.info("Waiting for checkpoint save to complete...")
            self._save_thread.join()

        logger.info("TrainWorker stopped")

    def _main_loop(self):
        """
        Main training loop
        """
        while not self._stop_event.is_set():
            # Waiting for a batch; not actively processing
            # Get batch from train batch pool
            batch = self._get_batch()
            if batch is None:
                # No batch available, wait a bit before trying again
                time.sleep(0.1)
                continue

            # Train the batch
            with self._activity_tracker.track(self.__class__.__name__, "batch"):
                self._step(batch)

            self._train_step += 1

            # Update activity tracker with current training step
            self._activity_tracker.set_training_step(self._train_step)

            # Notify weight manager of training completion
            execute(
                self._weight_sync_controller.train_worker_notify_weight_update,
                worker_name=self._module_name,
                model_tag=self._model_tag,
            )

            if self._train_step >= self._total_train_steps:
                self._stop_event.set()
                break

    def _get_batch(self) -> Batch | None:
        """
        Get a batch from _train_batch_pool

        Returns:
            Batch: Training batch or None if not available
        """
        assert self._train_batch_pool is not None, "TrainBatchPool must be set before getting batch"
        # TODO: Get model tag from config or parameter
        return execute(self._train_batch_pool.get_batch, self._model_tag)

    def get_train_step(self) -> int:
        """
        Get the current train step
        """
        return self._train_step

    def set_train_step(self, step: int) -> None:
        """
        Set the current train step (used for resuming from checkpoint)

        Args:
            step: The training step to set
        """
        self._train_step = step
        logger.info(f"Train step set to {step}")

    def _step(self, batch: Batch):
        """
        Train the batch

        Args:
            batch: Batch to train
        """
        batch_start_time = time.time()

        # Log batch information
        batch_size = len(batch)

        with self._train_service_client.actor_context():

            logger.info(f"Begin training batch with size: {batch_size} in actor context")

            update_actor_start_time = time.time()
            actor_output = self._train_service_client.update_actor(batch.to_nextrainer_batch())
            update_actor_end_time = time.time()
            update_actor_time = update_actor_end_time - update_actor_start_time

            metrics = TrainWorker.reduce_metrics(actor_output["meta_info"]["metrics"])

            self.training_stats["batches_processed"] += 1
            self.training_stats["total_samples"] += batch_size

            # Time the checkpoint save (sync weight buffer)
            save_checkpoint_start_time = time.time()

            self._train_service_client.save_checkpoint(
                local_path=self._config.sync_weight_path,
                global_step=0,
                saved_fully_shared_ckpt=False,
                save_weight_only=True,
                remove_previous_ckpt=False,
            )

            save_checkpoint_end_time = time.time()
            save_checkpoint_time = save_checkpoint_end_time - save_checkpoint_start_time

            # Store timing data for metrics
            metrics["timing/training/update_actor"] = update_actor_time
            metrics["timing/training/save_checkpoint"] = save_checkpoint_time

            if self._config.save_freq > 0 and self._train_step % self._config.save_freq == 0:
                # Run checkpoint saving in background to not block training
                logger.info(f"Saving checkpoint for step {self._train_step}...")
                import os

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
        metrics.update(
            {
                "timing/training/training_step": training_step_time,
                "training/global_step": self._train_step,
                "training/batch_size": batch_size,
            }
        )

        # Add step timing if available (not available for first batch)
        if step_time is not None:
            metrics["timing/step"] = step_time

        # Log metrics outside context
        self._activity_tracker.experiment_logger_post(
            backend="wandb", data=metrics, step=self._train_step
        )

    @staticmethod
    def reduce_metrics(metrics: dict):
        for key, val in metrics.items():
            metrics[key] = np.mean(val)
        return metrics

    def _save_checkpoint_async(self):
        """
        Start checkpoint saving in a background thread
        """
        # If a previous save is still running, wait for it to complete first
        if self._save_thread is not None and self._save_thread.is_alive():
            logger.warning(
                "Previous checkpoint save still in progress, waiting for it to complete..."
            )
            self._save_thread.join()

        # Capture current train step for the checkpoint
        current_step = self._train_step

        # Start new background save thread
        self._save_thread = threading.Thread(
            target=self._save_checkpoint,
            name=f"checkpoint_save_{current_step}",
        )
        self._save_thread.daemon = True
        self._save_thread.start()
        logger.info(f"Started background checkpoint save for step {current_step}")

    def _save_checkpoint(self):
        """
        Save the checkpoint (runs in background thread)

        Args:
            global_step: The training step to associate with this checkpoint
        """

        logger.info(f"Saving checkpoint for step {self._train_step}...")
        import os

        local_global_step_folder = os.path.join(
            self._config.train_worker.checkpoint_path, f"global_step_{self._train_step}"
        )
        self._train_service_client.save_checkpoint(
            local_path=local_global_step_folder,
            global_step=self._train_step,
            saved_fully_shared_ckpt=True,
            save_weight_only=False,
            remove_previous_ckpt=self._config.remove_previous_ckpt,
        )

        logger.info(f"Checkpoint save completed for step {self._train_step}")
