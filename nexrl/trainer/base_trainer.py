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
BaseTrainer - Abstract base class for trainers
"""

import logging
import threading
import time
from abc import abstractmethod
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from ..base_module import NexRLModule
from ..executor import execute
from ..nexrl_types import Trajectory

if TYPE_CHECKING:
    from ..trajectory_pool import TrajectoryPool
    from ..weight_sync.weight_sync_controller import WeightSyncController

logger = logging.getLogger(__name__)


class BaseTrainer(NexRLModule):
    """
    Abstract base class for trainers.

    Derived classes should implement the `train()` method which takes a list of
    trajectories and returns training metrics.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the base trainer

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self._config = config
        self._stop_event: threading.Event = threading.Event()
        self._stop_event.clear()
        self._thread: threading.Thread | None = None

        # Training state
        self._train_step: int = 0
        self._total_train_steps: int = config.total_train_steps

        # Module references (set via set_module_references)
        self._trajectory_pool: "TrajectoryPool" = None  # type: ignore
        self._weight_sync_controller: "WeightSyncController" = None  # type: ignore

        self._model_tag = config.get(
            "model_tag",
            config.get("train_service", {}).get("model_tag", "default"),
        )

        # Timing tracking
        self._batch_count: int = 0

    def set_module_references(
        self,
        trajectory_pool: "TrajectoryPool",
        weight_sync_controller: "WeightSyncController",
    ) -> None:
        """
        Set the module references for the trainer.

        Args:
            trajectory_pool: Reference to the trajectory pool
            weight_sync_controller: Reference to the weight sync controller
        """
        self._trajectory_pool = trajectory_pool
        self._weight_sync_controller = weight_sync_controller

    def initialize_workers(self) -> None:
        """
        Initialize backend workers. Override in derived classes if needed.
        """

    def run(self) -> None:
        """
        Start the training loop in a background thread.
        """
        assert self._trajectory_pool is not None, "TrajectoryPool is not set"
        assert self._weight_sync_controller is not None, "WeightSyncController is not set"

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._main_loop)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop the trainer. Override in derived classes for additional cleanup.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        logger.info("Trainer stopped")

    def _main_loop(self):
        """
        Main training loop:
        1. Get trajectories from trajectory pool
        2. Call train() to process and train
        3. Log metrics and notify weight sync controller
        """
        while not self._stop_event.is_set():
            # Get trajectories from trajectory pool
            trajectories = self._get_trajectories()
            if trajectories is None:
                time.sleep(0.1)
                continue

            logger.info("Trainer get trajectories successfully!")

            start_time = time.time()

            with self._activity_tracker.track(self._module_name, "training_step"):
                # Call the train method implemented by derived class
                metrics = self.train(trajectories)

            total_time = time.time() - start_time

            # Add timing metrics
            metrics["timing/training_time"] = total_time

            self._activity_tracker.experiment_logger_post(
                backend="wandb",
                data=metrics,
                step=self._train_step,
            )

            # Increment train step
            self._train_step += 1

            # Update activity tracker with current training step
            self._activity_tracker.set_training_step(self._train_step)

            # Notify weight sync controller of training completion
            execute(
                self._weight_sync_controller.train_worker_notify_weight_update,
                worker_name=self._module_name,
                model_tag=self._model_tag,
            )

            self._batch_count += 1

            # Check if training is complete
            if self._train_step >= self._total_train_steps:
                self._stop_event.set()
                break

    def _get_trajectories(self) -> list[Trajectory] | None:
        """
        Get trajectories from the trajectory pool.

        Returns:
            List of trajectories or None if not available
        """
        return execute(self._trajectory_pool.get_trajectories)

    @abstractmethod
    def train(self, trajectories: list[Trajectory]) -> dict:
        """
        Train on a list of trajectories.

        This is the core method that derived classes must implement.
        It should:
        1. Convert trajectories to batch format if needed
        2. Process the batch (e.g., compute advantages)
        3. Train on the processed batch
        4. Update self._train_step
        5. Return training metrics

        Args:
            trajectories: List of trajectories to train on

        Returns:
            Dictionary of training metrics
        """

    # ==================== Public Interface ====================

    def get_train_step(self) -> int:
        """Get the current train step"""
        return self._train_step

    def set_train_step(self, step: int) -> None:
        """
        Set the current train step (used for resuming from checkpoint)

        Args:
            step: The training step to set
        """
        self._train_step = step
        logger.info(f"Train step set to {step}")
