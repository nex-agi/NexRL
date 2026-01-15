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
Base Rollout Worker for NexRL framework
"""

import copy
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig

from ..base_module import NexRLModule
from ..data_loader import BaseDataLoader
from ..executor import execute
from ..nexrl_types import Trajectory
from ..trajectory_pool import TrajectoryPool
from ..validator import Validator
from ..weight_sync.weight_sync_controller import WeightSyncController

logger = logging.getLogger(__name__)


class BaseRolloutWorker(NexRLModule, ABC):
    """
    Rollout worker is used to execute rollout tasks.
    BaseRolloutWorker is an abstract class for various RolloutWorker classes to inherit from.
    Since RolloutWorker is user-implemented code, to ensure clean and readable code,
    the worker does not interact with any other modules in NexRL, only with worker_manager.
    Functions are called through worker_manager.
    Note: The worker does not actively establish connection with worker_manager when created.
    The worker_manager is responsible for establishing connection with the worker.
    """

    def __init__(self, config: DictConfig):
        """
        All necessary initialization work.

        Note: Inference service client initialization is deferred to init_inference_service_client()
        which is called by the controller after all dependencies are set up.

        Args:
            config: Configuration file
        """
        super().__init__()
        self._config = config
        self._stop_event = threading.Event()
        # These will be set before run() is called, so they're never actually None during operation
        self._thread: threading.Thread = None  # type: ignore  # Set in run()
        # Inference service client will be initialized via init_inference_service_client() if needed
        self._inference_client = None  # type: ignore  # Will be set if need_llm_inference=True

        self._trajectory_pool: TrajectoryPool = None  # type: ignore  # Set via set_module_references()
        self._dataloader: BaseDataLoader = None  # type: ignore  # Set via set_module_references()
        self._weight_sync_controller: WeightSyncController = None  # type: ignore  # Set via set_module_references()
        self._validate_dataloader: BaseDataLoader = None  # type: ignore  # Set via set_module_references()
        self._validator: Validator = None  # type: ignore  # Set via set_module_references()

        self._next_task: dict[str, Any] | None = None

        self._is_running_validate: bool = False

    def set_module_references(
        self,
        trajectory_pool: TrajectoryPool,
        dataloader: BaseDataLoader,
        weight_sync_controller: WeightSyncController,
        validate_dataloader: BaseDataLoader,
        validator: Validator,
    ):
        """
        Set the module references for the worker.
        """
        self._trajectory_pool = trajectory_pool
        self._dataloader = dataloader
        self._weight_sync_controller = weight_sync_controller
        self._validate_dataloader = validate_dataloader
        self._validator = validator
        if self._inference_client is not None:
            self._inference_client.set_weight_sync_controller(weight_sync_controller)

    def init_inference_service_client(self, service_holder=None):
        """
        Initialize inference service client for all backends.
        Called by controller after dependencies are set up.

        Args:
            service_holder: Shared backend-specific service holder (Tinker/Weaver)
        """
        # Check if this worker needs LLM inference
        need_llm_inference = self._config.get("need_llm_inference", False)
        if not need_llm_inference:
            logger.info("LLM inference not needed for this worker, skipping initialization")
            return

        from ..utils.init_utils import create_inference_service_client

        self._inference_client = create_inference_service_client(
            backend=self._config.inference_service.backend,
            config=self._config,
            tinker_service_holder=service_holder,
            weaver_service_holder=service_holder,
        )

    def run(self):
        """
        Main startup function for each worker. Starts a thread and runs _main_loop.
        """
        assert self._trajectory_pool is not None, "TrajectoryPool is not set"
        assert self._dataloader is not None, "DataLoader is not set"
        assert self._weight_sync_controller is not None, "WeightSyncController is not set"

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._main_loop)
        self._thread.start()

    @abstractmethod
    def rollout(self, task: dict[str, Any]) -> str | None:
        """
        Single rollout operation, defined by user. Derived worker classes should override
        this function to implement user-defined worker operations.

        The implementation should call _put_trajectory() and return its result.
        This allows the main loop to handle re-rollout logic.

        Returns:
            str: 'success', 'fail', 're-rollout' (from _put_trajectory), or None if processing failed
        """

    def stop(self):
        """Stop the worker"""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()

    def _main_loop(self):
        """
        Main body is a while loop. Exits when self._stop_event is set.
        In each loop, step() is run. Note that getting data through dataloader
        and writing back data through put_trajectory are both called by the derived class,
        ensuring flexible and customizable implementation.
        """
        while not self._stop_event.is_set():
            self._next_task = self._get_rollout_task()
            if self._next_task is None:
                time.sleep(0.1)
                continue
            with self._activity_tracker.track(self._module_name, "rollout"):
                # If the _next_task is None, the loop will end.
                # The activity tracker will mark the module as quiescent.
                while self._next_task is not None:
                    result = self.rollout(copy.deepcopy(self._next_task))
                    # If not re-rollout, get the next task
                    # Otherwise, the next task will not change.
                    if result != "re-rollout":
                        self._next_task = self._get_rollout_task()

    def _put_trajectory(self, trajectory: Trajectory) -> str:
        """
        Push the trajectory to the appropriate destination based on mode.
        Routes to ValidationCollector during validation, TrajectoryPool during training.

        Args:
            trajectory: Trajectory to push

        Returns:
            str: 'success', 'fail', or 're-rollout'
        """
        if self._is_running_validate:
            return execute(self._validator.put_trajectory, trajectory)
        else:
            return execute(self._trajectory_pool.put_trajectory, trajectory)

    def _put_rollout_task(self, task: dict[str, Any]) -> bool:
        """
        Push data item back to DataLoader for further processing. Returns whether the results are inserted successfully.

        Args:
            task: Rollout task to push
        Returns:
            bool: Whether the results are inserted successfully
        """
        return execute(self._dataloader.add_item, task)

    def _get_rollout_task(self) -> dict[str, Any] | None:
        """
        Get a rollout task from the DataLoader with weight-aware coordination.
        """
        if self._is_running_validate:
            return execute(self._validate_dataloader.get_next_item)
        else:
            return execute(self._dataloader.get_next_item)

    def begin_validate(self) -> None:
        """
        Begin validation mode - switch to using validation dataloader and collector.
        """
        self._is_running_validate = True

    def end_validate(self) -> None:
        """
        End validation mode - switch back to training dataloader and trajectory pool.
        """
        self._is_running_validate = False


def validate_trajectory(trajectory: Trajectory) -> bool:  # pylint: disable=unused-argument
    """
    Validate the trajectory.
    """
    return True
