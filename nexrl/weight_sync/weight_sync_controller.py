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
Weight Manager for NexRL framework
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import aiohttp
import requests  # type: ignore[import-untyped]
from omegaconf import DictConfig

from ..base_module import NexRLModule
from ..executor import execute
from ..nexrl_types import ModelTag

if TYPE_CHECKING:
    from ..data_loader import BaseDataLoader
    from ..tinker.tinker_service_holder import TinkerServiceHolder
    from ..trajectory_pool import TrajectoryPool
    from ..weaver.weaver_service_holder import WeaverServiceHolder

logger = logging.getLogger(__name__)


@dataclass
class RolloutServiceState:
    """State tracking for model weight synchronization and service info"""

    # State tracking
    state: Literal["need_sync", "syncing", "running"] = "running"
    rollout_model_version: int = 0
    train_model_version: int = 0

    # Service info
    model_name: str = ""
    weight_type: str = ""
    weight_path: str = ""
    backend: str = ""
    base_url: str = ""

    # Per-state lock for thread safety
    _lock: threading.Lock = threading.Lock()


class WeightSyncController(NexRLModule):
    """
    Weight Manager - Manages model weights and synchronization coordination

    Supports three synchronization modes:
    - sync: Block all workers until all sync to newest version
    - fully-async: No blocking, workers sync opportunistically
    - batch-async: Block individual workers when staleness exceeds threshold
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self._config = config

        # Model state management - merged rollout service info and states
        self._rollout_services: dict[ModelTag, RolloutServiceState] = {}
        self._sync_mode = config.sync_mode
        self._staleness_threshold = config.staleness_threshold

        # Validation coordination
        self._validate_freq = config.validate_freq  # 0 = disabled
        self._waiting_for_validation = False

        # Initialize rollout service with inference service config if available
        # Note: Currently only supports a single inference service
        self._inference_service_config = config.get("inference_service", {})
        # identifier serves as model_tag for weight sync coordination
        self._identifier = self._inference_service_config.get("identifier", "default")
        self._rollout_services[self._identifier] = RolloutServiceState(
            model_name=self._inference_service_config.model,
            weight_type=self._inference_service_config.weight_type,
            weight_path=self._config.get("sync_weight_path", ""),
            backend=self._inference_service_config.backend,
            base_url=self._inference_service_config.get("base_url", ""),
        )

        # References to other components (will be set via dependency injection)
        self._trajectory_pool: "TrajectoryPool" = None  # type: ignore  # Will be set by controller
        self._dataloader: "BaseDataLoader" = None  # type: ignore  # Will be set by controller
        self._tinker_service_holder: "TinkerServiceHolder | None" = None  # For Tinker backend
        self._weaver_service_holder: "WeaverServiceHolder | None" = None  # For Weaver backend

        logger.info(f"WeightSyncController initialized with sync_mode: {self._sync_mode}")

    def set_module_references(
        self, dataloader: "BaseDataLoader", trajectory_pool: "TrajectoryPool"
    ) -> None:
        """Set references to other modules for coordination"""
        self._trajectory_pool = trajectory_pool
        self._dataloader = dataloader

    def set_tinker_service_holder(self, tinker_service_holder: "TinkerServiceHolder") -> None:
        """Set reference to Tinker service holder (for Tinker backend only)"""
        self._tinker_service_holder = tinker_service_holder
        logger.info("TinkerServiceHolder reference set in WeightSyncController")

    def set_weaver_service_holder(self, weaver_service_holder: "WeaverServiceHolder") -> None:
        """Set reference to Weaver service holder (for Weaver backend only)"""
        self._weaver_service_holder = weaver_service_holder
        logger.info("WeaverServiceHolder reference set in WeightSyncController")

    def check_rollout_service_status(self, model_tag: ModelTag) -> Literal["continue", "block"]:
        """
        Check if rollout service should continue or block for the given model.
        Called by InferenceServiceClient before generate and completion calls.

        Args:
            model_tag: Model tag to check status for

        Returns:
            'continue' | 'block'
            continue: rollout service can proceed
            block: rollout service should block and retry
        """
        assert (
            model_tag in self._rollout_services
        ), f"Rollout service not found for model tag: {model_tag}"

        rollout_service = self._rollout_services[model_tag]

        with rollout_service._lock:  # pylint: disable=protected-access
            if rollout_service.state in ["need_sync", "syncing"]:
                return "block"
            else:
                return "continue"

    def is_waiting_for_validation(self) -> bool:
        """
        Check if weight sync is waiting for controller to handle validation

        Returns:
            True if waiting for validation to complete
        """
        return self._waiting_for_validation

    def end_validate(self, model_tag: ModelTag) -> None:
        """
        Controller signals validation is done - safe to unlock

        Args:
            model_tag: Model tag for validation
        """
        self._waiting_for_validation = False
        execute(self._dataloader.unlock_for_weight_sync)
        execute(self._trajectory_pool.unlock_for_weight_sync, model_tag)
        logger.info(f"Validation complete for {model_tag} - locks released")

    def get_rollout_model_version(self, model_tag: ModelTag) -> int:
        """
        Get current rollout model version

        Args:
            model_tag: Model tag to query

        Returns:
            Current rollout model version
        """
        rollout_service = self._rollout_services[model_tag]
        with rollout_service._lock:  # pylint: disable=protected-access
            return rollout_service.rollout_model_version

    def trajectory_pool_notify_batch_ready(self, model_tag: ModelTag) -> None:
        """
        Called when a batch becomes ready in TrajectoryPoolInstance.
        Checks if weight synchronization is needed and locks services if necessary.

        Args:
            model_tag: Model tag for which batch is ready
        """
        # Get or create the rollout service state
        assert (
            model_tag in self._rollout_services
        ), f"Rollout service not found for model tag: {model_tag}"

        rollout_service = self._rollout_services[model_tag]

        logger.info(f"Trajectory pool batch ready for {model_tag}, sync mode: {self._sync_mode}")

        with rollout_service._lock:  # pylint: disable=protected-access
            # Check if we need to block dataloader and trajectory pool for weight sync
            if self._sync_mode == "sync" or (
                self._sync_mode == "batch-async"
                and rollout_service.train_model_version + 1 - rollout_service.rollout_model_version
                > self._staleness_threshold
            ):
                assert (
                    rollout_service.state == "running"
                ), "Rollout service should be running when weight sync is needed"

                logger.info(
                    f"Batch ready for {model_tag}, weight sync needed. "
                    f"Rollout version: {rollout_service.rollout_model_version}, "
                    f"Incoming train version: {rollout_service.train_model_version + 1}"
                )

                # Lock rollout service and trajectory pool
                rollout_service.state = "need_sync"

                # Notify trajectory pool to lock for this model
                execute(self._trajectory_pool.notify_need_weight_sync, model_tag)

                logger.info(f"Locked rollout service and trajectory pool for {model_tag}")
            else:
                logger.info(f"Batch ready for {model_tag}, weight sync not needed. ")
                rollout_service.state = "running"
                execute(self._dataloader.unlock_for_weight_sync)
                execute(self._trajectory_pool.unlock_for_weight_sync, model_tag)

    def train_worker_notify_weight_update(self, worker_name: str, model_tag: ModelTag) -> None:
        """
        Handle training completion and coordinate weight synchronization.
        This method synchronously performs weight synchronization.

        Args:
            worker_name: Name of the training worker
            model_tag: Model tag that was trained
        """
        assert (
            model_tag in self._rollout_services
        ), f"Rollout service not found for model tag: {model_tag}"

        rollout_service = self._rollout_services[model_tag]
        need_sync = False

        with rollout_service._lock:  # pylint: disable=protected-access
            assert model_tag in self._rollout_services
            rollout_service.train_model_version += 1

            logger.info(
                f"Training completed by {worker_name} for {model_tag}, new weight version: {rollout_service.train_model_version}"
            )

            # If rollout service needs sync, prepare for synchronous weight update
            if rollout_service.state == "need_sync":
                rollout_service.state = "syncing"
                need_sync = True
                logger.info(f"Starting weight synchronization for {model_tag}")

        # Perform synchronous weight synchronization (outside of lock)
        if need_sync:
            success = self.sync_weight_to_rollout_service(model_tag)
            if not success:
                raise RuntimeError(f"Weight synchronization failed for {model_tag}")

            with rollout_service._lock:  # pylint: disable=protected-access
                rollout_service.rollout_model_version = rollout_service.train_model_version
                rollout_service.state = "running"

            logger.info(
                f"Weight synchronization completed for {model_tag} "
                f"to version {rollout_service.rollout_model_version}"
            )

            # Check if validation should happen at this version
            if (
                self._validate_freq > 0
                and rollout_service.rollout_model_version > 0
                and rollout_service.rollout_model_version % self._validate_freq == 0
            ):
                logger.info(
                    f"Version {rollout_service.rollout_model_version} requires validation "
                    f"(frequency: every {self._validate_freq} steps), keeping locks in place"
                )
                self._waiting_for_validation = True
                return  # Don't unlock - controller will handle it after validation

            # Unlock dataloader
            execute(self._dataloader.unlock_for_weight_sync)
            # Unlock trajectory pool
            execute(self._trajectory_pool.unlock_for_weight_sync, model_tag)

    def sync_weight_to_rollout_service(self, model_tag: ModelTag) -> bool:
        """
        Synchronously sync weights to rollout service.
        This is a placeholder for the actual implementation.

        Args:
            model_tag: Model tag to sync

        Returns:
            True if sync successful, False otherwise
        """
        if self._config.sync_method == "network":
            rollout_service = self._rollout_services[model_tag]
            if rollout_service.backend in ("vllm", "sglang"):
                t0 = time.time()
                response = requests.post(
                    rollout_service.base_url + "/update_weights",
                    json={
                        "model_name": rollout_service.model_name,
                        "weight_type": rollout_service.weight_type,
                        "weight_path": rollout_service.weight_path,
                    },
                    timeout=600,
                )
                t1 = time.time()
                logger.info(f"Finish updating weights in {t1 - t0} seconds")
                assert (
                    response is not None and response.status_code == 200
                ), f"Failed to update weights: {response.text if response else 'No response'}"
            else:
                raise ValueError(f"Unsupported backend: {rollout_service.backend}")
            return True
        elif self._config.sync_method == "disk":
            rollout_service = self._rollout_services[model_tag]
            if rollout_service.backend == "sglang":
                t0 = time.time()
                # Must using sglang-router to enable llm serive discovery.
                # We always query all workers to get the latest worker list.
                response = requests.get(rollout_service.base_url + "/workers", timeout=60)
                logger.info(
                    f"list_workers, base_url: {rollout_service.base_url}, response code: {response.status_code}, response: {response.text}"
                )
                assert response.status_code == 200, f"Failed to get worker list: {response.text}"

                # Parse response and extract worker URLs
                response_data = response.json()
                workers = response_data.get("workers", [])
                worker_urls = [worker.get("url") for worker in workers]

                logger.info(
                    f"Found {len(worker_urls)} total workers to update: {worker_urls}, "
                    f"weight_path: {rollout_service.weight_path}"
                )

                async def update_worker(session, worker_url):
                    try:
                        async with session.post(
                            worker_url + "/update_weights_from_disk",
                            json={"model_path": rollout_service.weight_path},
                            timeout=aiohttp.ClientTimeout(total=600),
                        ) as resp:
                            if resp.status != 200:
                                text = await resp.text()
                                logger.error(f"Failed to update worker {worker_url}: {text}")
                                return False
                            logger.info(f"Successfully updated worker {worker_url}")
                            return True
                    except Exception as e:
                        logger.error(f"Error updating worker {worker_url}: {e}")
                        return False

                async def update_all_workers():
                    async with aiohttp.ClientSession() as session:
                        tasks = [update_worker(session, url) for url in worker_urls]
                        results = await asyncio.gather(*tasks)
                        return all(results)

                # Run async updates
                success = asyncio.run(update_all_workers())
                t1 = time.time()

                if not success:
                    raise RuntimeError("Failed to update some workers")
                logger.info(f"Finish updating weights in {t1 - t0} seconds")
            else:
                raise ValueError(f"Unsupported backend: {rollout_service.backend}")
            return True
        elif self._config.sync_method == "mock":
            time.sleep(2)
            return True
        elif self._config.sync_method == "tinker":
            assert self._tinker_service_holder is not None, "TinkerServiceHolder not set"
            self._tinker_service_holder.update_sampling_client()
            return True
        elif self._config.sync_method == "weaver":
            assert self._weaver_service_holder is not None, "WeaverServiceHolder not set"
            self._weaver_service_holder.update_sampling_client()
            return True
        else:
            raise ValueError(f"Unsupported sync method: {self._config.sync_method}")
