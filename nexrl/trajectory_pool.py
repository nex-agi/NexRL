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

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig

from .base_module import NexRLModule
from .executor import execute
from .nexrl_types import Batch, ModelTag, Trajectory

if TYPE_CHECKING:
    from nexrl.activity_tracker import ActivityTrackerProxy

    from .data_loader import BaseDataLoader
    from .weight_sync.weight_sync_controller import WeightSyncController


logger = logging.getLogger(__name__)


class TrajectoryStoreBase(ABC):
    """Abstract base class for trajectory stores"""

    # Public interface for trajectory stores
    model_tag: ModelTag

    def __init__(self, config: DictConfig, model_tag: ModelTag):
        self._config = config
        self._lock = threading.Lock()
        # Unified finished samples storage - all stores use this
        self._finished_samples: list[Trajectory] = []
        self.model_tag = model_tag

    @abstractmethod
    def put_trajectory(self, trajectory: Trajectory) -> bool:
        """
        Add a trajectory to the pool.

        Args:
            trajectory: Trajectory data to add

        Returns:
            True if successfully added, False otherwise
        """
        pass

    def get_batch(self, batch_size: int | None = None) -> Batch | None:
        """
        Get a batch of trajectories from finished_samples (unified implementation).

        Args:
            batch_size: Number of trajectories to include in the batch

        Returns:
            Batch containing the trajectories, or None if not enough samples
        """
        with self._lock:
            if batch_size is None:
                batch_size = len(self._finished_samples)
            if len(self._finished_samples) < batch_size:
                return None

            # Get batch_size trajectories from finished_samples
            batch_trajectories = self._finished_samples[:batch_size]
            self._finished_samples = self._finished_samples[batch_size:]
            return self._to_batch(batch_trajectories)

    def is_empty(self) -> bool:
        """
        Check if the pool is empty (unified implementation).

        Returns:
            True if the pool has no available samples
        """
        return len(self._finished_samples) == 0 and self._is_active_storage_empty()

    @abstractmethod
    def _is_active_storage_empty(self) -> bool:
        """
        Check if the active storage (before finished_samples) is empty.

        Returns:
            True if active storage is empty
        """
        pass

    def _to_batch(self, trajectories: list[Trajectory]) -> Batch:
        """
        Convert list of trajectories to a Batch object.
        This is shared logic across all stores.
        """
        if not trajectories:
            raise ValueError("Cannot create batch from empty trajectories")

        # Check that all trajectories have the same keys
        if len(trajectories) > 1:
            first_keys = set(trajectories[0].keys())
            for i, trajectory in enumerate(trajectories[1:], 1):
                current_keys = set(trajectory.keys())
                if current_keys != first_keys:
                    raise ValueError(
                        f"Trajectory at index {i} has different keys. Expected: {first_keys}, Got: {current_keys}"
                    )

        # Initialize the values dictionary
        values: dict[str, Any] = {}

        # Initialize data structures based on the first trajectory
        first_trajectory = trajectories[0]
        for key in first_trajectory.keys():
            values[key] = []

        # Collect all values
        for trajectory in trajectories:
            for key, value in trajectory.items():
                values[key].append(value)

        # Merge tensors into 2D tensors
        for key, value_list in values.items():
            if value_list and isinstance(value_list[0], torch.Tensor):
                if all(t.dim() == 1 for t in value_list):
                    # All tensors are 1D, stack them to create 2D tensor
                    values[key] = torch.stack(value_list, dim=0)
                elif all(t.dim() == 2 for t in value_list):
                    # All tensors are 2D, stack them to create 3D tensor, then reshape to 2D
                    stacked = torch.stack(value_list, dim=0)
                    batch_size, seq_len, feature_dim = stacked.shape
                    values[key] = stacked.view(batch_size, -1)
                else:
                    # Mixed dimensions or higher dimensions, raise error
                    raise ValueError(f"Tensors for key '{key}' have inconsistent dimensions")

        # Add batch_size to the metadata
        metadata = {
            "batch_size": len(trajectories),
            "model_tag": self.model_tag,
            "temperature": trajectories[0].get("temperature", 1.0),
        }

        return Batch(values=values, metadata=metadata)


class SimpleTrajectoryStore(TrajectoryStoreBase):
    """Simple trajectory store without any grouping - directly adds to finished_samples"""

    def __init__(self, config: DictConfig, model_tag: ModelTag):
        super().__init__(config, model_tag)
        # No intermediate storage needed for simple store

    def put_trajectory(self, trajectory: Trajectory) -> bool:
        """Add trajectory directly to finished_samples"""
        try:
            with self._lock:
                self._finished_samples.append(trajectory)
            return True
        except Exception as e:
            logger.error(f"Error adding trajectory to simple store: {e}")
            return False

    def _is_active_storage_empty(self) -> bool:
        """Simple store has no active storage"""
        return True


class GroupedTrajectoryStore(TrajectoryStoreBase):
    """Trajectory store with single-level grouping (e.g., by uid)"""

    def __init__(self, config: DictConfig, model_tag: ModelTag):
        super().__init__(config, model_tag)
        self._group_key = config.get("group_key", "uid")  # Key to group by
        self._group_size = config.get("group_size", 1)  # Size of each group
        self._groups: dict[Any, list[Trajectory]] = {}  # Active groups

    def put_trajectory(self, trajectory: Trajectory) -> bool:
        """Add trajectory to appropriate group"""
        try:
            if self._group_key not in trajectory:
                raise ValueError(f"Group key '{self._group_key}' not found in trajectory")

            group_value = trajectory[self._group_key]

            with self._lock:
                # Add to appropriate group
                if group_value not in self._groups:
                    self._groups[group_value] = []
                self._groups[group_value].append(trajectory)

                # Check if group is complete
                if len(self._groups[group_value]) >= self._group_size:
                    # Move completed group trajectories to finished_samples (flattened)
                    completed_group = self._groups.pop(group_value)
                    self._finished_samples.extend(completed_group)

            return True
        except Exception as e:
            logger.error(f"Error adding trajectory to grouped store: {e}")
            return False

    def _is_active_storage_empty(self) -> bool:
        """Check if active groups storage is empty"""
        return len(self._groups) == 0


class HierarchicalTrajectoryStore(TrajectoryStoreBase):
    """Trajectory store with multi-level hierarchical grouping"""

    def __init__(self, config: DictConfig, model_tag: ModelTag):
        super().__init__(config, model_tag)
        self._key_list: list[str] = config.get("key_list", [])  # Hierarchical keys
        self._group_size: int = config.get("group_size", 1)  # Size at leaf level
        self._hierarchy: dict[Any, Any] = {}  # Hierarchical structure

    def put_trajectory(self, trajectory: Trajectory) -> bool:
        """Add trajectory to hierarchical structure"""
        try:
            if not self._key_list:
                raise ValueError("key_list cannot be empty for hierarchical pool")

            with self._lock:
                self._insert_into_hierarchy(trajectory)
            return True
        except Exception as e:
            logger.error(f"Error adding trajectory to hierarchical store: {e}")
            return False

    def _insert_into_hierarchy(self, trajectory: Trajectory):
        """Insert trajectory into the hierarchical structure"""
        current_level = self._hierarchy
        path = []

        # Navigate through the hierarchy
        for i, key_name in enumerate(self._key_list):
            if key_name not in trajectory:
                raise ValueError(f"Key '{key_name}' not found in trajectory")

            key_value = trajectory[key_name]
            path.append((key_name, key_value))

            if i == len(self._key_list) - 1:
                # Leaf level: store as list
                if key_value not in current_level:
                    current_level[key_value] = []
                current_level[key_value].append(trajectory)

                # Check if leaf group is complete
                if len(current_level[key_value]) >= self._group_size:
                    completed_group = current_level.pop(key_value)
                    self._finished_samples.extend(completed_group)
            else:
                # Intermediate level: create dict if needed
                if key_value not in current_level:
                    current_level[key_value] = {}
                current_level = current_level[key_value]

    def _is_active_storage_empty(self) -> bool:
        """Check if hierarchical storage is empty"""

        def _is_dict_empty(d):
            """Recursively check if nested dict structure is empty"""
            if not isinstance(d, dict):
                return len(d) == 0
            return all(_is_dict_empty(v) for v in d.values()) if d else True

        return _is_dict_empty(self._hierarchy)


class TrajectoryPoolInstance:
    """
    Instance for a single ModelTag that manages a store, lock, and ready batches
    Each instance is responsible for a single model.
    """

    def __init__(self, config: DictConfig, model_tag: ModelTag):
        self.model_tag = model_tag
        self._config = config
        self._lock = threading.Lock()
        self._store = self._create_store(config, model_tag)
        self._check_ready_func = config.get("check_batch_ready_function", "loaded_batch_finished")

        self._weight_sync_event: threading.Event = (
            threading.Event()
        )  # Event for weight sync coordination
        self._weight_sync_event.clear()  # Initially unlocked (set)

        self._activity_tracker: "ActivityTrackerProxy" = None  # type: ignore   # Will be set by trajectory pool
        self._dataloader: "BaseDataLoader" = None  # type: ignore   # Will be set by trajectory pool
        self._weight_sync_controller: "WeightSyncController" = None  # type: ignore   # Will be set by trajectory pool

        # Timing tracking for rollout time
        self._start_rollout_time: float | None = None
        self._batch_count: int = 0

    def _create_store(self, config: DictConfig, model_tag: ModelTag) -> TrajectoryStoreBase:
        """Create appropriate store based on configuration"""
        key_list = config.get("key_list", [])

        if not key_list:
            # No grouping keys specified, use simple store
            return SimpleTrajectoryStore(config, model_tag)
        elif len(key_list) == 1:
            # Single grouping key, use grouped store
            config_copy = config.copy()
            config_copy["group_key"] = key_list[0]
            return GroupedTrajectoryStore(config_copy, model_tag)
        else:
            # Multiple grouping keys, use hierarchical store
            return HierarchicalTrajectoryStore(config, model_tag)

    def put_trajectory(self, trajectory: Trajectory) -> str:
        """Add trajectory to store and check if batch can be formed

        Returns:
            'success': Trajectory added successfully
            'fail': Failed to add trajectory
            're-rollout': Should not happen with event-based approach, kept for compatibility
        """
        # Wait for weight sync to complete if needed (with timeout)
        if self._weight_sync_event.is_set():  # 60 seconds timeout
            logger.debug(f"TrajectoryPoolInstance for {self.model_tag} is waiting for weight sync")
            return "re-rollout"

        # Add to store first
        success = self._store.put_trajectory(trajectory)
        if not success:
            return "fail"

        return "success"

    def _prepare_batch_if_ready(self) -> Batch | None:
        """
        If the trajectory store has enough trajectories to form a batch, organize them into a batch and return.
        """
        batch_ready = False
        batch_size: int | None = None  # None means get all trajectories from store

        if self._check_ready_func == "batch_size_reached":
            batch_ready = self._check_batch_size_reached()
            batch_size = self._config.get("batch_size", 32)
        elif self._check_ready_func == "loaded_batch_finished":
            batch_ready = self._check_loaded_batch_finished()
        elif self._check_ready_func == "batch_size_reached_and_loaded_batch_finished":
            batch_ready = self._check_batch_size_reached() and self._check_loaded_batch_finished()
        else:
            raise ValueError(f"Invalid check_ready_function: {self._check_ready_func}")

        # Check if we can form a batch
        if batch_ready:
            logger.info(f"Batch ready for model_tag {self.model_tag}")

            # Try to get a batch from the store
            batch = self._store.get_batch(
                batch_size=batch_size
            )  # batch_size = None means get all trajectories from store
            assert batch is not None, "Batch should not be None"
            return batch

        return None

    def get_batch(self) -> Batch | None:
        """
        Get a batch if possible

        Args:
            batch_size: Size of batch (currently ignored as we return pre-formed batches)

        Returns:
            A ready batch if available, None otherwise
        """
        batch = None
        with self._lock:
            batch = self._prepare_batch_if_ready()

        if batch is not None:
            current_time = time.time()
            if self._start_rollout_time is not None:
                rollout_time = current_time - self._start_rollout_time
                self._activity_tracker.experiment_logger_post(
                    backend="wandb",
                    data={
                        "timing/rollout_time": rollout_time,
                    },
                    step=self._batch_count,
                )
            self._batch_count += 1

            execute(self._weight_sync_controller.trajectory_pool_notify_batch_ready, self.model_tag)
        return batch

    def is_empty(self) -> bool:
        """Check if store are empty"""
        with self._lock:
            return self._store.is_empty()

    def _check_batch_size_reached(self) -> bool:
        """Check if the store has enough trajectories to form a batch"""
        batch_size = self._config.get("batch_size", 32)
        return len(self._store._finished_samples) >= batch_size

    def _check_loaded_batch_finished(self) -> bool:
        """Check if all workers are finished and dataloader is drained"""
        have_samples_in_store = len(self._store._finished_samples) > 0
        if not have_samples_in_store:
            return False
        dataloader_drained = not execute(self._dataloader.can_return_item)
        rollout_worker_quiescent = self._activity_tracker.is_rollout_worker_quiescent()
        logger.debug(
            f"Checking loaded batch finished: dataloader_drained={dataloader_drained} and rollout_worker_quiescent={rollout_worker_quiescent} and have_samples_in_store={have_samples_in_store}"
        )
        return dataloader_drained and rollout_worker_quiescent and have_samples_in_store

    def lock_for_weight_sync(self) -> None:
        """Lock this instance - block new trajectories until weight sync completes"""
        self._weight_sync_event.set()  # Clear the event (block waiting threads)
        logger.info(f"TrajectoryPoolInstance for {self.model_tag} locked for weight sync")

    def unlock_for_weight_sync(self) -> None:
        """Unlock this instance - allow new trajectories to proceed"""
        self._weight_sync_event.clear()  # Set the event (unblock waiting threads)
        self._start_rollout_time = time.time()  # Mark start of rollout period
        logger.info(f"TrajectoryPoolInstance for {self.model_tag} unlocked from weight sync")

    def is_locked(self) -> bool:
        """Check if this instance is locked for weight sync"""
        return self._weight_sync_event.is_set()

    def set_module_references(
        self,
        dataloader: "BaseDataLoader",
        weight_sync_controller: "WeightSyncController",
        activity_tracker: "ActivityTrackerProxy",
    ) -> None:
        """Set reference to dataloader, weight sync controller, and activity tracker for coordination"""
        self._dataloader = dataloader
        self._weight_sync_controller = weight_sync_controller
        self._activity_tracker = activity_tracker


class TrajectoryPool(NexRLModule):
    """
    Multi-instance trajectory pool that manages separate TrajectoryPoolInstance objects for different models.
    Route trajectory operations to the appropriate instance according to the model_tag.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self._config = config
        self._instances: dict[ModelTag, TrajectoryPoolInstance] = {}
        self._lock = threading.Lock()
        self._weight_sync_controller: "WeightSyncController" = None  # type: ignore # will be set by controller

    def _get_or_create_instance(self, model_tag: ModelTag) -> TrajectoryPoolInstance:
        """Get existing instance or create new one for the given model_tag"""
        if model_tag not in self._instances:
            logger.info(f"Creating new instance for model_tag: {model_tag}")
            instance = TrajectoryPoolInstance(self._config, model_tag)
            instance.set_module_references(
                dataloader=self._dataloader,
                weight_sync_controller=self._weight_sync_controller,
                activity_tracker=self._activity_tracker,
            )
            self._instances[model_tag] = instance
        return self._instances[model_tag]

    def put_trajectory(self, trajectory: Trajectory) -> str:
        """
        Add trajectory to appropriate instance based on ModelTag
        Args:
            trajectory: Trajectory to add

        Returns:
            True if trajectory is added successfully, False otherwise
        """
        model_tag = trajectory.get("model_tag", "default")

        with self._lock:
            instance = self._get_or_create_instance(model_tag)

        return instance.put_trajectory(trajectory)

    def get_batch(self, model_tag: ModelTag | None = None) -> Batch | None:
        """
        Get a batch from specified model_tag instance, or from default if not specified

        Args:
            batch_size: Size of the batch to retrieve
            model_tag: ModelTag to get batch from. If None, uses default model_tag

        Returns:
            Batch from the specified instance, or None if not enough samples
        """
        if model_tag is None:
            return self._get_batch_any()

        assert model_tag in self._instances, f"Model tag {model_tag} not found in instances"

        return self._instances[model_tag].get_batch()

    def _get_batch_any(self) -> Batch | None:
        """
        Get a batch from any available instance that contains enough trajectories to form a batch

        Args:

        Returns:
            Batch from any instance that contains enough trajectories to form a batch, or None if no instance has enough
        """
        for model_tag in self._instances.keys():
            batch = self.get_batch(model_tag)
            if batch is not None:
                return batch

        return None

    def is_empty(self, model_tag: ModelTag | None = None) -> bool:
        """
        Check if specified instance is empty, or if all instances are empty

        Args:
            model_tag: ModelTag to check. If None, checks if all instances are empty

        Returns:
            True if specified instance (or all instances) is empty
        """
        with self._lock:
            if model_tag is not None:
                if model_tag not in self._instances:
                    return True
                return self._instances[model_tag].is_empty()
            else:
                # Check if all instances are empty
                return all(instance.is_empty() for instance in self._instances.values())

    def get_model_tags(self) -> list[ModelTag]:
        """Get list of all ModelTags that have instances"""
        with self._lock:
            return list(self._instances.keys())

    def notify_need_weight_sync(self, model_tag: ModelTag) -> None:
        """Lock a specific instance. Called by weight manager."""
        with self._lock:
            if model_tag in self._instances:
                self._instances[model_tag].lock_for_weight_sync()

    def unlock_for_weight_sync(self, model_tag: ModelTag) -> None:
        """Unlock a specific instance. Called by weight manager."""
        with self._lock:
            if model_tag in self._instances:
                self._instances[model_tag].unlock_for_weight_sync()

    def set_module_references(
        self, dataloader: "BaseDataLoader", weight_sync_controller: "WeightSyncController"
    ) -> None:
        """Set reference to weight manager for coordination"""
        self._dataloader = dataloader
        self._weight_sync_controller = weight_sync_controller
