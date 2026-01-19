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
DataLoader - Abstract base class for data loading
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, override

from omegaconf import DictConfig

from ..base_module import NexRLModule
from ..weight_sync.weight_sync_controller import WeightSyncController

logger = logging.getLogger(__name__)


class BaseDataLoader(NexRLModule, ABC):
    """
    Abstract base class for data loading
    """

    def __init__(
        self, config: DictConfig, is_validate: bool = False
    ):  # pylint: disable=unused-argument
        """
        Read file path from config.file, call _read_file to read file

        Args:
            config: Configuration file
        """
        super().__init__()
        self._weight_sync_controller: WeightSyncController = None  # type: ignore  # Set via set_module_references()
        self._is_validate: bool = is_validate

    def add_item(self, item: dict[str, Any]) -> None:
        """
        Add a data item to the data. By default, it is added at the end of the data
        (returned by the next `__getitem__` call), i.e., calls `add_item_back`

        Args:
            item: Data item to add
        """
        self.add_item_back(item)

    @abstractmethod
    def add_item_front(self, item: dict[str, Any]) -> None:
        """
        Add a data item to the data, adding it at the beginning of the data
        (returned last by `__getitem__`)

        Args:
            item: Data item to add
        """

    @abstractmethod
    def add_item_back(self, item: dict[str, Any]) -> None:
        """
        Add a data item to the data, adding it at the end of the data
        (returned by the next `__getitem__` call)

        Args:
            item: Data item to add
        """

    @abstractmethod
    def get_next_item(self) -> dict[str, Any] | None:
        """
        Get the next item in sequence (mock implementation of sequential access)

        Returns:
            dict[str, Any]: Next data item or None if no more items
        """

    @abstractmethod
    def can_return_item(self) -> bool:
        """
        Check if the data loader can return an item
        """

    @abstractmethod
    def is_finished(self) -> bool:
        """
        Check if the data loader is empty
        """

    @abstractmethod
    def unlock_for_weight_sync(self) -> None:
        """
        Notify the data loader that the weight sync is finished.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the data loader to the beginning.

        This allows the data loader to be reused, particularly useful for
        validation data loaders that need to iterate over the same dataset
        multiple times.
        """

    @abstractmethod
    def skip_batches(self, num_batches: int) -> None:
        """
        Skip the first num_batches batches from the dataloader.
        """

    def set_module_references(self, weight_sync_controller: WeightSyncController):
        """
        Set the weight sync controller for this data loader.
        """
        self._weight_sync_controller = weight_sync_controller

    @staticmethod
    def repeat_item(item: dict[str, Any], n: int) -> list[dict[str, Any]]:
        """
        Repeat a data item n times, adding group_id and run_id to each copy

        Args:
            item: Original data item to repeat
            n: Number of times to repeat the item

        Returns:
            list[dict[str, Any]]: List of n repeated items, each with group_id and run_id
        """
        repeated_items = []
        import uuid

        group_id = str(uuid.uuid4())

        for run_id in range(n):
            # Create a copy of the item
            item_copy = item.copy()
            # Add group_id and run_id
            item_copy["group_id"] = group_id
            item_copy["run_id"] = run_id
            repeated_items.append(item_copy)

        return repeated_items


class SequentialDataLoader(BaseDataLoader, ABC):
    """
    Abstract base class for sequential data loaders that maintain insertion order

    This class provides the sequential iteration logic using __getitem__ while leaving
    the actual storage mechanism to concrete implementations. It maintains a current
    index for sequential access and implements get_next_item() using the abstract
    __getitem__ method.

    Concrete implementations must provide:
    - __len__, __getitem__, add_item_front, add_item_back, is_finished
    """

    def __init__(self, config: DictConfig, is_validate: bool = False):
        super().__init__(config, is_validate)

        self._lock: threading.Lock = threading.Lock()
        self._data_index: int = 0  # index of next loaded data, also the number of loaded data

        # Buffering configuration
        self._batch_size: int = config.get("batch_size", 32)
        self._data_buffer: list[dict[str, Any]] = (
            []
        )  # buffer of items to return. If no buffered data is available, the data loader will fetch a new batch from the underlying storage.
        self._buffer_index: int = 0  # index of next item to return from _data_buffer

        # Batch order control
        self._keep_batch_order: bool = config.get("keep_batch_order", False)

        if self._is_validate:
            self._keep_batch_order = False

        # Rollout repeat configuration
        self._rollout_repeat_n: int = config.get("rollout_repeat_n", 1)

        self._lock_for_weight_sync_event: threading.Event = threading.Event()
        self._lock_for_weight_sync_event.clear()

        logger.info(
            f"SequentialDataLoader initialized - is_validate: {self._is_validate}, batch_size: {self._batch_size}, "
            f"keep_batch_order: {self._keep_batch_order}, rollout_repeat_n: {self._rollout_repeat_n}"
        )

    @override
    def can_return_item(self) -> bool:
        """
        Check if the data loader can return an item
        """
        with self._lock:
            return self._buffer_index < len(self._data_buffer)

    @override
    def get_next_item(self) -> dict[str, Any] | None:
        """
        Get the next item from the data buffer

        If buffer is empty or exhausted, fetches a new batch using _fetch_one_batch().
        Handles batch order control if keep_batch_order is enabled.

        Returns:
            dict[str, Any]: Next data item or None if no more items available
        """
        with self._lock:
            # Check if we need to fetch a new batch
            if self._buffer_index >= len(self._data_buffer):
                # Buffer is exhausted, try to fetch a new batch.
                # If keep_batch_order is enabled, cannot fetch batch, unless it is the initial fetch.
                # The _try_fetch_batch will be called by weight sync controller with keep_batch_order.
                if not self._keep_batch_order or self._data_index == 0:
                    fetch_success = self._try_fetch_batch()
                    if not fetch_success:
                        return None
                else:
                    # Need to wait for weight sync
                    if not self._lock_for_weight_sync_event.is_set():
                        # First time buffer exhausted, set the lock
                        logger.info(
                            "SequentialDataLoader: Buffer exhausted, locking for weight sync"
                        )
                        self._lock_for_weight_sync_event.set()
                    else:
                        # Lock already set, continue waiting
                        logger.debug("SequentialDataLoader: Waiting for weight sync unlock")
                    return None

            # Return item from buffer
            item = self._data_buffer[self._buffer_index]
            self._buffer_index += 1
            self._data_index += 1
            return item

    @override
    def unlock_for_weight_sync(self) -> None:
        """
        Release the batch order lock, allowing the next batch to be fetched

        This should be called when the current batch has been fully processed
        by downstream components.
        """
        with self._lock:
            logger.info("SequentialDataLoader: Unlocking for weight sync")
            self._lock_for_weight_sync_event.clear()
            # Fetch next batch if buffer is exhausted and keep_batch_order is enabled
            if self._keep_batch_order and self._buffer_index >= len(self._data_buffer):
                self._try_fetch_batch()

    @override
    def add_item_front(self, item: dict[str, Any]) -> None:
        """
        Add a data item to the front of the current buffer

        The item will be returned by the next get_next_item() call.
        This operates on the buffer level, not the underlying storage.

        Args:
            item: Data item to add to the front of the buffer
        """
        with self._lock:
            # Insert the item at the current buffer position so it's returned next
            self._data_buffer.insert(self._buffer_index, item)

    def _try_fetch_batch(self) -> bool:
        """
        Try to fetch a new batch, respecting batch order control

        Returns:
            bool: True if batch was fetched successfully, False if no more data
        """

        # Fetch new batch from implementation
        new_batch = self._fetch_batch_data()
        if not new_batch:  # Empty batch means no more data
            return False

        # Apply rollout repetition if configured
        # if self._rollout_repeat_n > 1:
        if not self._is_validate:
            repeated_batch = []
            for item in new_batch:
                repeated_items = BaseDataLoader.repeat_item(item, self._rollout_repeat_n)
                repeated_batch.extend(repeated_items)
            new_batch = repeated_batch
            logger.info(
                f"SequentialDataLoader: Applied rollout_repeat_n={self._rollout_repeat_n}, "
                f"expanded batch to {len(new_batch)} items"
            )

        # Replace buffer with new batch
        self._data_buffer = new_batch
        self._buffer_index = 0

        logger.info(f"SequentialDataLoader: Fetched new batch of {len(new_batch)} items")
        return True

    @abstractmethod
    def _fetch_batch_data(self) -> list[dict[str, Any]]:
        """
        Fetch one batch of items from the underlying storage

        This method should return up to batch_size items. Concrete implementations
        should handle their specific data source logic here.

        Returns:
            list[dict[str, Any]]: List of data items (up to batch_size)
        """

    @abstractmethod
    def _reset_iterator(self) -> None:
        """
        Reset the underlying iterator/data source to the beginning.

        Concrete implementations should override this to handle their specific
        iterator reset logic (e.g., creating a new DataLoader iterator).
        """

    @override
    def reset(self) -> None:
        """
        Reset the dataloader to the beginning.

        This allows the data loader to be reused across multiple iterations,
        particularly useful for validation data loaders.

        The method:
        - Resets the underlying iterator via _reset_iterator()
        - Clears the buffer and resets buffer/data indices
        - Clears any weight sync locks
        """
        with self._lock:
            # Reset the underlying iterator (implementation-specific)
            self._reset_iterator()

            # Clear the buffer and reset indices
            self._data_buffer = []
            self._buffer_index = 0
            self._data_index = 0

            # Clear weight sync lock if set
            self._lock_for_weight_sync_event.clear()

            logger.info(
                f"SequentialDataLoader: Reset dataloader to beginning "
                f"(is_validate={self._is_validate})"
            )

    def skip_batches(self, num_batches: int) -> None:
        """
        Skip the first num_batches batches from the dataloader.

        This is used for resuming training from a checkpoint. When resuming,
        we need to skip the batches that have already been consumed in the
        previous training run.

        Note: This method skips underlying data batches (before rollout_repeat_n
        expansion). If rollout_repeat_n > 1, each underlying batch will be expanded
        into multiple repeated items, but we skip at the batch level.

        Args:
            num_batches: Number of batches to skip

        Raises:
            RuntimeError: If dataloader has already started iteration
        """
        with self._lock:
            if self._data_index > 0:
                raise RuntimeError(
                    f"Cannot skip batches after dataloader has started iteration. "
                    f"Current data_index: {self._data_index}"
                )

            if num_batches <= 0:
                logger.info("No batches to skip (num_batches <= 0)")
                return

            logger.info(
                f"SequentialDataLoader: Skipping {num_batches} batches for resume "
                f"(rollout_repeat_n={self._rollout_repeat_n})"
            )

            skipped_count = 0
            for i in range(num_batches):
                # Fetch and discard the batch
                success = self._try_fetch_batch()
                if not success:
                    logger.warning(
                        f"SequentialDataLoader: Reached end of data after skipping "
                        f"{skipped_count}/{num_batches} batches. Dataset may be smaller "
                        f"than expected or checkpoint step is invalid."
                    )
                    break

                skipped_count += 1

                # Log progress for large skips
                if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                    logger.info(f"Skipped {i + 1}/{num_batches} batches...")

            # After skipping, clear the buffer so the next get_next_item() starts fresh
            # Update _data_index to reflect the skipped items
            total_items_skipped = self._data_index
            self._data_buffer = []
            self._buffer_index = 0

            logger.info(
                f"SequentialDataLoader: Successfully skipped {skipped_count} batches "
                f"({total_items_skipped} items after rollout_repeat_n expansion)"
            )
