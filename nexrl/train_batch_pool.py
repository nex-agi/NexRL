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
Training Batch Pool for NexRL framework
"""

import logging
from typing import Any

from omegaconf import DictConfig

from .base_module import NexRLModule
from .nexrl_types import Batch

# Type alias for model tags
ModelTag = str

logger = logging.getLogger(__name__)


class TrainBatchPool(NexRLModule):
    """
    Training Batch Pool - Manages training batches
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the training batch pool
        """
        super().__init__()
        self._config = config
        self._batch_pool_table: dict[ModelTag, list[Batch]] = {}
        logger.info("TrainBatchPool initialized")

    def put_batch(self, batch: Batch, update_fn: str) -> bool:
        """
        Insert a batch to the batch pool

        Args:
            batch: Batch to add
            model: The model to train

        Returns:
            bool: Whether the insert is successful
        """
        try:
            model = batch.metadata.get("model_tag", "default")
            if model not in self._batch_pool_table:
                self._batch_pool_table[model] = []
            self._batch_pool_table[model].append(batch)
            logger.info(
                f"Added batch with {batch.metadata.get('batch_size', 0)} samples to training pool for model {model}"
            )
            return True
        except Exception as e:
            logger.error(f"Error adding batch to training pool: {e}")
            return False

    def get_batch(self, model: ModelTag | None = "default") -> Batch | None:
        """
        Return a batch to the model

        Args:
            model: The model requiring the batch

        Returns:
            Batch: Training batch or None if not available
        """
        try:
            if model not in self._batch_pool_table or not self._batch_pool_table[model]:
                # logger.warning(f"No batches available for model {model}")
                return None

            batch = self._batch_pool_table[model].pop(0)
            logger.info(
                f"Retrieved batch with {batch.metadata.get('batch_size', 0)} samples for model {model}"
            )
            return batch
        except Exception as e:
            logger.error(f"Error getting batch from training pool: {e}")
            return None

    def is_empty(self) -> bool:
        """
        Check if the train batch pool is empty
        """
        for _, batches in self._batch_pool_table.items():
            if len(batches) > 0:
                return False
        return True
