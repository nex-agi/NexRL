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
Type definitions for NexRL framework
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

# Type aliases for better readability
ModelTag = str  # Type alias for model tags

# Trajectory is now a dictionary with string keys
Trajectory = dict[str, Any]


class NexRLRole(Enum):
    """Define different roles in NexRL that can be mapped to resource pools."""

    ROLLOUT_WORKER = "rollout_worker"
    TRAIN_WORKER = "train_worker"
    ALGORITHM_PROCESSOR = "algorithm_processor"
    TRAJECTORY_POOL = "trajectory_pool"
    TRAIN_BATCH_POOL = "train_batch_pool"
    WEIGHT_SYNC_CONTROLLER = "weight_sync_controller"
    DATA_LOADER = "data_loader"
    VALIDATE_DATALOADER = "validate_dataloader"
    VALIDATOR = "validator"


@dataclass
class Batch:
    """A batch of rollout results for training"""

    values: dict[
        str, Any
    ]  # Tensor or non-Tensor iterable data, length should be metadata['batch_size']
    metadata: dict[str, Any]  # Metadata about the batch

    def __len__(self) -> int:
        assert "batch_size" in self.metadata, "batch_size must be in metadata"
        return self.metadata["batch_size"]

    def copy(self) -> "Batch":
        return Batch(self.values.copy(), self.metadata.copy())

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Batch to a single dictionary containing both values and metadata.
        If there are overlapping keys, metadata keys will overwrite values keys.
        """
        ret = {**self.values, **self.metadata}
        return ret

    @staticmethod
    def remove_redundant_left_padding(
        data: "Batch",
        pad_token_id: int,
        fields: list[str] | None = None,
        anchor_field: str = "input_ids",
        max_strip_threshold: int = -1,
    ) -> "Batch":
        """Remove redundant left padding tokens that are common across all sequences.

        Args:
            data: The batch to process
            pad_token_id: The ID of the padding token
            fields: List of field names to strip padding from. If None, strips from all 2D tensors
            anchor_field: The field to use for determining padding length
            max_strip_threshold: Maximum number of tokens to strip. If -1, no limit

        Returns:
            A new Batch with left padding removed
        """
        # If anchor_field not in data.values, return data as is
        if anchor_field not in data.values:
            return data

        anchor_tensor = data.values[anchor_field]

        # If not a tensor or not 2D, return as is
        if not isinstance(anchor_tensor, torch.Tensor) or len(anchor_tensor.shape) != 2:
            return data

        batch_size, seq_len = anchor_tensor.shape

        # Find left padding count for each sequence
        padding_mask = anchor_tensor == pad_token_id
        non_pad_mask = ~padding_mask
        cumsum = non_pad_mask.cumsum(dim=1)
        left_pad_counts = (cumsum == 0).sum(dim=1).tolist()

        # Find minimum padding count across all sequences
        min_pad_count = min(left_pad_counts)

        if min_pad_count == 0:
            return data

        # Apply max_strip_threshold if specified
        if max_strip_threshold >= 0:
            min_pad_count = min(min_pad_count, max_strip_threshold)

        # Create new values dict with padding stripped
        new_values = {}
        for key, value in data.values.items():
            # Only strip 2D tensors/arrays in specified fields (or all if fields is None)
            if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                if fields is None or key in fields:
                    new_values[key] = value[:, min_pad_count:]
                else:
                    new_values[key] = value
            else:
                new_values[key] = value

        # Return new Batch with stripped values
        return Batch(values=new_values, metadata=data.metadata.copy())

    @staticmethod
    def remove_redundant_right_padding(
        data: "Batch",
        pad_token_id: int,
        fields: list[str] | None = None,
        anchor_field: str = "input_ids",
        max_strip_threshold: int = -1,
    ) -> "Batch":
        """Remove redundant right padding tokens that are common across all sequences.

        Args:
            data: The batch to process
            pad_token_id: The ID of the padding token
            fields: List of field names to strip padding from. If None, strips from all 2D tensors
            anchor_field: The field to use for determining padding length
            max_strip_threshold: Maximum number of tokens to strip. If -1, no limit

        Returns:
            A new Batch with right padding removed
        """
        # If anchor_field not in data.values, return data as is
        if anchor_field not in data.values:
            return data

        anchor_tensor = data.values[anchor_field]

        # If not a tensor or not 2D, return as is
        if not isinstance(anchor_tensor, torch.Tensor) or len(anchor_tensor.shape) != 2:
            return data

        batch_size, seq_len = anchor_tensor.shape

        # Find right padding count for each sequence by flipping and counting from the start
        padding_mask = anchor_tensor == pad_token_id
        reversed_mask = padding_mask.flip(dims=[1])
        cumsum = (~reversed_mask).cumsum(dim=1)
        right_pad_counts = (cumsum == 0).sum(dim=1)

        # Find minimum padding count across all sequences
        min_pad_count = right_pad_counts.min().item()

        if min_pad_count == 0:
            return data

        # Apply max_strip_threshold if specified
        if max_strip_threshold >= 0:
            min_pad_count = min(min_pad_count, max_strip_threshold)

        # Create new values dict with padding stripped
        new_values = {}
        for key, value in data.values.items():
            # Only strip 2D tensors/arrays in specified fields (or all if fields is None)
            if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                if fields is None or key in fields:
                    new_values[key] = value[:, :-min_pad_count]
                else:
                    new_values[key] = value
            else:
                new_values[key] = value

        # Return new Batch with stripped values
        return Batch(values=new_values, metadata=data.metadata.copy())

    def to_nextrainer_batch(self) -> dict[str, Any]:
        """
        Prepare batch for NexTrainer
        """
        tensor_batch = {}
        non_tensor_batch = {}

        # Separate tensor and non-tensor values
        for key, value in self.values.items():
            if isinstance(value, torch.Tensor):
                tensor_batch[key] = value
            else:
                non_tensor_batch[key] = np.asarray(value, dtype=object)

        batch = {
            "batch": tensor_batch,
            "non_tensor_batch": non_tensor_batch,
            "meta_info": self.metadata,
        }
        return batch
