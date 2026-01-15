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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

# Type aliases for better readability
ModelTag = str  # Type alias for model tags


@dataclass
class Trajectory:
    """
    A trajectory dataclass representing a single rollout episode.

    Required fields:
        tokens: List of token IDs for the full sequence (prompt + response)
        loss_mask: List of integers (0/1) indicating which tokens to compute loss on
        reward: Float reward value for the trajectory
        is_val: Boolean indicating if this is a validation trajectory

    Additional fields can be added as needed via the extra_fields dict.
    """

    tokens: list[int]
    loss_mask: list[int]
    reward: float
    is_val: bool
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for backwards compatibility."""
        if key == "tokens":
            return self.tokens
        elif key == "loss_mask":
            return self.loss_mask
        elif key == "reward":
            return self.reward
        elif key == "is_val":
            return self.is_val
        elif key == "extra_fields":
            return self.extra_fields
        elif key in self.extra_fields:
            return self.extra_fields[key]
        else:
            raise KeyError(f"Key '{key}' not found in Trajectory")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like setting for backwards compatibility."""
        if key == "tokens":
            self.tokens = value
        elif key == "loss_mask":
            self.loss_mask = value
        elif key == "reward":
            self.reward = value
        elif key == "is_val":
            self.is_val = value
        elif key == "extra_fields":
            self.extra_fields = value
        else:
            self.extra_fields[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method with default value."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Return all keys (required + extra_fields)."""
        return ["tokens", "loss_mask", "reward", "is_val"] + list(self.extra_fields.keys())

    def items(self):
        """Return all items (required + extra_fields)."""
        base_items = [
            ("tokens", self.tokens),
            ("loss_mask", self.loss_mask),
            ("reward", self.reward),
            ("is_val", self.is_val),
        ]
        return base_items + list(self.extra_fields.items())

    def __contains__(self, key: str) -> bool:
        """Check if key exists in trajectory."""
        return key in ["tokens", "loss_mask", "reward", "is_val"] or key in self.extra_fields


class NexRLRole(Enum):
    """Define different roles in NexRL that can be mapped to resource pools."""

    ROLLOUT_WORKER = "rollout_worker"
    TRAINER = "trainer"
    TRAJECTORY_POOL = "trajectory_pool"
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

    def pad_to_world_size(self, world_size: int) -> "Batch":
        """Pad the batch to be divisible by world_size"""
        batch_size = len(self)
        if batch_size % world_size != 0:
            pad_size = world_size - (batch_size % world_size)
            for key, value in self.values.items():
                if isinstance(value, torch.Tensor):
                    self.values[key] = torch.cat([value, value[:pad_size]], dim=0)
                else:
                    # For non-tensor data, append directly
                    if isinstance(value, list):
                        self.values[key] = value + value[:pad_size]
                    else:
                        # numpy array or other iterable
                        value_array = np.asarray(value, dtype=object)
                        padding = value_array[:pad_size]
                        self.values[key] = np.concatenate([value_array, padding], axis=0)
            self.metadata["batch_size"] = batch_size + pad_size
        return self

    @classmethod
    def from_trajectories(
        cls, trajectories: list["Trajectory"], model_tag: "ModelTag | None" = None
    ) -> "Batch":
        """
        Create a Batch from a list of trajectories.

        Args:
            trajectories: List of Trajectory dataclass instances
            model_tag: Optional model tag for the batch metadata

        Returns:
            A Batch object containing the trajectories
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
                        f"Trajectory at index {i} has different keys. "
                        f"Expected: {first_keys}, Got: {current_keys}"
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
                    batch_size, _, _ = stacked.shape
                    values[key] = stacked.view(batch_size, -1)
                else:
                    # Mixed dimensions or higher dimensions, raise error
                    raise ValueError(f"Tensors for key '{key}' have inconsistent dimensions")

        # Determine model_tag from trajectories if not provided
        if model_tag is None:
            model_tag = trajectories[0].get("model_tag", "default")

        # Add batch_size to the metadata
        metadata = {
            "batch_size": len(trajectories),
            "model_tag": model_tag,
            "temperature": trajectories[0].get("temperature", 1.0),
        }

        return cls(values=values, metadata=metadata)

    def to_trajectories(self) -> list["Trajectory"]:
        """
        Convert a Batch back to a list of trajectories.

        Returns:
            List of Trajectory dataclass instances
        """
        batch_size = len(self)
        trajectories: list[Trajectory] = []

        for i in range(batch_size):
            # Extract required fields
            tokens = None
            loss_mask = None
            reward = None
            is_val = None
            extra_fields = {}

            for key, value in self.values.items():
                extracted_value = None
                if isinstance(value, torch.Tensor):
                    # Extract the i-th row from the tensor
                    extracted_value = value[i]
                elif isinstance(value, (list, np.ndarray)):
                    # Extract the i-th element from the list/array
                    extracted_value = value[i]
                else:
                    # For scalar values, copy as-is (same for all trajectories)
                    extracted_value = value

                # Assign to required fields or extra_fields
                if key == "tokens":
                    tokens = extracted_value
                elif key == "loss_mask":
                    loss_mask = extracted_value
                elif key == "reward":
                    reward = extracted_value
                elif key == "is_val":
                    is_val = extracted_value
                else:
                    extra_fields[key] = extracted_value

            # Add metadata fields to extra_fields
            if "model_tag" in self.metadata:
                extra_fields["model_tag"] = self.metadata["model_tag"]
            if "temperature" in self.metadata:
                extra_fields["temperature"] = self.metadata["temperature"]

            # Create Trajectory dataclass instance
            trajectory = Trajectory(
                tokens=tokens if tokens is not None else [],
                loss_mask=loss_mask if loss_mask is not None else [],
                reward=reward if reward is not None else 0.0,
                is_val=is_val if is_val is not None else False,
                extra_fields=extra_fields,
            )
            trajectories.append(trajectory)

        return trajectories

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

        # Find right padding count for each sequence by flipping and counting from the start
        padding_mask = anchor_tensor == pad_token_id
        reversed_mask = padding_mask.flip(dims=[1])
        cumsum = (~reversed_mask).cumsum(dim=1)
        right_pad_counts = (cumsum == 0).sum(dim=1)

        # Find minimum padding count across all sequences
        min_pad_count = int(right_pad_counts.min().item())

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
