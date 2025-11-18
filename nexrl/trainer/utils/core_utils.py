# Copyright (c) Nex-AGI. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Utility functions for NexTrainer
"""

import base64
import copy
import heapq
import itertools
import json
import pickle
from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor, nn
from torch.distributed import ProcessGroup

# Optional imports for HTTP functionality
try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = lambda *args, **kwargs: None

# Import DataProto from protocol module

from .protocol import DataProto

if DataProto is None:
    try:
        from .protocol import DataProto
    except ImportError:
        pass

if DataProto is None:
    try:
        # Try to import from the same directory
        import os
        import sys

        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.dirname(current_dir))
        from .protocol import DataProto
    except ImportError:
        pass

# Import distributed utilities for backward compatibility
from .dist_utils import (
    allgather_dict_tensors,
    gather_heads_scatter_seq,
    gather_outpus_and_unpad,
    gather_seq_scatter_heads,
    ulysses_pad_and_slice_inputs,
)

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False

# Export list for the module
__all__ = [
    # Core utilities
    "append_to_dict",
    "gather_from_labels",
    "logprobs_from_logits",
    "entropy_from_logits",
    "masked_sum",
    "masked_mean",
    "masked_var",
    "masked_whiten",
    "get_reverse_idx",
    "rearrange_micro_batches",
    "compute_edge_entropy_loss",
    # HTTP utilities
    "TensorData",
    "NumpyData",
    "DataProtoRequest",
    "DataProtoResponse",
    "CheckpointRequest",
    "SaveCheckpointRequest",
    "ConvertCheckpointRequest",
    "StatusResponse",
    "tensor_to_data",
    "data_to_tensor",
    "numpy_to_data",
    "data_to_numpy",
    "prepare_data_proto_request",
    "process_data_proto_response",
    "Timer",
    # Re-exported from dist_utils for backward compatibility
    "gather_seq_scatter_heads",
    "gather_heads_scatter_seq",
    "ulysses_pad_and_slice_inputs",
    "gather_outpus_and_unpad",
    "allgather_dict_tensors",
]


# ==============================================================================
# Python Functional Utils
# ==============================================================================


def _is_json_serializable(obj) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def split_for_requests(payload: Mapping[Any, Any]) -> tuple[dict, dict]:
    """
    Split a dict into (clean, rejected) for JSON sending.
    - Keeps fully JSON-serializable entries.
    - Recursively splits nested dicts.
    - If a non-dict value isn't fully serializable, rejects the whole value.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("split_for_requests expects a dict-like mapping at the top level.")

    clean, rejected = {}, {}

    for k, v in payload.items():
        if _is_json_serializable(v):
            clean[k] = v
            continue

        # If it's a dict, try to salvage serializable subparts recursively.
        if isinstance(v, Mapping):
            sub_clean, sub_rejected = split_for_requests(v)
            if sub_clean:  # keep whatever is serializable
                clean[k] = sub_clean
            if sub_rejected:  # store what couldn't be serialized
                rejected[k] = sub_rejected
            # If both empty, nothing to keep; nothing to reject.
        else:
            # Non-dict and not serializable -> reject entire value
            rejected[k] = v

    return clean, rejected


def deep_merge(a: Mapping[Any, Any], b: Mapping[Any, Any]) -> dict:
    """
    Non-destructive deep merge: values in b take precedence.
    Merges recursively for dicts; replaces for non-dicts.
    """
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return b
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def restore_payload(clean: dict, rejected: dict) -> dict:
    """
    Reconstruct the original (best-effort) payload by deep-merging rejected keys back.
    Safer than dict.update() because it merges nested dicts instead of overwriting them wholesale.
    """
    if not isinstance(clean, Mapping) or not isinstance(rejected, Mapping):
        raise TypeError("restore_payload expects two dict-like mappings.")
    return deep_merge(clean, rejected)


def append_to_dict(data: dict[Any, Any], new_data: dict[Any, Any]):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


# ==============================================================================
# Torch Functional Utils
# ==============================================================================


def gather_from_labels(data, label):
    """Gather the label from data. The value in label should be [0, vocab_size)

    Args:
        data: (..., vocab_size)
        label (torch.IntTensor) : (...,)

    Returns:

    """

    output = torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
    return output


def logprobs_from_logits(logits, labels, inplace_backward=False):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels, inplace_backward)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels, inplace_backward):
    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(
        output, tuple
    ), "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    return -output[0]


def logprobs_from_logits_v2(logits: torch.FloatTensor, labels):
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis)


def masked_mean(values, mask, use_max_len_normalization=False, axis=None):
    """Compute mean of tensor with a masked values."""
    if mask.sum() == 0:
        return (values * mask).sum(axis=axis)  # should return 0, use * to keep tracking gradient

    if use_max_len_normalization:
        max_len = mask.shape[-1]
        return (values * mask).sum(axis=axis) / max_len
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if mask.sum() == 0:
        return variance
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


# ==============================================================================
# Sequence Length Balancing Utils
# ==============================================================================


def get_reverse_idx(indices):
    """Get reverse indices for reordering"""
    reverse_idx = [0] * len(indices)
    for i, idx in enumerate(indices):
        reverse_idx[idx] = i
    return reverse_idx


def rearrange_micro_batches(batch, max_token_len):
    """Rearrange micro batches based on sequence length"""
    # Simplified implementation - just split evenly
    batch_size = batch["input_ids"].shape[0]
    seq_len = batch["input_ids"].shape[1]

    # Calculate number of batches needed
    tokens_per_sample = seq_len
    samples_per_batch = max(1, max_token_len // tokens_per_sample)

    micro_batches = []
    indices = []

    for i in range(0, batch_size, samples_per_batch):
        end_idx = min(i + samples_per_batch, batch_size)
        micro_batch = {}
        for key, val in batch.items():
            micro_batch[key] = val[i:end_idx]
        micro_batches.append(micro_batch)
        indices.append(list(range(i, end_idx)))

    return micro_batches, indices


# ==============================================================================
# Entropy Monitor Utils
# ==============================================================================


@torch.no_grad()
def compute_edge_entropy_loss(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the entropy loss of the first 1000 tokens and the last 1000 tokens.
    """
    first_mask = torch.zeros_like(response_mask)
    last_mask = torch.zeros_like(response_mask)

    batch_size, _ = response_mask.size()
    for i in range(batch_size):
        valid_len = int(response_mask[i].sum().item())
        n = min(1000, valid_len)

        first_mask[i, :n] = 1
        start = max(0, valid_len - n)
        last_mask[i, start:valid_len] = 1

    first_mask *= response_mask
    last_mask *= response_mask

    device = entropy.device
    if first_mask.sum() > 0:
        first_entropy_loss = masked_mean(entropy, first_mask)
    else:
        first_entropy_loss = torch.tensor(0.0, device=device)

    if last_mask.sum() > 0:
        last_entropy_loss = masked_mean(entropy, last_mask)
    else:
        last_entropy_loss = torch.tensor(0.0, device=device)

    return first_entropy_loss, last_entropy_loss


# ==============================================================================
# HTTP Utilities for DataProto Serialization
# ==============================================================================


class TensorData(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore[misc]
    """Represents a PyTorch tensor in HTTP request/response

    This class handles serialization of PyTorch tensors using base64 encoding
    of pickled tensor data, along with shape and dtype metadata.
    """

    if PYDANTIC_AVAILABLE:
        # Pydantic field definitions
        data: str = Field(...)
        shape: list[int] = Field(...)
        dtype: str = Field(...)
    else:
        # Fallback for non-Pydantic
        def __init__(
            self,
            data: str | None = None,
            shape: list[int] | None = None,
            dtype: str | None = None,
            **kwargs,
        ):
            if data:
                self.data = data
            if shape:
                self.shape = shape
            if dtype:
                self.dtype = dtype

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorData":
        """Create TensorData from a PyTorch tensor

        Args:
            tensor: PyTorch tensor to serialize

        Returns:
            TensorData object with serialized tensor data
        """
        return cls(
            data=base64.b64encode(pickle.dumps(tensor.cpu())).decode("utf-8"),
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
        )

    def to_tensor(self) -> torch.Tensor:
        """Convert TensorData back to PyTorch tensor

        Returns:
            Deserialized PyTorch tensor
        """
        if self.data is None:
            raise ValueError("Cannot convert TensorData to tensor: data is None")
        tensor_bytes = base64.b64decode(self.data.encode("utf-8"))
        return pickle.loads(tensor_bytes)


class NumpyData(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore[misc]
    """Represents a numpy array in HTTP request/response

    This class handles serialization of numpy arrays using base64 encoding
    of pickled array data, along with shape and dtype metadata.
    """

    if PYDANTIC_AVAILABLE:
        # Pydantic field definitions
        data: str = Field(...)
        shape: list[int] = Field(...)
        dtype: str = Field(...)
    else:
        # Fallback for non-Pydantic
        def __init__(
            self,
            data: str | None = None,
            shape: list[int] | None = None,
            dtype: str | None = None,
            **kwargs,
        ):
            if data:
                self.data = data
            if shape:
                self.shape = shape
            if dtype:
                self.dtype = dtype

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "NumpyData":
        """Create NumpyData from a numpy array

        Args:
            array: Numpy array to serialize

        Returns:
            NumpyData object with serialized array data
        """
        return cls(
            data=base64.b64encode(pickle.dumps(array)).decode("utf-8"),
            shape=list(array.shape),
            dtype=str(array.dtype),
        )

    def to_numpy(self) -> np.ndarray:
        """Convert NumpyData back to numpy array

        Returns:
            Deserialized numpy array
        """
        if self.data is None:
            raise ValueError("Cannot convert NumpyData to numpy: data is None")
        array_bytes = base64.b64decode(self.data.encode("utf-8"))
        return pickle.loads(array_bytes)


class DataProtoRequest(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore[misc]
    """HTTP request model for DataProto-based endpoints

    This model represents a DataProto object in HTTP JSON format, with
    separate handling for tensor data (batch) and non-tensor data
    (non_tensor_batch) following the DataProto design pattern.
    """

    if PYDANTIC_AVAILABLE:
        # Pydantic field definitions
        batch: dict[str, TensorData] | None = Field(default=None)
        non_tensor_batch: dict[str, NumpyData] | None = Field(default=None)
        meta_info: dict[str, Any] | None = Field(default=None)
    else:
        # Fallback for non-Pydantic
        def __init__(
            self,
            batch: dict[str, TensorData] | None = None,
            non_tensor_batch: dict[str, NumpyData] | None = None,
            meta_info: dict[str, Any] | None = None,
            **kwargs,
        ):
            self.batch = batch
            self.non_tensor_batch = non_tensor_batch
            self.meta_info = meta_info

    def to_data_proto(self) -> "DataProto":
        """Convert HTTP request to DataProto object

        Returns:
            DataProto object for direct ZMQ transmission or local processing

        Raises:
            ImportError: If DataProto is not available
        """
        if DataProto is None:
            raise ImportError("DataProto not available")

        # Convert tensor data
        tensors = {}
        if self.batch:
            for key, tensor_data in self.batch.items():
                tensors[key] = tensor_data.to_tensor()

        # Convert numpy data
        non_tensors = {}
        if self.non_tensor_batch:
            for key, numpy_data in self.non_tensor_batch.items():
                non_tensors[key] = numpy_data.to_numpy()

        # Create TensorDict if we have tensors
        tensor_dict = None
        if tensors:
            # Infer batch size from first tensor
            first_tensor = next(iter(tensors.values()))
            batch_size = first_tensor.shape[:1]  # Assume first dim is batch
            tensor_dict = TensorDict(source=tensors, batch_size=batch_size)

        return DataProto(
            batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=self.meta_info or {}
        )


class DataProtoResponse(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore[misc]
    """HTTP response model for DataProto results

    This model represents a DataProto object in HTTP JSON format for responses,
    with proper serialization of both tensor and non-tensor components.
    """

    if PYDANTIC_AVAILABLE:
        # Pydantic field definitions
        batch: dict[str, TensorData] | None = Field(default=None)
        non_tensor_batch: dict[str, NumpyData] | None = Field(default=None)
        meta_info: dict[str, Any] | None = Field(default=None)
    else:
        # Fallback for non-Pydantic
        def __init__(
            self,
            batch: dict[str, TensorData] | None = None,
            non_tensor_batch: dict[str, NumpyData] | None = None,
            meta_info: dict[str, Any] | None = None,
            **kwargs,
        ):
            self.batch = batch
            self.non_tensor_batch = non_tensor_batch
            self.meta_info = meta_info

    @classmethod
    def from_data_proto(cls, data_proto: Any) -> "DataProtoResponse":
        """Create HTTP response from DataProto object

        Args:
            data_proto: DataProto object to serialize

        Returns:
            DataProtoResponse object for HTTP transmission
        """
        batch_data = {}
        if hasattr(data_proto, "batch") and data_proto.batch is not None:
            for key, tensor in data_proto.batch.items():
                batch_data[key] = TensorData.from_tensor(tensor)

        non_tensor_data = {}
        if hasattr(data_proto, "non_tensor_batch") and data_proto.non_tensor_batch:
            for key, array in data_proto.non_tensor_batch.items():
                non_tensor_data[key] = NumpyData.from_numpy(array)

        return cls(
            batch=batch_data if batch_data else None,
            non_tensor_batch=non_tensor_data if non_tensor_data else None,
            meta_info=getattr(data_proto, "meta_info", None),
        )


class CheckpointRequest(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore[misc]
    """Request model for checkpoint loading operations"""

    if PYDANTIC_AVAILABLE:
        # Pydantic field definitions
        path: str | None = Field(...)
        del_local_after_load: bool | None = Field(default=True)
        load_weight_only: bool | None = Field(default=False)

    def __init__(
        self,
        path: str | None = None,
        del_local_after_load: bool | None = True,
        load_weight_only: bool | None = False,
        **kwargs,
    ):
        if PYDANTIC_AVAILABLE:
            super().__init__(
                path=path,
                del_local_after_load=del_local_after_load,
                load_weight_only=load_weight_only,
                **kwargs,
            )
        else:
            self.path = path
            self.del_local_after_load = del_local_after_load
            self.load_weight_only = load_weight_only


class SaveCheckpointRequest(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore[misc]
    """Request model for checkpoint saving operations"""

    if PYDANTIC_AVAILABLE:
        # Pydantic field definitions
        local_path: str | None = Field(...)
        hdfs_path: str | None = Field(default=None)
        global_step: int | None = Field(default=0)
        saved_fully_shared_ckpt: bool | None = Field(default=True)
        save_weight_only: bool | None = Field(default=False)
        remove_previous_ckpt: bool | None = Field(default=True)

    def __init__(
        self,
        local_path: str | None = None,
        hdfs_path: str | None = None,
        global_step: int | None = 0,
        saved_fully_shared_ckpt: bool | None = True,
        save_weight_only: bool | None = False,
        remove_previous_ckpt: bool | None = True,
        **kwargs,
    ):
        if PYDANTIC_AVAILABLE:
            super().__init__(
                local_path=local_path,
                hdfs_path=hdfs_path,
                global_step=global_step,
                saved_fully_shared_ckpt=saved_fully_shared_ckpt,
                save_weight_only=save_weight_only,
                remove_previous_ckpt=remove_previous_ckpt,
                **kwargs,
            )
        else:
            self.local_path = local_path
            self.hdfs_path = hdfs_path
            self.global_step = global_step
            self.saved_fully_shared_ckpt = saved_fully_shared_ckpt
            self.save_weight_only = save_weight_only
            self.remove_previous_ckpt = remove_previous_ckpt


class ConvertCheckpointRequest(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore[misc]
    """Request model for checkpoint conversion operations"""

    if PYDANTIC_AVAILABLE:
        # Pydantic field definitions
        local_path: str | None = Field(...)

    def __init__(self, local_path: str | None = None, **kwargs):
        if PYDANTIC_AVAILABLE:
            super().__init__(local_path=local_path, **kwargs)
        else:
            self.local_path = local_path


class StatusResponse(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore[misc]
    """Response model for status and health check endpoints"""

    if PYDANTIC_AVAILABLE:
        # Pydantic field definitions
        status: str | None = Field(default=None)
        message: str | None = Field(default=None)
        workers_connected: int | None = Field(default=None)
        total_workers: int | None = Field(default=None)
        worker_initialized: bool | None = Field(default=None)
        worker_role: str | None = Field(default=None)

    def __init__(
        self,
        status: str | None = None,
        message: str | None = None,
        workers_connected: int | None = None,
        total_workers: int | None = None,
        worker_initialized: bool | None = None,
        worker_role: str | None = None,
        **kwargs,
    ):
        if PYDANTIC_AVAILABLE:
            super().__init__(
                status=status,
                message=message,
                workers_connected=workers_connected,
                total_workers=total_workers,
                worker_initialized=worker_initialized,
                worker_role=worker_role,
                **kwargs,
            )
        else:
            self.status = status
            self.message = message
            self.workers_connected = workers_connected
            self.total_workers = total_workers
            self.worker_initialized = worker_initialized
            self.worker_role = worker_role


# HTTP Utility functions for client-side serialization
def tensor_to_data(tensor: torch.Tensor) -> dict[str, Any]:
    """Convert tensor to serializable dictionary format

    Args:
        tensor: PyTorch tensor to convert

    Returns:
        Dictionary with serialized tensor data
    """
    return {
        "data": base64.b64encode(pickle.dumps(tensor.cpu())).decode("utf-8"),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def data_to_tensor(data: dict[str, Any]) -> torch.Tensor:
    """Convert serializable dictionary back to tensor

    Args:
        data: Dictionary with serialized tensor data

    Returns:
        Deserialized PyTorch tensor
    """
    tensor_bytes = base64.b64decode(data["data"].encode("utf-8"))
    return pickle.loads(tensor_bytes)


def numpy_to_data(array: np.ndarray) -> dict[str, Any]:
    """Convert numpy array to serializable dictionary format

    Args:
        array: Numpy array to convert

    Returns:
        Dictionary with serialized array data
    """
    return {
        "data": base64.b64encode(pickle.dumps(array)).decode("utf-8"),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def data_to_numpy(data: dict[str, Any]) -> np.ndarray:
    """Convert serializable dictionary back to numpy array

    Args:
        data: Dictionary with serialized array data

    Returns:
        Deserialized numpy array
    """
    array_bytes = base64.b64decode(data["data"].encode("utf-8"))
    return pickle.loads(array_bytes)


def prepare_data_proto_request(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Prepare DataProto request by converting tensors and numpy arrays

    Args:
        data: Dictionary containing batch, non_tensor_batch, and meta_info

    Returns:
        Dictionary ready for HTTP transmission
    """

    if isinstance(data, dict) and "batch" not in data.keys():
        data_ = {"batch": {}, "non_tensor_batch": {}, "meta_info": data.get("meta_info", {})}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data_["batch"][key] = tensor_to_data(value)
            elif isinstance(value, np.ndarray):
                data_["non_tensor_batch"][key] = numpy_to_data(value)
            else:
                data_["non_tensor_batch"][key] = value
        data = data_

    request_data = {"batch": {}, "non_tensor_batch": {}, "meta_info": data.get("meta_info", {})}

    # Convert tensors in batch
    if isinstance(data, dict):
        if "batch" in data and data["batch"]:
            for key, tensor in data["batch"].items():
                if isinstance(tensor, torch.Tensor):
                    request_data["batch"][key] = tensor_to_data(tensor)
                # else:
                #     # Handle non-tensor data in batch
                #     request_data["batch"][key] = tensor

        # Convert numpy arrays in non_tensor_batch
        if "non_tensor_batch" in data and data["non_tensor_batch"]:
            for key, array in data["non_tensor_batch"].items():
                if isinstance(array, np.ndarray):
                    request_data["non_tensor_batch"][key] = numpy_to_data(array)
                # elif isinstance(array, torch.Tensor):
                #     request_data["batch"][key] = tensor_to_data(array)
                # else:
                #     request_data["meta_info"][key] = array

    return request_data


def process_data_proto_response(response_data: dict[str, Any]) -> dict[str, Any]:
    """Process DataProto response by converting tensor and numpy data back

    Args:
        response_data: HTTP response data

    Returns:
        Dictionary with deserialized tensors and arrays
    """
    # Convert tensors in batch
    if "batch" in response_data and response_data["batch"]:
        for key, tensor_data in response_data["batch"].items():
            if isinstance(tensor_data, dict) and "data" in tensor_data:
                response_data["batch"][key] = data_to_tensor(tensor_data)

    # Convert numpy arrays in non_tensor_batch
    if "non_tensor_batch" in response_data and response_data["non_tensor_batch"]:
        for key, numpy_data in response_data["non_tensor_batch"].items():
            if isinstance(numpy_data, dict) and "data" in numpy_data:
                response_data["non_tensor_batch"][key] = data_to_numpy(numpy_data)

    return response_data


# ==============================================================================
# Timer Utilities
# ==============================================================================

import time


class Timer:
    """Simple timer context manager"""

    def __init__(self, name=None, logger=None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.last = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            self.last = int(time.time() - self.start_time)
            if self.name and self.logger:
                self.logger.info(f"{self.name}: {self.last:.4f}s")
            elif self.name:
                print(f"{self.name}: {self.last:.4f}s")
