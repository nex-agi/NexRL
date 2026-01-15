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
Distributed utilities including FSDP and Ulysses integration for NexTrainer
"""

import functools
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers.trainer_pt_utils import get_module_class_from_name

from .protocol import DataProto

# ==============================================================================
# Ulysses Sequence Parallel Utilities
# ==============================================================================

# Global variable to store the current sequence parallel group
_ULYSSES_SEQUENCE_PARALLEL_GROUP: torch.distributed.ProcessGroup | None = None


def set_ulysses_sequence_parallel_group(group: torch.distributed.ProcessGroup):
    """Set the ulysses sequence parallel group"""
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


def get_ulysses_sequence_parallel_group() -> torch.distributed.ProcessGroup | None:
    """Get the ulysses sequence parallel group"""
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


def get_ulysses_sequence_parallel_world_size(
    group: torch.distributed.ProcessGroup | None = None,
) -> int:
    """Get ulysses sequence parallel world size."""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return torch.distributed.get_world_size(group) if group else 1


def get_ulysses_sequence_parallel_rank(
    group: torch.distributed.ProcessGroup | None = None,
) -> int:
    """Get ulysses sequence parallel rank."""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return torch.distributed.get_rank(group) if group else 0


def _pad_tensor(x: torch.Tensor, dim: int, padding_size: int) -> torch.Tensor:
    """Add padding to tensor along specified dimension"""
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def _unpad_tensor(x: torch.Tensor, dim: int, padding_size: int) -> torch.Tensor:
    """Remove padding from tensor along specified dimension"""
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]


def slice_input_tensor(x: torch.Tensor, dim: int, padding: bool = True, group=None) -> torch.Tensor:
    """Slice input tensor for sequence parallel processing"""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)
    sp_rank = get_ulysses_sequence_parallel_rank()
    dim_size = x.size(dim)

    # pad before slice
    if padding and dim_size % sp_world_size:
        padding_size = sp_world_size - (dim_size % sp_world_size)
        x = _pad_tensor(x, dim, padding_size)

    # slice the input tensor
    parts = x.size(dim) // sp_world_size
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
    return x[slc].contiguous()


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    seq_world_size = dist.get_world_size(group)
    input_list = [
        t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


def all_gather_tensor(local_tensor: torch.Tensor, group=None, async_op: bool = False):
    """All-gather tensor across group"""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output


class SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> tuple[None, Tensor, None, None, None, None]:
        if ctx.async_op:
            input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous()
        else:
            input_t = grad_output[0]
        return (
            None,
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            None,
            None,
            None,
            None,
        )


class Gather(torch.autograd.Function):
    """Proper autograd function for gathering with gradient support"""

    @staticmethod
    def forward(ctx, group, local_tensor, gather_dim, grad_scaler=True, async_op=False):
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]  # store original size
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),
            None,
            None,
            None,
            None,
        )


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup | None = None,
) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    x = SeqAllToAll.apply(group, x, head_dim, seq_dim)
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = _unpad_tensor(x, seq_dim, padding_size)
    return x


def gather_heads_scatter_seq(
    x: Tensor, head_dim: int, seq_dim: int, group: ProcessGroup | None = None
) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


def ulysses_pad_and_slice_inputs(
    input_ids_rmpad: torch.Tensor, position_ids_rmpad: torch.Tensor = None, sp_size: int = 1
):
    """
    Pad and slice input_ids to be divisible by sp_size
    Pad position_ids to be divisible by sp_size.

    Note both input_ids_rmpad and position_ids_rmpad will be padded and sliced.

    The is the utility of pre-forward for ulysses sequence parallelism

    Args:
        input_ids_rmpad: shape of [bsz, seqlen]
        position_ids_rmpad: shape of [bsz, seqlen], where bsz must be 1
        sp_size (int): ulysses sequence parallelism size

    Returns:
        torch.Tensor: padded and sliced input_ids
        torch.Tensor: padded and sliced position_ids
        int: pad size
    """
    if position_ids_rmpad is not None:
        assert position_ids_rmpad.size(0) == 1
        assert input_ids_rmpad.size(1) == position_ids_rmpad.size(1)
    if sp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0
    _, total_seq_len = input_ids_rmpad.shape
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    if pad_size > 0:
        input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=0)
        if position_ids_rmpad is not None:
            pad_pos_ids = torch.arange(pad_size, device=position_ids_rmpad.device).unsqueeze(0)
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
    if position_ids_rmpad is not None:
        position_ids_rmpad = slice_input_tensor(position_ids_rmpad, dim=1, padding=False)
    return input_ids_rmpad, position_ids_rmpad, pad_size


def gather_outpus_and_unpad(
    x: torch.Tensor,
    gather_dim: int,
    unpad_dim: int | None = None,
    padding_size: int = 0,
    grad_scaler: bool = True,
    group=None,
):
    """
    Gather outputs from sequence parallel ranks and unpad
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_size = get_ulysses_sequence_parallel_world_size()
    if group == None:
        return x
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    if unpad_dim is not None:
        assert isinstance(padding_size, int), "padding size is not given or is not an integer"
        if padding_size == 0:
            return x
        x = _unpad_tensor(x, unpad_dim, padding_size)
    return x


def allgather_dict_tensors(tensors: dict[str, torch.Tensor] | TensorDict, size, group, dim=0):
    """
    Allgather tensors across distributed group.

    Args:
        tensors: Dictionary of tensors or TensorDict
        size: Number of processes in the group
        group: Distributed process group
        dim: Dimension to concatenate along

    Returns:
        Dictionary or TensorDict with allgathered tensors
    """
    if isinstance(tensors, TensorDict):
        is_tensor_dict = True
        tensors_as_dict = tensors.to_dict()
        original_batch_size = tensors.batch_size
    else:
        tensors_as_dict = tensors
        is_tensor_dict = False
        original_batch_size = None

    output = {}
    sorted_keys = sorted(tensors_as_dict.keys())
    for key in sorted_keys:
        val = tensors_as_dict[key]
        output[key] = [torch.empty_like(val) for _ in range(size)]
        torch.distributed.all_gather(output[key], val, group=group, async_op=False)
        output[key] = torch.cat(output[key], dim=dim)

    if is_tensor_dict and original_batch_size is not None:
        output = TensorDict(source=output, batch_size=original_batch_size[0] * size)

    return output


# ==============================================================================
# FSDP Wrap Policies and Initialization
# ==============================================================================


def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    """Get FSDP wrap policy for the module."""
    if config is None:
        config = {}

    if config.get("disable", False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )
    min_num_params = config.get("min_num_params", 0)
    auto_wrap_policy = None

    policies = []

    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:

        def lambda_policy_fn(module):
            if (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            ):
                return True
            return False

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy


def init_fn(x: torch.nn.Module):
    """FSDP parameter initialization function"""
    if not torch.distributed.get_rank() == 0:
        x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()
    return x


def get_init_weight_context_manager(use_meta_tensor=True, mesh=None):
    """Get initialization weight context manager"""
    try:
        from accelerate import init_empty_weights
    except ImportError:
        # Fallback if accelerate not available
        @contextmanager
        def init_empty_weights():
            with torch.device("meta"):
                yield

    @contextmanager
    def cpu_init_weights():
        with torch.device("cpu"):
            yield

    if use_meta_tensor:
        if mesh is None:
            init_context = (
                init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
            )
        else:
            init_context = (
                init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
            )
    else:
        init_context = cpu_init_weights
    return init_context


# ==============================================================================
# FSDP + Ulysses Integration
# ==============================================================================


class FSDPUlyssesShardingManager:
    """
    Sharding manager to support data resharding when using FSDP + Ulysses
    """

    def __init__(self, device_mesh: DeviceMesh):
        self.device_mesh = device_mesh
        self.seed_offset = 12345

    def __enter__(self):
        if self.device_mesh is not None:
            # We have a global SP group
            # so we have to change to use model-specific sp group
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())

    def __exit__(self, exc_type, exc_value, traceback):
        # restore random states
        if self.device_mesh is not None:
            # revert to previous sp group
            set_ulysses_sequence_parallel_group(self.prev_sp_group)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        AllGather data from sp region
        This is because the data is first sharded along the FSDP dimension as we utilize the DP_COMPUTE
        In Ulysses, we need to make sure the same data is used across a SP group
        """
        if self.device_mesh is not None:
            sp_size = self.device_mesh["sp"].size()
            group = self.device_mesh["sp"].get_group()

            prev_device = data.batch.device
            data.batch = data.batch.cuda(device=torch.cuda.current_device())
            data.batch = allgather_dict_tensors(
                data.batch.contiguous(), size=sp_size, group=group, dim=0
            )
            data.batch = data.batch.to(prev_device)

            # all gather non_tensor_batch
            all_non_tensor_batch: list[dict[str, Any]] = [None for _ in range(sp_size)]  # type: ignore
            torch.distributed.all_gather_object(
                all_non_tensor_batch, data.non_tensor_batch, group=group
            )
            data.non_tensor_batch = {
                k: np.concatenate([d[k] for d in all_non_tensor_batch])
                for k in data.non_tensor_batch
            }
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """
        Split the data to follow FSDP partition
        """
        if self.device_mesh is not None:
            sp_size = self.device_mesh["sp"].size()
            sp_rank = self.device_mesh["sp"].get_local_rank()
            data = data.chunk(chunks=sp_size)[sp_rank]
        return data


# Note: FSDP offloading functions removed - not implemented in this version
