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
Apply monkey-patch function to models
"""

import os
import sys
from typing import Any

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import PreTrainedModel

from .core_utils import gather_heads_scatter_seq, gather_seq_scatter_heads
from .dist_utils import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    position_ids: torch.Tensor | None = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"
        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # TODO: all_gather position_ids because `prepare_fa2_from_position_ids` needs it, we can eliminate
        # this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly.
        # https://github.com/huggingface/transformers/pull/33932

        # (bsz, seq_len/n) -> (bsz, seq_len)
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(
            position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group()
        )
        position_ids = torch.concat(position_ids_list, dim=-1)

    # (bsz, seq_len, n_head/n, head_dim)
    attn_output = _flash_attention_forward(
        query_states, key_states, value_states, *args, position_ids=position_ids, **kwargs
    )

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


def _ring_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    position_ids: torch.Tensor | None = None,
    **kwargs,
):
    """Ring flash attention forward pass with ring sequence parallelism.

    This function implements ring-attention for flash attention, which uses a ring topology
    for sequence parallelism instead of all-to-all communication. It performs distributed
    attention computation across sequence-parallel ranks using the ring-flash-attn library.

    Args:
        query_states (torch.Tensor): Query tensor with shape (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): Key tensor with shape (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): Value tensor with shape (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): Position IDs tensor with shape (batch_size, seqlen/sp_size).
            Required for preparing flash attention cu_seqlens and max_seq_lens.

    Returns:
        torch.Tensor: Attention output tensor with shape (batch_size, seqlen/sp_size, nheads, head_dim)

    Raises:
        ImportError: If `ring_flash_attn` module or required transformers utilities are not available
    """

    try:
        from ring_flash_attn import (
            llama3_flash_attn_prepare_cu_seqlens,
            llama3_flash_attn_varlen_func,
        )
    except Exception as e:
        raise ImportError(
            "Failed to import `ring_flash_attn`. Please check the GPU worker image version "
            "and ensure `ring-flash-attn` is properly installed in the image. "
            "This package should be included in the GPU worker Docker image."
        ) from e

    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    sp_group = get_ulysses_sequence_parallel_group()
    import torch.distributed as dist

    sp_world_size = dist.get_world_size(sp_group)
    sp_rank = get_ulysses_sequence_parallel_rank()

    # Sigh... Transformers keeps reshuffling these internal helpers and breaking BC.
    # `prepare_fa2_from_position_ids` is deprecated and use `_prepare_flash_attention_from_position_ids` in this pr https://github.com/huggingface/transformers/pull/39474/files
    # `_prepare_flash_attention_from_position_ids` is deprecated and use `_prepare_from_posids` in this pr https://github.com/huggingface/transformers/pull/40069/files
    try:
        from transformers.modeling_flash_attention_utils import prepare_fa2_from_position_ids
    except ImportError:
        try:
            from transformers.modeling_flash_attention_utils import (
                _prepare_flash_attention_from_position_ids,
            )

            prepare_fa2_from_position_ids = _prepare_flash_attention_from_position_ids
        except ImportError:
            try:
                from transformers.modeling_flash_attention_utils import _prepare_from_posids

                prepare_fa2_from_position_ids = _prepare_from_posids
            except ImportError as e:
                raise ImportError(
                    "Cannot import `prepare_fa2_from_position_ids` from Transformers.\n"
                    "Cannot import `_prepare_flash_attention_from_position_ids` from Transformers.\n"
                    "Cannot import `_prepare_from_posids` from Transformers.\n"
                    "Your installed `transformers` version likely doesn't provide it.\n"
                    "Fix options:\n"
                    "- Install a compatible Transformers version (see repo pyproject.toml pin), or\n"
                    "- Disable ring sequence parallel (use_ring_sequence_parallel=False).\n"
                ) from e

    position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
    torch.distributed.all_gather(
        position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group()
    )
    position_ids = torch.concat(position_ids_list, dim=-1)
    try:
        query_states, key_states, value_states, _, cu_seq_lens, max_seq_lens = (
            prepare_fa2_from_position_ids(query_states, key_states, value_states, position_ids)
        )
    except Exception as first_error:
        try:
            query_states, key_states, value_states, cu_seq_lens, max_seq_lens = (
                prepare_fa2_from_position_ids(query_states, key_states, value_states, position_ids)
            )
        except Exception as second_error:
            raise ValueError(
                f"Failed to unpack return values from `prepare_fa2_from_position_ids`. "
                f"This function's return signature varies across transformers versions. "
                f"Expected either 5 or 6 values, but got incompatible format.\n"
                f"First attempt (6-value unpacking) error: {first_error}\n"
                f"Second attempt (5-value unpacking) error: {second_error}\n"
                f"This may indicate an unsupported transformers version. "
                f"Please check if your transformers version is compatible with ring sequence parallelism."
            ) from second_error
    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    max_seq_lens_q, max_seq_lens_k = max_seq_lens

    (
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        local_k_slice,
    ) = llama3_flash_attn_prepare_cu_seqlens(
        cu_seqlens_q,
        causal=True,
        rank=sp_rank,
        world_size=sp_world_size,
    )

    ring_allgather_heads_k_stride = 1

    attn_output = llama3_flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        heads_k_stride=ring_allgather_heads_k_stride,
        local_k_slice=local_k_slice,
        causal=True,
        dropout_p=0.0,
        return_attn_probs=False,
        group=sp_group,
    )
    return attn_output


def apply_monkey_patch(model: PreTrainedModel, use_ring_sequence_parallel: bool = False):
    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""
    module = sys.modules[model.__module__]

    # transformers<=4.47.1
    if hasattr(module, "_flash_attention_forward"):
        if use_ring_sequence_parallel:
            setattr(module, "_flash_attention_forward", _ring_flash_attention_forward)
            print(
                f"Monkey patch _flash_attention_forward _ring_flash_attention_forward in {module.__name__}"
            )
        else:
            setattr(module, "_flash_attention_forward", _ulysses_flash_attention_forward)
            print(
                f"Monkey patch _flash_attention_forward _ulysses_flash_attention_forward in {model.__module__}"
            )
    else:
        # transformers>=4.48.0
        from transformers.integrations import flash_attention

        if use_ring_sequence_parallel:
            flash_attention._flash_attention_forward = _ring_flash_attention_forward
            print(
                f"Monkey patch _flash_attention_forward _ring_flash_attention_forward in {flash_attention.__name__}"
            )
        else:
            flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
            print(
                f"Monkey patch _flash_attention_forward _ulysses_flash_attention_forward in {flash_attention.__name__}"
            )
