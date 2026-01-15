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

"""NexTrainer utilities module"""

from .config_loader import (
    interpolate_config_values,
    load_config_with_references,
    load_nextrainer_config,
    resolve_config_references,
)
from .core_algos import (
    agg_loss,
    compute_policy_loss,
    compute_policy_loss_impl,
    compute_policy_loss_NX_20250515,
    kl_penalty,
)
from .core_utils import (
    CheckpointRequest,
    ConvertCheckpointRequest,
    DataProtoRequest,
    DataProtoResponse,
    NumpyData,
    SaveCheckpointRequest,
    StatusResponse,
    TensorData,
    Timer,
    allgather_dict_tensors,
    append_to_dict,
    compute_edge_entropy_loss,
    entropy_from_logits,
    gather_from_labels,
    gather_outpus_and_unpad,
    get_reverse_idx,
    logprobs_from_logits,
    masked_mean,
    masked_sum,
    masked_var,
    masked_whiten,
    rearrange_micro_batches,
    ulysses_pad_and_slice_inputs,
)
from .debug import log_gpu_memory_usage
from .dist_utils import (
    FSDPUlyssesShardingManager,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_ulysses_sequence_parallel_group,
    init_fn,
    set_ulysses_sequence_parallel_group,
)
from .model_utils import (
    FlopsCounter,
    PrecisionType,
    compute_position_id_with_mask,
    copy_local_path_from_hdfs,
    get_checkpoint_manager,
    get_constant_schedule_with_warmup,
    get_generation_config,
    hf_tokenizer,
    import_external_libs,
    print_model_size,
    set_pad_token_id,
    update_model_config,
)

# Imports from the moved core modules
from .protocol import DataProto, union_two_dict

# Megatron utilities removed


__all__ = [
    # Config loader functions
    "load_nextrainer_config",
    "load_config_with_references",
    "resolve_config_references",
    "interpolate_config_values",
    # Debug utilities
    "log_gpu_memory_usage",
    # FSDP utilities
    "FSDPUlyssesShardingManager",
    "get_ulysses_sequence_parallel_group",
    "set_ulysses_sequence_parallel_group",
    "get_fsdp_wrap_policy",
    "init_fn",
    "get_init_weight_context_manager",
    # Core functionality (moved from nextrainer.core)
    "DataProto",
    "union_two_dict",
    "append_to_dict",
    "gather_from_labels",
    "logprobs_from_logits",
    "entropy_from_logits",
    "masked_sum",
    "masked_mean",
    "masked_var",
    "masked_whiten",
    "ulysses_pad_and_slice_inputs",
    "gather_outpus_and_unpad",
    "get_reverse_idx",
    "rearrange_micro_batches",
    "compute_edge_entropy_loss",
    "allgather_dict_tensors",
    "Timer",
    "TensorData",
    "NumpyData",
    "DataProtoRequest",
    "DataProtoResponse",
    "CheckpointRequest",
    "SaveCheckpointRequest",
    "ConvertCheckpointRequest",
    "StatusResponse",
    "agg_loss",
    "compute_policy_loss_NX_20250515",
    "compute_policy_loss",
    "compute_policy_loss_impl",
    "kl_penalty",
    "PrecisionType",
    "hf_tokenizer",
    "set_pad_token_id",
    "get_generation_config",
    "print_model_size",
    "update_model_config",
    "copy_local_path_from_hdfs",
    "import_external_libs",
    "compute_position_id_with_mask",
    "FlopsCounter",
    "get_checkpoint_manager",
    "get_constant_schedule_with_warmup",
]
