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


"""Distributed training components for NexTrainer."""

from ..fsdp_worker.fsdp_actor import DataParallelPPOActor
from ..utils import (
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
)
from ..utils.dist_utils import (
    FSDPUlyssesShardingManager,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)

# from ..fsdp_worker.fsdp_workers import ModelWorker  # Commented out to avoid circular import
# Note: worker_process is not imported here to avoid conflicts when running as __main__
# It's meant to be run as a standalone script: python -m nexrl.trainer.distributed.worker_process

__all__ = [
    "DataParallelPPOActor",
    "FSDPUlyssesShardingManager",
    "get_ulysses_sequence_parallel_group",
    "set_ulysses_sequence_parallel_group",
    "get_fsdp_wrap_policy",
    "init_fn",
    "get_init_weight_context_manager",
]
