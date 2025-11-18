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
NexRL - A large-scale reinforcement learning training framework
"""

# Configure logging BEFORE importing any modules


from .algorithm_processor import BaseAlgorithmProcessor
from .base_module import NexRLModule
from .controller import NexRLController
from .data_loader import BaseDataLoader
from .nexrl_types import Batch, ModelTag, Trajectory
from .rollout_worker import AgentRolloutWorker, BaseRolloutWorker, SimpleRolloutWorker
from .train_batch_pool import TrainBatchPool
from .train_worker import TrainWorker
from .trajectory_pool import TrajectoryPool
from .weight_sync.weight_sync_controller import WeightSyncController

__version__ = "0.1.0"

__all__ = [
    "NexRLController",
    "NexRLModule",
    "BaseRolloutWorker",
    "AgentRolloutWorker",
    "SimpleRolloutWorker",
    "TrajectoryPool",
    "BaseAlgorithmProcessor",
    "TrainBatchPool",
    "TrainWorker",
    "WeightSyncController",
    "BaseDataLoader",
    "Batch",
    "Trajectory",
    "ControllerMetadata",
    "ModelTag",
]
