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

"""Train Service Backend.

This module provides the backend infrastructure for training with distributed workers.

DirectZMQTrainServiceClient is the production client that sends data directly to
workers via ZMQ while using HTTP for coordination with the API server.

Use `create_train_service_client()` from `nexrl.utils.init_utils` to create clients
based on configuration, or import directly for explicit control.
"""

from .api.direct_zmq_client import DirectZMQTrainServiceClient
from .fsdp_worker.fsdp_actor import DataParallelPPOActor
from .utils.core_utils import (
    CheckpointRequest,
    ConvertCheckpointRequest,
    DataProtoRequest,
    DataProtoResponse,
    NumpyData,
    SaveCheckpointRequest,
    StatusResponse,
    TensorData,
)
from .utils.protocol import DataProto

__all__ = [
    # Clients
    "DirectZMQTrainServiceClient",
    # Protocol
    "DataProto",
    # Worker
    "DataParallelPPOActor",
    # Data types
    "TensorData",
    "NumpyData",
    "DataProtoRequest",
    "DataProtoResponse",
    "CheckpointRequest",
    "SaveCheckpointRequest",
    "ConvertCheckpointRequest",
    "StatusResponse",
]

__version__ = "0.1.0"
