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

"""API components for train service backend.

This module provides the DirectZMQTrainServiceClient for interacting with the
train service. It uses direct ZMQ connections to workers for data operations
(update_actor, compute_log_prob, etc.), while using HTTP to API server for
coordination (commands, health checks, worker registration).
"""

from .direct_zmq_client import DirectZMQTrainServiceClient

__all__ = ["DirectZMQTrainServiceClient"]
