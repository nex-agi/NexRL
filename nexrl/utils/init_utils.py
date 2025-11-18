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

from ..train_service_client import TrainServiceClient


def create_train_service_client(
    backend: str, url: str, identifier: str | None = None
) -> TrainServiceClient:
    """
    Create a train service client based on the backend configuration

    Args:
        backend: The backend type ('nextrainer' or 'mock')
        url: The service URL
        identifier: Optional identifier for the worker group

    Returns:
        An ActorWorkerClient instance

    Raises:
        NotImplementedError: If the backend is not supported
    """
    if backend == "nextrainer":
        from ..trainer.api.client import ActorWorkerClient

        return ActorWorkerClient(url, identifier)
    elif backend == "mock":
        from ..mock.mock_train_service_client import MockActorWorkerClient

        return MockActorWorkerClient(url, identifier)
    else:
        raise NotImplementedError(f"Train service backend {backend} not implemented")
