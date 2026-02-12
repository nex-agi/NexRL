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

from ..train_service_client import TrainServiceClient  # pylint: disable=relative-beyond-top-level


def create_train_service_client(
    backend: str,
    url: str,
    identifier: str | None = None,
    **kwargs,
) -> TrainServiceClient:
    """
    Create a train service client based on the backend configuration

    Args:
        backend: The backend type ('direct-zmq', 'http', 'nextrainer', 'mock')
        url: The service URL
        identifier: Optional identifier for the worker group
        **kwargs: Additional arguments:
            - config: Optional dict with client configuration (e.g., timeouts)
            - tinker_service_holder: For Tinker backend
            - weaver_service_holder: For Weaver backend

    Returns:
        A TrainServiceClient instance

    Raises:
        NotImplementedError: If the backend is not supported

    Backend options:
        - 'direct-zmq' (RECOMMENDED): Uses direct ZMQ connections to workers for data
                   operations, HTTP to API server for coordination only. This is
                   the production backend that:
                   - Eliminates API server OOM (data never lands there)
                   - Reduces network hops (single serialization)
                   - Supports two-phase protocol for NCCL timing
        - 'http' or 'nextrainer': Now mapped to 'direct-zmq' for backward compatibility.
                   All operations use DirectZMQ client.
        - 'mock': Mock client for testing
    """
    # Map 'http' and 'nextrainer' to 'direct-zmq' for backward compatibility
    if backend in ("nextrainer", "http"):
        import warnings

        warnings.warn(
            f"Backend '{backend}' is deprecated and now uses DirectZMQTrainServiceClient. "
            f"Please update your config to use backend='direct-zmq' explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        backend = "direct-zmq"

    if backend == "direct-zmq":
        from ..train_service_backend.api.direct_zmq_client import (  # pylint: disable=relative-beyond-top-level
            DirectZMQTrainServiceClient,
        )

        # Extract optional config parameters
        config = kwargs.get("config", {})
        zmq_recv_timeout_ms = config.get("zmq_recv_timeout_ms", 600000)  # 10 minutes
        zmq_total_timeout_ms = config.get("zmq_total_timeout_ms", 3600000)  # 60 minutes

        return DirectZMQTrainServiceClient(
            base_url=url,
            identifier=identifier,
            zmq_recv_timeout_ms=zmq_recv_timeout_ms,
            zmq_total_timeout_ms=zmq_total_timeout_ms,
        )
    elif backend == "mock":
        from ..mock.mock_train_service_client import (  # pylint: disable=relative-beyond-top-level
            MockTrainServiceClient,
        )

        return MockTrainServiceClient(url, identifier)
    else:
        raise NotImplementedError(f"Train service backend {backend} not implemented")


def create_inference_service_client(backend: str, config, **kwargs):
    """
    Create an inference service client based on the backend configuration

    Args:
        backend: The backend type ('openai', 'vllm', 'sglang', 'tinker')
        config: Configuration for the LLM service
        **kwargs: Additional arguments (e.g., tinker_service_holder for Tinker)

    Returns:
        An InferenceServiceClient instance

    Raises:
        NotImplementedError: If the backend is not supported
    """
    if backend in ("openai", "vllm", "sglang"):
        from ..inference_service_client import (  # pylint: disable=relative-beyond-top-level
            OpenAIInferenceServiceClient,
        )

        return OpenAIInferenceServiceClient(config)
    elif backend == "tinker":
        from ..tinker import (  # pylint: disable=relative-beyond-top-level
            TinkerInferenceServiceClient,
        )

        return TinkerInferenceServiceClient(
            config=config,
            tinker_service_holder=kwargs.get("tinker_service_holder"),  # type: ignore
        )
    elif backend == "weaver":
        from ..weaver import (  # pylint: disable=relative-beyond-top-level
            WeaverInferenceServiceClient,
        )

        return WeaverInferenceServiceClient(
            config=config,
            weaver_service_holder=kwargs.get("weaver_service_holder"),  # type: ignore
        )
    else:
        raise NotImplementedError(f"Inference service backend {backend} not implemented")
