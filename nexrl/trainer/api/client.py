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
ActorWorker HTTP Client

This module provides a client class for interacting with the ActorWorker HTTP service.

Each client instance is bound to a specific worker group via an immutable identifier.
Create separate client instances for different worker groups.

Example:
    >>> # Single worker group
    >>> client = ActorWorkerClient("http://localhost:8000", identifier="nexrl0")
    >>> client.save_checkpoint("/path/to/ckpt")  # No need to pass identifier

    >>> # Multiple worker groups - use separate clients
    >>> client1 = ActorWorkerClient("http://localhost:8000", identifier="group1")
    >>> client2 = ActorWorkerClient("http://localhost:8000", identifier="group2")
"""

from contextlib import contextmanager
from typing import Any

import numpy as np
import requests
import torch

from ...train_service_client import TrainServiceClient

# Import utils for serialization utilities
from ..utils.core_utils import (
    data_to_numpy,
    data_to_tensor,
    numpy_to_data,
    prepare_data_proto_request,
    process_data_proto_response,
    restore_payload,
    split_for_requests,
    tensor_to_data,
)
from ..utils.protocol import DataProto


class ActorWorkerClient(TrainServiceClient):
    """Client for interacting with ActorWorker HTTP service"""

    def __init__(self, base_url: str = "http://localhost:8000", identifier: str | None = None):
        """Initialize the client

        Args:
            base_url: Base URL of the API server
            identifier: Worker group identifier. If None, the client will work with
                       the only/first group (useful for single-group setups).
                       Once set, the identifier is immutable for this client instance.
        """
        super().__init__(base_url, identifier)
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        # Set reasonable timeout for requests (passed to individual request methods)
        self.request_timeout = 200

    @property
    def identifier(self) -> str | None:
        """Get the immutable identifier for this client"""
        return self._identifier

    # Use serialization utilities from the serialization module
    def _tensor_to_data(self, tensor: torch.Tensor) -> dict[str, Any]:
        """Convert tensor to serializable format"""
        return tensor_to_data(tensor)

    def _data_to_tensor(self, data: dict[str, Any]) -> torch.Tensor:
        """Convert serializable format back to tensor"""
        return data_to_tensor(data)

    def _numpy_to_data(self, array: np.ndarray) -> dict[str, Any]:
        """Convert numpy array to serializable format"""
        return numpy_to_data(array)

    def _data_to_numpy(self, data: dict[str, Any]) -> np.ndarray:
        """Convert serializable format back to numpy array"""
        return data_to_numpy(data)

    def _prepare_data_proto_request(self, data: dict[str, Any]) -> dict[str, Any]:
        """Prepare DataProto request by converting tensors and numpy arrays"""
        return prepare_data_proto_request(data)

    def _process_data_proto_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Process DataProto response by converting tensor and numpy data back"""
        return process_data_proto_response(response_data)

    def health_check(self) -> dict[str, Any]:
        """Check service health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_models(self) -> dict[str, Any]:
        """Get all available worker group identifiers

        Returns:
            Dict with 'identifiers' list and 'models' info dict
        """
        response = self.session.get(f"{self.base_url}/get_models")
        response.raise_for_status()
        return response.json()

    def worker_info(self) -> dict[str, Any]:
        """Get worker information for this client's worker group"""
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.get(f"{self.base_url}/worker_info", params=params)
        response.raise_for_status()
        return response.json()

    def initialize_worker(
        self,
        config_path: str | None = None,
        config_dict: dict[str, Any] | None = None,
        role: str = "actor",
        world_size: int | None = None,
        zmq_base_port: int | None = None,
        dispatch_mode: str | None = None,
    ) -> dict[str, Any]:
        """Initialize the worker

        Args:
            config_path: Path to YAML configuration file
            config_dict: Optional dictionary configuration (overrides config_path)
            role: Worker role (e.g., 'actor', 'critic', 'reward')
            world_size: Number of workers (required for new worker groups)
            zmq_base_port: Base port for ZMQ (optional, auto-assigned if not provided)
            dispatch_mode: Dispatch mode (optional, defaults to 'broadcast')
        """
        request_data: dict[str, Any] = {"role": role}

        if config_path:
            request_data["config_path"] = config_path
        elif config_dict:
            request_data["config_dict"] = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        # Add identifier and worker group parameters
        if self.identifier:
            request_data["identifier"] = self.identifier
        if world_size:
            request_data["world_size"] = world_size
        if zmq_base_port:
            request_data["zmq_base_port"] = zmq_base_port
        if dispatch_mode:
            request_data["dispatch_mode"] = dispatch_mode

        response = self.session.post(f"{self.base_url}/initialize", json=request_data)
        response.raise_for_status()
        return response.json()

    def init_model(self) -> dict[str, Any]:
        """Initialize the model (background task)"""
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.post(f"{self.base_url}/init_model", params=params)
        response.raise_for_status()
        return response.json()

    def update_actor(self, data: dict[str, Any]) -> dict[str, Any]:
        """Update actor policy

        Args:
            data: Data to send to workers
        """
        request_data = self._prepare_data_proto_request(data)
        serializable_metadata, rejected_metadata = split_for_requests(request_data["meta_info"])
        request_data["meta_info"] = serializable_metadata

        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.post(
            f"{self.base_url}/update_actor", json=request_data, params=params
        )
        response.raise_for_status()
        ret = self._process_data_proto_response(response.json())
        ret["meta_info"] = restore_payload(ret["meta_info"], rejected_metadata)
        return ret

    def compute_log_prob(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute log probabilities

        Args:
            data: Data to send to workers
        """
        request_data = self._prepare_data_proto_request(data)
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.post(
            f"{self.base_url}/compute_log_prob", json=request_data, params=params
        )
        response.raise_for_status()
        return self._process_data_proto_response(response.json())

    def compute_ref_log_prob(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute reference log probabilities

        Args:
            data: Data to send to workers
        """
        request_data = self._prepare_data_proto_request(data)
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.post(
            f"{self.base_url}/compute_ref_log_prob", json=request_data, params=params
        )
        response.raise_for_status()
        return self._process_data_proto_response(response.json())

    def load_checkpoint(
        self, path: str, del_local_after_load: bool = True, load_weight_only: bool = False
    ) -> dict[str, Any]:
        """Load checkpoint

        Args:
            path: Path to checkpoint
            del_local_after_load: Whether to delete local checkpoint after loading
            load_weight_only: Whether to load weights only
        """
        request_data = {
            "path": path,
            "del_local_after_load": del_local_after_load,
            "load_weight_only": load_weight_only,
        }
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.post(
            f"{self.base_url}/load_checkpoint", json=request_data, params=params
        )
        response.raise_for_status()
        return response.json()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        saved_fully_shared_ckpt: bool = True,
        save_weight_only: bool = False,
        remove_previous_ckpt: bool = True,
    ) -> dict[str, Any]:
        """Save checkpoint

        Args:
            local_path: Local path to save checkpoint
            hdfs_path: HDFS path to save checkpoint (optional)
            global_step: Global step number
            saved_fully_shared_ckpt: Whether to save fully shared checkpoint
            save_weight_only: Whether to save weights only
            remove_previous_ckpt: Whether to remove previous checkpoint
        """
        request_data = {
            "local_path": local_path,
            "hdfs_path": hdfs_path,
            "global_step": global_step,
            "saved_fully_shared_ckpt": saved_fully_shared_ckpt,
            "save_weight_only": save_weight_only,
            "remove_previous_ckpt": remove_previous_ckpt,
        }
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.post(
            f"{self.base_url}/save_checkpoint", json=request_data, params=params
        )
        response.raise_for_status()
        return response.json()

    def convert_checkpoint_to_huggingface(self, local_path: str) -> dict[str, Any]:
        """Convert checkpoint to HuggingFace format

        Args:
            local_path: Local path to checkpoint
        """
        request_data = {"local_path": local_path}
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.post(
            f"{self.base_url}/convert_checkpoint_to_huggingface", json=request_data, params=params
        )
        response.raise_for_status()
        return response.json()

    def barrier(self) -> dict[str, Any]:
        """Execute synchronization barrier"""
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.post(f"{self.base_url}/barrier", params=params)
        response.raise_for_status()
        return response.json()

    @contextmanager
    def actor_context(self):
        """Context manager for actor model GPU loading

        Loads the actor model to GPU on entry and offloads it to CPU on exit.

        Example:
            >>> with client.actor_context():
            ...     # Actor is loaded on GPU
            ...     # Perform operations with actor
            ...     pass
            ... # Actor is offloaded to CPU
        """
        yield

    def __getattr__(self, name: str) -> Any:
        """Override attribute access to fetch from API server coordinator

        When an attribute is not found in the client, this method will
        send a request to the API server to get the attribute from the
        ZMQ coordinator.

        Args:
            name: Name of the attribute to fetch

        Returns:
            The attribute value from the coordinator

        Raises:
            AttributeError: If the attribute doesn't exist on coordinator or request fails
        """
        # Avoid infinite recursion for internal attributes
        if name.startswith("_") or name in ["session", "base_url"]:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        try:
            params = {"identifier": self.identifier} if self.identifier else {}
            response = self.session.get(f"{self.base_url}/getattr/{name}", params=params)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise AttributeError(f"Server error: {result['error']}")

            return result.get("value")

        except requests.exceptions.RequestException as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' - failed to fetch from server: {e}"
            )
        except Exception as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' - error: {e}"
            )
