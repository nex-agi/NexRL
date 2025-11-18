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
Train Service Client Factory

This module provides a factory function to create train service clients
based on the backend configuration.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any


class TrainServiceClient(ABC):
    """
    Base class for train service clients

    This abstract base class defines the interface that all train service clients
    must implement, whether they are actual training service clients or mock clients.
    """

    def __init__(self, url: str, identifier: str | None = None):
        """
        Initialize the train service client

        Args:
            url: The service URL
            identifier: Optional identifier for the worker group
        """
        self._url = url
        self._identifier = identifier

    @abstractmethod
    def initialize_worker(
        self,
        config_path: str | None = None,
        config_dict: dict[str, Any] | None = None,
        role: str = "actor",
        world_size: int | None = None,
        zmq_base_port: int | None = None,
        dispatch_mode: str | None = None,
    ) -> dict[str, Any]:
        """
        Initialize the worker

        Args:
            config_path: Path to YAML configuration file
            config_dict: Optional dictionary configuration (overrides config_path)
            role: Worker role (e.g., 'actor', 'critic', 'reward')
            world_size: Number of workers (required for new worker groups)
            zmq_base_port: Base port for ZMQ (optional, auto-assigned if not provided)
            dispatch_mode: Dispatch mode (optional, defaults to 'broadcast')

        Returns:
            Response dictionary with initialization status
        """
        pass

    @abstractmethod
    def init_model(self) -> dict[str, Any]:
        """
        Initialize the model (background task)

        Returns:
            Response dictionary with model initialization status
        """
        pass

    @abstractmethod
    def update_actor(self, batch: dict) -> dict[str, Any]:
        """
        Update actor policy

        Args:
            batch: Data batch to send to workers

        Returns:
            Response dictionary with meta_info and metrics
        """
        pass

    @abstractmethod
    def compute_log_prob(self, batch: dict) -> dict[str, Any]:
        """
        Compute log probabilities

        Args:
            batch: Data batch to compute log probs for

        Returns:
            Dictionary containing log probabilities
        """
        pass

    @abstractmethod
    @contextmanager
    def actor_context(self):
        """
        Context manager for actor operations

        Yields control for actor-related operations within the context.
        """
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        saved_fully_shared_ckpt: bool = True,
        save_weight_only: bool = False,
        remove_previous_ckpt: bool = True,
    ) -> dict[str, Any]:
        """
        Save checkpoint

        Args:
            local_path: Local path to save checkpoint
            hdfs_path: HDFS path to save checkpoint (optional)
            global_step: Global step number
            saved_fully_shared_ckpt: Whether to save fully shared checkpoint
            save_weight_only: Whether to save weights only
            remove_previous_ckpt: Whether to remove previous checkpoint

        Returns:
            Response dictionary with save status
        """
        pass

    @abstractmethod
    def load_checkpoint(
        self, path: str, del_local_after_load: bool = True, load_weight_only: bool = False
    ) -> dict[str, Any]:
        """
        Load checkpoint

        Args:
            path: Path to checkpoint
            del_local_after_load: Whether to delete local checkpoint after loading
            load_weight_only: Whether to load weights only

        Returns:
            Response dictionary with load status
        """
        pass
