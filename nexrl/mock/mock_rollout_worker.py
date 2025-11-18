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
Mock Rollout Worker for testing purposes
"""

import logging
import random
import threading
import time
from typing import Any

from omegaconf import DictConfig

from ..base_module import NexRLModule
from ..rollout_worker.base_rollout_worker import BaseRolloutWorker
from .mock_llm_service_client import MockLLMServiceClient

logger = logging.getLogger(__name__)


class MockRolloutWorker(BaseRolloutWorker):
    """
    Mock Rollout Worker for testing purposes.
    Uses MockLLMServiceClient instead of real LLM service to avoid network dependencies.
    """

    def __init__(self, config: DictConfig):
        # Initialize NexRLModule (grandparent)
        NexRLModule.__init__(self)

        # Initialize attributes from BaseRolloutWorker without calling its __init__
        # to avoid creating a real LLMServiceClient
        self._config = config
        self._stop_event = threading.Event()
        self._thread: threading.Thread = None  # type: ignore

        # Use MockLLMServiceClient instead of real client
        self._llm_client = MockLLMServiceClient(config)

        # Initialize other attributes that would be set by BaseRolloutWorker
        self._trajectory_pool = None  # type: ignore
        self._dataloader = None  # type: ignore
        self._weight_sync_controller = None  # type: ignore
        self._validate_dataloader = None  # type: ignore
        self._validator = None  # type: ignore
        self._next_task = None
        self._is_running_validate = False

        # Mock-specific attributes
        self._processed_count: int = 0
        self._mock_delay: float = 0.1

        logger.info("MockRolloutWorker initialized with MockLLMServiceClient")

    def step(self, task: dict[str, Any]) -> None:
        """
        Mock implementation of a single step operation.

        Args:
            task: Task to process
        """
        # Simulate processing time
        time.sleep(self._mock_delay)

        # Generate mock rollout result
        mock_result = {
            "prompt": task["prompt"],
            "response": f"Mock response {self._processed_count}",
            "reward": random.uniform(0.0, 1.0),
        }

        self._processed_count += 1
        logger.info(f"Mock rollout worker: Successfully processed step #{self._processed_count}")

        self._put_trajectory(mock_result)
