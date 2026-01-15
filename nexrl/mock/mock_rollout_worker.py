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
from ..nexrl_types import Trajectory
from ..rollout_worker.base_rollout_worker import BaseRolloutWorker
from .mock_inference_service_client import MockInferenceServiceClient

logger = logging.getLogger(__name__)


class MockRolloutWorker(BaseRolloutWorker):
    """
    Mock Rollout Worker for testing purposes.
    Uses MockInferenceServiceClient instead of real inference service to avoid network dependencies.
    """

    def __init__(self, config: DictConfig):  # pylint: disable=super-init-not-called
        # Initialize NexRLModule (grandparent)
        NexRLModule.__init__(self)  # pylint: disable=non-parent-init-called

        # Initialize attributes from BaseRolloutWorker without calling its __init__
        # to avoid creating a real inference service client
        self._config = config
        self._stop_event = threading.Event()
        self._thread: threading.Thread = None  # type: ignore

        # Use MockInferenceServiceClient instead of real client
        self._inference_client: MockInferenceServiceClient = MockInferenceServiceClient(config)  # type: ignore

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

        logger.info("MockRolloutWorker initialized with MockInferenceServiceClient")

    def rollout(self, task: dict[str, Any]) -> None:
        """
        Mock implementation of a single step operation.

        Args:
            task: Task to process
        """
        # Simulate processing time
        time.sleep(self._mock_delay)

        # Generate mock tokens and loss_mask (simple mock: 10 prompt tokens, 5 response tokens)
        prompt_tokens = list(range(100, 110))
        response_tokens = list(range(200, 205))
        tokens = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)

        # Create Trajectory dataclass
        mock_trajectory = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=random.uniform(0.0, 1.0),
            is_val=task.get("is_val", False),
            extra_fields={
                "prompt": task["prompt"],
                "response": f"Mock response {self._processed_count}",
                "logprobs": [0.0] * len(tokens),
            },
        )

        self._processed_count += 1
        logger.info(f"Mock rollout worker: Successfully processed step #{self._processed_count}")

        self._put_trajectory(mock_trajectory)
