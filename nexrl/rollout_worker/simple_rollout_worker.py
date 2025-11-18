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
Derived Rollout Worker for NexRL framework
"""

import logging
import time
from typing import Any

import torch
from omegaconf import DictConfig

from .base_rollout_worker import BaseRolloutWorker

logger = logging.getLogger(__name__)


class SimpleRolloutWorker(BaseRolloutWorker):
    """
    User-defined RolloutWorker logic.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def step(self, task: dict[str, Any]) -> str | None:
        """
        Single step operation, defined by user. Derived worker classes should override
        this function to implement user-defined worker operations.
        """
        # Extract prompt from task
        if "prompt" not in task:
            logger.error(f"Task missing 'prompt' field: {task}")
            return None

        prompt = task["prompt"]

        # Call LLM completion
        completion_result = self._llm_client.completion(prompt)

        # Create trajectory from the completion result
        trajectory = {
            "prompt": prompt,
            "response": completion_result["response"],
            "finish_reason": completion_result["finish_reason"],
            **{k: v for k, v in task.items() if k != "prompt"},  # Include other task fields
        }

        # Put the trajectory into the trajectory pool
        return self._put_trajectory(trajectory)
