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
Base Agent - Abstract class for task-specific agent logic
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for agents.

    Agents implement pure task logic - how to process a task, call LLM,
    and compute rewards. They are backend-agnostic.

    Rollout workers use agents via the run() method and handle
    trajectory creation based on their specific backend requirements.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the agent.

        Args:
            config: Configuration containing agent-specific settings
        """
        self._config = config

    @abstractmethod
    def run(self, task: dict[str, Any], llm_client: Any) -> dict[str, Any] | None:
        """
        Execute the agent to process a single task.

        Args:
            task: Task dictionary containing:
                - 'prompt': Input prompt (format depends on task type)
                - 'ground_truth': Expected answer for reward calculation
                - 'is_val': Whether this is validation data
                - Other task-specific fields

            llm_client: LLM service client for inference
                - Has completion() and generate() methods
                - Has tokenizer attribute

        Returns:
            dict containing at minimum:
                - 'response': Generated response text
                - 'finish_reason': Why generation stopped
                - 'reward': Scalar reward for the response
                - 'score': Detailed score breakdown (dict)
            Plus all fields from the LLM client response (tokens, logprobs, etc.)
            Returns None if the agent run failed.
        """
