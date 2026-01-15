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
Agent Rollout Worker for NexRL framework
Unified rollout worker supporting single-turn and multi-turn agents.
Trajectory processing is handled by trainers.
"""

import logging
from typing import Any

from omegaconf import DictConfig

from ..agent import BaseAgent
from ..nexrl_types import Trajectory
from .base_rollout_worker import BaseRolloutWorker

logger = logging.getLogger(__name__)


def create_agent(agent_cls_name: str, config: DictConfig) -> BaseAgent:
    """
    Factory function to create an agent instance.

    Args:
        agent_cls_name: Name of the agent class (e.g., "single_turn_math", "nexau")
        config: Configuration for the agent

    Returns:
        Agent instance
    """
    from ..agent import SingleTurnMathAgent

    agent_registry = {
        "single_turn_math": SingleTurnMathAgent,
    }

    if agent_cls_name not in agent_registry:
        raise ValueError(
            f"Unknown agent class: {agent_cls_name}. " f"Available: {list(agent_registry.keys())}"
        )

    return agent_registry[agent_cls_name](config)


class AgentRolloutWorker(BaseRolloutWorker):
    """
    Unified agent-based rollout worker.

    Uses an agent_cls instance for task-specific logic and creates
    simple trajectories with only nexrl_train fields. Trajectory processing
    (padding, tensor creation, etc.) is handled by trainers.

    Supports both single-turn and multi-turn agents:
    - Single-turn agents: agent.run() returns a single dict -> 1 trajectory
    - Multi-turn agents: agent.run() returns a list of dicts -> multiple trajectories
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the Agent Rollout Worker.

        Args:
            config: Configuration containing:
                - agent_cls: Name of the agent class to use
        """
        super().__init__(config)

        # Create agent instance
        agent_cls_name = config.get("agent_cls", "single_turn_math")
        self._agent = create_agent(agent_cls_name, config)

    def rollout(self, task: dict[str, Any]) -> str | None:
        """
        Execute agent rollout and create trajectories.

        Supports both single-turn and multi-turn agents:
        - Single-turn: agent.run() returns a dict (wrapped as 1-element list)
        - Multi-turn: agent.run() returns a list of dicts (one per turn)

        Each agent result is converted to a simple trajectory containing:
        - nexrl_train fields (prompt_tokens, response_tokens, response_logprobs)
        - reward, score, finish_reason
        - metadata from task

        Trajectory processing (padding, tensors) is handled by trainers.

        Args:
            task: A dictionary containing task information

        Returns:
            str: 'success', 'fail', or 're-rollout' (from _put_trajectory)
        """
        # Run the agent - returns dict or list of dicts
        agent_results_raw = self._agent.run(task, self._inference_client)

        if agent_results_raw is None:
            logger.warning("Agent finished without response, skipping")
            return None

        # Wrap single-turn result in a list for uniform processing
        if isinstance(agent_results_raw, dict):
            agent_results = [agent_results_raw]
        else:
            agent_results = agent_results_raw

        if not isinstance(agent_results, list) or len(agent_results) == 0:
            logger.warning("Agent returned empty or invalid results, skipping")
            return None

        # Extract ground_truth - could be in task directly or in reward_model dict
        ground_truth = task["reward_model"].get("ground_truth", "")

        # Process each agent result into a trajectory
        last_result = None
        for agent_result in agent_results:
            # Validate agent result
            if "nexrl_train" not in agent_result:
                logger.warning("Agent result missing 'nexrl_train' field, skipping this turn")
                continue

            train_fields = agent_result["nexrl_train"]

            # Get prompt and response tokens
            prompt_tokens = train_fields.get("prompt_tokens", [])
            response_tokens = train_fields.get("response_tokens", [])
            response_logprobs = train_fields.get("response_logprobs", [])

            # Create tokens and loss_mask
            tokens = prompt_tokens + response_tokens
            loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)

            # Create Trajectory dataclass
            trajectory = Trajectory(
                tokens=tokens,
                loss_mask=loss_mask,
                reward=agent_result.get("reward", 0.0),
                is_val=task.get("is_val", False),
                extra_fields={
                    # Metadata fields
                    "ground_truth": ground_truth,
                    "group_id": task.get("group_id", ""),
                    "run_id": task.get("run_id", 0),
                    "task_id": task.get("task_id", 0),
                    "temperature": self._config.temperature,
                    "finish_reason": agent_result.get("finish_reason", "stop"),
                    # Logprobs field (0.0 for prompt, actual logprobs for response)
                    "logprobs": [0.0] * len(prompt_tokens) + response_logprobs,
                },
            )

            # Add extra task fields (exclude prompt and ground_truth)
            excluded_keys = {"prompt", "ground_truth", "is_val", "reward_model"}
            for k, v in task.items():
                if k not in excluded_keys:
                    trajectory[k] = v

            # Add any additional agent_result fields (exclude LLM-returned fields)
            llm_return_keys = {
                "id",
                "object",
                "created",
                "model",
                "choices",
                "usage",
                "nexrl_train",
                "tool_calls",
                "response",
                "finish_reason",
                "prompt_tokens",
                "response_tokens",
                "response_logprobs",
                "loss_mask",
                "reward",
            }
            for k, v in agent_result.items():
                if k not in llm_return_keys and k not in trajectory:
                    trajectory[k] = v

            # Put trajectory into pool
            last_result = self._put_trajectory(trajectory)

        return last_result
