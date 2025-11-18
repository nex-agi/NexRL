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
Base class for agent-based rollout workers that process agent outputs into trajectories.
"""

import logging
from abc import abstractmethod
from typing import Any

import torch
from omegaconf import DictConfig

from ..nexrl_types import Trajectory
from ..utils.torch_functional import compute_position_id_with_mask, padding_data
from .base_rollout_worker import BaseRolloutWorker

logger = logging.getLogger(__name__)


class AgentRolloutWorker(BaseRolloutWorker):
    """
    Base class for agent-based rollout workers.

    This class provides common functionality for processing agent outputs into trajectories,
    including token processing, padding, and trajectory construction.

    Derived classes should override the _agent_run method to implement specific agent logic.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the Agent Rollout Worker.

        Args:
            config: Configuration containing data settings for max lengths
        """
        super().__init__(config)
        # Configuration parameters
        self._max_prompt_length = config.get("max_prompt_length", 4096)
        self._max_response_length = config.get("max_response_length", 2048)

    def _process_agent_results(
        self, prompt_messages: list[dict[str, Any]], response_tokens: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Process prompt messages and response tokens to generate input_ids, attention_mask,
        position_ids, loss_mask, etc.

        Args:
            prompt_messages: List of message dictionaries in chat format
            response_tokens: List of response token IDs

        Returns:
            dict containing:
                - input_ids: concatenated prompt and response token IDs
                - attention_mask: attention mask for the sequence
                - position_ids: position IDs for the sequence
                - loss_mask: mask indicating which tokens to compute loss on
                - prompts: padded prompt token IDs only
                - responses: padded response token IDs only
        """
        tokenizer = self._llm_client.tokenizer
        pad_token_id = tokenizer.pad_token_id

        # Apply chat template to prompt messages
        prompt_with_chat_template = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
            add_special_tokens=True,
        )

        # Tokenize prompt
        prompt_tokens = tokenizer(
            prompt_with_chat_template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].tolist()

        # Example:
        # max_prompt_length: 10, max_response_length: 10
        # prompt_tokens: [101, 102, 103, 104, 105]
        # prompt_input_ids: [0, 0, 0, 0, 0, 101, 102, 103, 104, 105]
        # prompt_attention_mask: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        # prompt_loss_mask: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # response_tokens: [106, 107, 108, 109, 110]
        # response_input_ids: [106, 107, 108, 109, 110, 0, 0, 0, 0, 0]
        # response_attention_mask: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # response_loss_mask: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        # input_ids: [0, 0, 0, 0, 0, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 0, 0, 0, 0, 0]
        # attention_mask: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # loss_mask: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # prompts: [0, 0, 0, 0, 0, 101, 102, 103, 104, 105]
        # responses: [106, 107, 108, 109, 110, 0, 0, 0, 0, 0]

        # position_ids: [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9]

        # Pad prompt tokens and masks (left padding)
        prompt_input_ids = padding_data(
            prompt_tokens,
            max_length=self._max_prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation="error",
        )
        prompt_attention_mask = padding_data(
            torch.ones((1, len(prompt_tokens)), dtype=torch.int),
            max_length=self._max_prompt_length,
            pad_token_id=0,
            left_pad=True,
            truncation="error",
        )
        prompt_loss_mask = torch.zeros_like(prompt_input_ids, dtype=torch.int)

        # Pad response tokens and masks (right padding)
        response_input_ids = padding_data(
            response_tokens,
            max_length=self._max_response_length,
            pad_token_id=pad_token_id,
            left_pad=False,
            truncation="error",
        )
        response_attention_mask = padding_data(
            torch.ones((1, len(response_tokens)), dtype=torch.int),
            max_length=self._max_response_length,
            pad_token_id=0,
            left_pad=False,
            truncation="error",
        )
        response_loss_mask = padding_data(
            torch.ones((1, len(response_tokens)), dtype=torch.int),
            max_length=self._max_response_length,
            pad_token_id=0,
            left_pad=False,
            truncation="error",
        )

        # Concatenate prompt and response
        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        loss_mask = torch.cat([prompt_loss_mask, response_loss_mask], dim=1)

        # Compute position_ids from attention_mask
        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "attention_mask": attention_mask.squeeze(0),
            "position_ids": position_ids.squeeze(0),
            "loss_mask": loss_mask.squeeze(0),
            "prompts": prompt_input_ids.squeeze(0),
            "responses": response_input_ids.squeeze(0),
        }

    def step(self, task: dict[str, Any]) -> str | None:
        """
        Single step operation for agent-based rollout.

        The _agent_run function is exactly what an agent will originally do.
        While the step function handles the remaining work that organize the agent execution results into a trajectory.

        Args:
            task: A dictionary containing:
                - 'prompt': list of message dictionaries in chat format, e.g.,
                    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
                - 'ground_truth': the correct answer
                - 'is_val': whether this is validation data (optional, defaults to False)
                - Other metadata fields (group_id, run_id, train_step, etc.)

        Returns:
            str: 'success', 'fail', or 're-rollout' (from _put_trajectory)
        """
        agent_result = self._agent_run(task)

        if agent_result is None or agent_result.get("response") is None:
            logger.warning("Agent finish without response, skipping")
            return None

        # Process tokens to get input_ids, attention_mask, etc.
        processed_agent_result = self._process_agent_results(
            prompt_messages=task["prompt"].tolist(),
            response_tokens=agent_result.get("response_tokens", []),
        )

        # Extract ground_truth - could be in task directly or in reward_model dict
        ground_truth = task["reward_model"].get("ground_truth", "")

        trajectory: Trajectory = {
            # input fields
            "ground_truth": ground_truth,
            "group_id": task.get("group_id", ""),
            "run_id": task.get("run_id", 0),
            "task_id": task.get("task_id", 0),
            "is_val": task.get("is_val", False),
            "temperature": self._config.temperature,
            # output fields
            "input_ids": processed_agent_result[
                "input_ids"
            ],  # shape: (1, max_prompt_length + max_response_length)
            "attention_mask": processed_agent_result[
                "attention_mask"
            ],  # shape: (1, max_prompt_length + max_response_length)
            "position_ids": processed_agent_result[
                "position_ids"
            ],  # shape: (1, max_prompt_length + max_response_length)
            "loss_mask": processed_agent_result[
                "loss_mask"
            ],  # shape: (1, max_prompt_length + max_response_length)
            "prompts": processed_agent_result["prompts"],  # shape: (1, max_prompt_length)
            "responses": processed_agent_result["responses"],  # shape: (1, max_response_length)
            "finish_reason": agent_result["finish_reason"],  #
            # reward fields
            "reward": agent_result["reward"],
            "score": agent_result["score"],
            **{k: v for k, v in task.items() if k not in ["prompt", "ground_truth", "is_val"]},
        }

        # Add any additional fields from agent_result that aren't already in trajectory
        for k, v in agent_result.items():
            if k not in trajectory:
                trajectory[k] = v

        # Put trajectory into pool and return result
        return self._put_trajectory(trajectory)

    @abstractmethod
    def _agent_run(self, task: dict[str, Any]) -> dict[str, Any] | None:
        """
        Execute the agent to process a single task.

        This method should be overridden by derived classes to implement specific agent logic.

        Args:
            task: A dictionary containing task information including:
                - 'prompt': list of message dictionaries
                - 'ground_truth': the correct answer
                - 'is_val': whether this is validation data
                - Other metadata fields

        Returns:
            dict containing agent results with keys:
                - 'response': the generated response string
                - 'response_tokens': list of response token IDs
                - 'finish_reason': why generation stopped
                - 'reward': reward score for the response
                - 'format_correct': whether the format is correct
                - 'correct': whether the answer is correct
            or None if the agent run failed
        """
        pass
