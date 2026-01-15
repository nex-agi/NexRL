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
import re
from typing import Any

from ..nexrl_types import Trajectory
from .base_rollout_worker import BaseRolloutWorker

logger = logging.getLogger(__name__)


class SimpleRolloutWorker(BaseRolloutWorker):
    """
    User-defined RolloutWorker logic.
    """

    def rollout(self, task: dict[str, Any]) -> str | None:
        """
        Single rollout operation, defined by user. Derived worker classes should override
        this function to implement user-defined worker operations.
        """
        # Extract prompt from task
        if "prompt" not in task:
            logger.error(f"Task missing 'prompt' field: {task}")
            return None

        prompt = task["prompt"]

        # Call inference service completion
        assert self._inference_client is not None, "Inference client not initialized"
        completion_result = self._inference_client.completion(prompt)

        # Extract tokens and logprobs from completion result
        prompt_tokens = completion_result.get("prompt_tokens", [])
        response_tokens = completion_result.get("response_tokens", [])
        response_logprobs = completion_result.get("response_logprobs", [])

        # Create tokens and loss_mask
        tokens = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)

        # Extract answer from response using <answer></answer> tags
        response = completion_result.get("response", "")
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        extracted_answer = answer_match.group(1).strip() if answer_match else ""

        # Get ground truth from task
        ground_truth = task.get("ground_truth", "")
        if "reward_model" in task:
            ground_truth = task["reward_model"].get("ground_truth", ground_truth)

        # Calculate reward: 1.0 if answer matches ground_truth, else 0.0
        reward = 1.0 if extracted_answer == ground_truth else 0.0

        # Create Trajectory dataclass
        trajectory = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=reward,
            is_val=task.get("is_val", False),
            extra_fields={
                # Metadata fields
                "ground_truth": ground_truth,
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
                "task_id": task.get("task_id", 0),
                "temperature": self._config.temperature,
                "finish_reason": completion_result.get("finish_reason", "stop"),
                # Logprobs field (0.0 for prompt, actual logprobs for response)
                "logprobs": [0.0] * len(prompt_tokens) + response_logprobs,
                # Additional fields for debugging
                "response": response,
                "extracted_answer": extracted_answer,
            },
        )

        # Add extra task fields (exclude prompt and ground_truth)
        excluded_keys = {"prompt", "ground_truth", "is_val", "reward_model"}
        for k, v in task.items():
            if k not in excluded_keys:
                trajectory[k] = v

        # Put the trajectory into the trajectory pool
        return self._put_trajectory(trajectory)
