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
Single-Turn Math Agent - Pure task logic for math problem solving
"""

import logging
from typing import Any

from omegaconf import DictConfig

from ..utils.reward_score.bp_math import compute_score
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SingleTurnMathAgent(BaseAgent):
    """
    Single-Turn Math Agent for math problem solving.

    This agent handles single-turn interactions where the model receives
    a math problem and generates a solution in one shot.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        # Get math-specific config from rollout_worker.agent.math

        self._judge_mode = config.agent.math.judge_mode
        self._max_prompt_length = config.max_prompt_length

    def run(self, task: dict[str, Any], llm_client: Any) -> dict[str, Any] | None:
        """
        Execute single-turn math problem solving.

        Args:
            task: Task dictionary containing:
                - 'prompt': List of message dicts in chat format
                - 'ground_truth' or 'reward_model.ground_truth': Expected answer
                - 'is_val': Whether this is validation data

            llm_client: LLM service client with generate() method and tokenizer

        Returns:
            dict with response, reward, score, and all LLM client fields
        """
        if task.get("prompt") is None:
            logger.warning(f"Task missing 'prompt' field: {task}")
            return None

        # Get prompt - handle both tensor and list formats
        prompt = task["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()

        # Apply chat template to get prompt string
        prompt_str = llm_client.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )

        # Check prompt length
        prompt_len = len(prompt_str)
        if prompt_len > self._max_prompt_length:
            logger.warning(f"Prompt too long ({prompt_len} > {self._max_prompt_length}), skipping")
            return None

        # Call LLM - get OpenAI-style payload
        llm_result: dict[str, Any] = llm_client.generate(messages=prompt)

        if "choices" not in llm_result or not llm_result["choices"]:
            raise ValueError("LLM result missing 'choices' field or is empty")

        first_choice = llm_result["choices"][0]

        message = first_choice.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError("LLM result missing response content")
        response = message["content"]

        finish_reason = first_choice.get("finish_reason")
        if finish_reason is None:
            raise ValueError("LLM result missing finish_reason")

        if response is None:
            logger.warning("LLM returned no response")
            return None

        # Calculate reward
        reward = self._calc_reward(response, finish_reason, task, prompt_str)

        logger.debug(f"Reward: {reward}")

        # Build result dict - start with all LLM fields, then add computed fields
        result = dict(llm_result)  # Copy all LLM client fields (keep original immutable)
        result.update(
            {
                "reward": reward,
                "score": {
                    "reward_score": reward,
                    "format_correct": reward > 0.0,
                    "correct": reward >= 1.0,
                },
                "prompt_str": prompt_str,
            }
        )

        return result

    def _calc_reward(
        self, response: str, finish_reason: str | None, task: dict[str, Any], prompt_str: str
    ) -> float:
        """
        Calculate reward for the response.

        Args:
            llm_result: The LLM response
            task: The task
            prompt_str: The prompt string

        Returns:
            float: The calculated reward
        """
        # Extract ground truth
        ground_truth = task.get("ground_truth", "")
        if not ground_truth and "reward_model" in task:
            ground_truth = task["reward_model"].get("ground_truth", "")

        # Handle mocked data
        if ground_truth == "whatever":
            import random

            # Return a random reward for mocked data
            # 50% chance of getting reward 1, otherwise uniform between 0 and 1.0
            if random.random() < 0.5:
                return 1.0
            else:
                return random.uniform(0.0, 1.0)

        # Handle truncated response
        if finish_reason == "length":
            logger.debug("Reward = 0.0, due to length truncation")
            return 0.0

        # Compute actual reward
        stage = "val" if task.get("is_val", False) else "train"
        prompt_data = task.get("prompt")
        raw_prompt = (
            prompt_data.tolist()
            if prompt_data is not None and hasattr(prompt_data, "tolist")
            else None
        )

        reward = compute_score(
            prompt_str=prompt_str,
            solution_str=response,
            ground_truth=ground_truth,
            stage=stage,
            raw_prompt=raw_prompt,
            judge_mode=self._judge_mode,
        )

        return reward
