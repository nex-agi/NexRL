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
Derived Single-Turn Math Agent for NexRL framework
"""

import logging
from typing import Any

from omegaconf import DictConfig

from ..utils.reward_score.bp_math import compute_score
from .agent_rollout_worker import AgentRolloutWorker

logger = logging.getLogger(__name__)


class SingleTurnMathAgent(AgentRolloutWorker):
    """
    Single-Turn Math Agent for math problem solving.
    This agent handles single-turn interactions where the model receives a math problem
    and generates a solution in one shot.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._judge_mode = config.get("judge_mode", "rule")

    def _calc_reward(
        self, agent_result: dict[str, Any], task: dict[str, Any], prompt_str: str
    ) -> float:
        """
        Calculate reward for the response.

        Args:
            agent_result: The agent result
            task: The task

        Returns:
            float: The calculated reward
        """
        if task.get("ground_truth", "") == "whatever":  # mocked data
            import random

            # return a random reward for mocked data
            # 50% chance of getting reward 1, otherwise uniform between 0 and 1.2
            if random.random() < 0.5:
                return 1.0
            else:
                return random.uniform(0.0, 1.0)
        elif agent_result["finish_reason"] == "length":
            # Response was truncated
            # logger.warning("Response was truncated due to length")
            return 0.0
        else:
            stage = "val" if task.get("is_val", False) else "train"

            ground_truth = task["reward_model"]["ground_truth"]

            reward = compute_score(
                prompt_str=prompt_str,
                solution_str=agent_result["response"],
                ground_truth=ground_truth,
                stage=stage,
                raw_prompt=task["prompt"].tolist() if task.get("prompt") is not None else None,
                judge_mode=self._judge_mode,
            )
            return reward

    def _agent_run(self, task: dict[str, Any]) -> dict[str, Any] | None:
        if task.get("prompt") is None:
            logger.warning(f"Task missing 'prompt' field: {task}")
            return None

        prompt = task["prompt"].tolist()

        prompt_str = self._llm_client.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )
        # Check prompt length
        prompt_len = len(prompt_str)
        if prompt_len > self._max_prompt_length:
            logger.warning(f"Prompt too long ({prompt_len} > {self._max_prompt_length}), skipping")
            return None

        agent_result: dict[str, Any] = self._llm_client.generate(messages=prompt)

        dump_result = {
            "prompt_str": prompt_str,
        }
        dump_result.update(agent_result)
        self.easy_dump(dump_result, keys=["agent_result"])

        if agent_result is None or agent_result.get("response") is None:
            return None

        reward = self._calc_reward(agent_result, task, prompt_str)

        logger.debug(f"Reward: {reward}")

        agent_result.update(
            {
                "reward": reward,
                "correct": reward >= 1.0,
                # for logging metrics
                "score": {
                    "reward_score": reward,
                    "format_correct": reward > 0.0,
                    "correct": reward >= 1.0,
                },
            }
        )

        return agent_result
