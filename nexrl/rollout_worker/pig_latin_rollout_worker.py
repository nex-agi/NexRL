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
Pig Latin Rollout Worker for supervised learning with cross entropy loss
"""

import logging
from typing import Any

from omegaconf import DictConfig

from ..nexrl_types import Trajectory
from .base_rollout_worker import BaseRolloutWorker

logger = logging.getLogger(__name__)


class PigLatinRolloutWorker(BaseRolloutWorker):
    """
    Rollout worker for pig latin translation task.

    This worker generates trajectories for supervised learning using cross entropy loss.
    Each task contains an "input" (English phrase) and "output" (expected pig latin translation).
    The worker generates trajectories with proper tokenization for supervised fine-tuning.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the pig latin rollout worker.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._tokenizer = None  # type: ignore
        logger.info("PigLatinRolloutWorker initialized")

    def init_inference_service_client(self, service_holder=None):
        """
        Initialize inference service client and tokenizer.

        Args:
            service_holder: Shared backend-specific service holder (Tinker/Weaver)
        """
        super().init_inference_service_client(service_holder)

        # Initialize tokenizer from config
        # For pig latin rollout worker, we always load tokenizer directly
        # since we don't use the inference client for generation
        from transformers import AutoTokenizer

        tokenizer_path = self._config.get("tokenizer", self._config.data.tokenizer_path)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Loaded tokenizer from {tokenizer_path}")

    def rollout(self, task: dict[str, Any]) -> str | None:
        """
        Generate a trajectory for pig latin translation.

        Args:
            task: Task dictionary containing:
                - "input": English phrase to translate
                - "output": Expected pig latin translation
                - Other metadata fields

        Returns:
            str: 'success', 'fail', or 're-rollout' (from _put_trajectory)
        """
        # Validate task
        if "input" not in task or "output" not in task:
            logger.error(f"Task missing 'input' or 'output' field: {task}")
            return None

        # Construct prompt and expected completion
        english_input = task["input"]
        pig_latin_output = task["output"]

        # Create prompt in the format: "English: {input}\nPig Latin:"
        prompt = f"English: {english_input}\nPig Latin:"
        completion = f" {pig_latin_output}\n\n"

        # Tokenize prompt and completion
        assert self._tokenizer is not None, "Tokenizer not initialized"
        prompt_tokens = self._tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = self._tokenizer.encode(completion, add_special_tokens=False)

        # Combine tokens
        tokens = prompt_tokens + completion_tokens

        # Create loss mask (0 for prompt, 1 for completion)
        loss_mask = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

        # For supervised learning, we can use exact match as reward
        # In cross entropy training, this reward is mainly for logging purposes
        reward = 1.0  # All supervised examples get reward of 1.0

        # Create trajectory dataclass
        trajectory = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=reward,
            is_val=task.get("is_val", False),
            extra_fields={
                "input": english_input,
                "output": pig_latin_output,
                "prompt": prompt,
                "completion": completion,
                "temperature": self._config.get("temperature", 0.0),
                "finish_reason": "supervised",  # Mark as supervised learning
                "model_tag": task.get("model_tag", "default"),
            },
        )

        # Add extra task fields to trajectory
        excluded_keys = {"input", "output", "is_val", "model_tag"}
        for k, v in task.items():
            if k not in excluded_keys:
                trajectory[k] = v

        # Put the trajectory into the trajectory pool
        return self._put_trajectory(trajectory)
