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
Algorithm Processor for NexRL framework
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import DictConfig

from ..base_module import NexRLModule
from ..executor import execute
from ..nexrl_types import Batch

if TYPE_CHECKING:
    from ..train_batch_pool import TrainBatchPool
    from ..trajectory_pool import TrajectoryPool

logger = logging.getLogger(__name__)


class BaseAlgorithmProcessor(NexRLModule, ABC):
    """
    Base class for algorithm processors
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the algorithm processor

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self._config: DictConfig = config
        self._service_table: dict[str, Any] = (
            {}
        )  # the dict to get the actual model service using the model id
        self._stop_event: threading.Event = threading.Event()
        self._stop_event.clear()

        # Timing tracking for batch processing
        self._batch_count: int = 0

    def set_module_references(
        self, trajectory_pool: "TrajectoryPool", train_batch_pool: "TrainBatchPool"
    ) -> None:
        """
        Set the module references for the algorithm processor.
        """
        self._trajectory_pool = trajectory_pool
        self._train_batch_pool = train_batch_pool

    def run(self):
        """
        Main startup function. Starts a thread and runs the main loop.
        """
        self._thread = threading.Thread(target=self._main_loop)
        self._thread.start()

    def stop(self):
        """
        Stop the algorithm processor.
        """
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()

    def _main_loop(self):
        """
        The main loop of the AlgorithmProcessor. It contains a while loop
        """
        while not self._stop_event.is_set():
            # Get batch from rollout result pool
            batch = self._get_batch()
            if batch is None:
                time.sleep(0.1)
                continue

            start_time = time.time()

            with self._activity_tracker.track(self._module_name, "algorithm_processing"):
                self._fit(batch, "update")

            processing_time = time.time() - start_time

            self._activity_tracker.experiment_logger_post(
                backend="wandb",
                data={
                    "timing/batch_processing": processing_time,
                },
                step=self._batch_count,
            )
            self._batch_count += 1

    def _get_batch(self) -> Batch:
        """
        Get the result trajectory batch from the TrajectoryPool.

        Returns:
            Batch: Batch of rollout results
        """
        return execute(self._trajectory_pool.get_batch)

    def _put_batch(self, batch: Batch, update_fn: str) -> bool:
        """
        Put the prepared batch into the training batch pool of target model.

        Args:
            model: Model tag
            update_fn: Update function name
            batch: Batch to put

        Returns:
            bool: Whether the batch is inserted correctly
        """
        return execute(self._train_batch_pool.put_batch, batch, update_fn)

    @abstractmethod
    def _fit(self, batch: Batch, update_fn: str):
        """
        To correctly estimate advantage, we need to compute the logprobs or logits from multiple models,
        including but not limited to the reward model, the reference model, and critic models.
        This function calls the services of the models and gets the output value.

        Args:
            batch: Batch to process
            update_fn: Update function name

        Returns:
            Batch: Processed batch
        """
        pass

    # ------------ Tools for logging metrics ------------
    @staticmethod
    def reduce_metrics(metrics: dict):
        for key, val in metrics.items():
            metrics[key] = np.mean(val)
        return metrics

    @staticmethod
    def _compute_response_info(batch: Batch) -> dict:
        response_length = batch.values["responses"].shape[-1]

        prompt_mask = batch.values["attention_mask"][:, :-response_length]
        response_mask = batch.values["attention_mask"][:, -response_length:]

        prompt_length = prompt_mask.sum(-1).float()
        response_length = response_mask.sum(-1).float()  # (batch_size,)

        return dict(
            response_mask=response_mask,
            prompt_length=prompt_length,
            response_length=response_length,
        )

    @staticmethod
    def compute_data_metrics(batch: Batch, use_critic: bool = True) -> dict:
        sequence_score = batch.values["token_level_scores"].sum(-1)
        sequence_reward = batch.values["token_level_rewards"].sum(-1)

        advantages = batch.values["advantages"]
        returns = batch.values["returns"]

        max_response_length = batch.values["responses"].shape[-1]

        prompt_mask = batch.values["attention_mask"][:, :-max_response_length].bool()
        response_mask = batch.values["attention_mask"][:, -max_response_length:].bool()

        max_prompt_length = prompt_mask.size(-1)

        response_info = BaseAlgorithmProcessor._compute_response_info(batch)
        prompt_length = response_info["prompt_length"]
        response_length = response_info["response_length"]

        valid_adv = torch.masked_select(advantages, response_mask)
        valid_returns = torch.masked_select(returns, response_mask)

        if use_critic:
            values = batch.values["values"]
            valid_values = torch.masked_select(values, response_mask)
            return_diff_var = torch.var(valid_returns - valid_values)
            return_var = torch.var(valid_returns)

        metrics = {
            # score
            "critic/score/mean": torch.mean(sequence_score).detach().item(),
            "critic/score/max": torch.max(sequence_score).detach().item(),
            "critic/score/min": torch.min(sequence_score).detach().item(),
            # accuracy
            "critic/accuracy/mean": torch.mean(torch.round(sequence_score)).detach().item(),
            "critic/accuracy/max": torch.max(torch.round(sequence_score)).detach().item(),
            "critic/accuracy/min": torch.min(torch.round(sequence_score)).detach().item(),
            # reward
            "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
            "critic/rewards/max": torch.max(sequence_reward).detach().item(),
            "critic/rewards/min": torch.min(sequence_reward).detach().item(),
            # adv
            "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
            "critic/advantages/max": torch.max(valid_adv).detach().item(),
            "critic/advantages/min": torch.min(valid_adv).detach().item(),
            # returns
            "critic/returns/mean": torch.mean(valid_returns).detach().item(),
            "critic/returns/max": torch.max(valid_returns).detach().item(),
            "critic/returns/min": torch.min(valid_returns).detach().item(),
            **(
                {
                    # values
                    "critic/values/mean": torch.mean(valid_values).detach().item(),
                    "critic/values/max": torch.max(valid_values).detach().item(),
                    "critic/values/min": torch.min(valid_values).detach().item(),
                    # vf explained var
                    "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5))
                    .detach()
                    .item(),
                }
                if use_critic
                else {}
            ),
            # response length
            "response_length/mean": torch.mean(response_length).detach().item(),
            "response_length/max": torch.max(response_length).detach().item(),
            "response_length/min": torch.min(response_length).detach().item(),
            "response_length/clip_ratio": torch.mean(
                torch.eq(response_length, max_response_length).float()
            )
            .detach()
            .item(),
            # prompt length
            "prompt_length/mean": torch.mean(prompt_length).detach().item(),
            "prompt_length/max": torch.max(prompt_length).detach().item(),
            "prompt_length/min": torch.min(prompt_length).detach().item(),
            "prompt_length/clip_ratio": torch.mean(
                torch.eq(prompt_length, max_prompt_length).float()
            )
            .detach()
            .item(),
        }

        if "average_agent_acc" in batch.metadata:
            for k, v in batch.metadata["average_agent_acc"].items():
                metrics["critic/average_agent_acc/" + k] = v
        return metrics
