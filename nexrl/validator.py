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
Validator for NexRL framework
"""

import logging
import threading

from omegaconf import DictConfig

from .base_module import NexRLModule
from .data_loader.data_loader import BaseDataLoader
from .executor import execute
from .nexrl_types import Trajectory

logger = logging.getLogger(__name__)


class Validator(NexRLModule):
    """
    Validator collects validation trajectories, computes metrics, and logs results.
    Unlike TrajectoryPool, this does not need batching logic - just simple collection.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the validator

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self._config = config
        self._trajectories: list[Trajectory] = []
        self._lock = threading.Lock()
        self._validate_dataloader: BaseDataLoader = None  # type: ignore  # Set via set_module_references()

    def set_module_references(self, validate_dataloader: BaseDataLoader):
        """
        Set the module references for the validator.
        """
        self._validate_dataloader = validate_dataloader

    def put_trajectory(self, trajectory: Trajectory) -> str:
        """
        Store validation trajectory

        Args:
            trajectory: Trajectory to store

        Returns:
            'success' to match the signature of TrajectoryPool.put_trajectory
        """
        with self._lock:
            self._trajectories.append(trajectory)
        return "success"

    def is_complete(self) -> bool:
        """
        Check if all validation trajectories collected: dataloader is drained and rollout workers are quiescent

        Returns:
            True if all expected trajectories are collected
        """
        return (
            execute(self._validate_dataloader.is_finished)
            and self._activity_tracker.is_rollout_worker_quiescent()
        )

    def compute_and_log_metrics(self) -> dict[str, float]:
        """
        Compute validation metrics and log them via activity tracker.
        For each key in the score dictionary, compute the mean across all trajectories.

        Args:
            train_step: Current training step for logging

        Returns:
            Dictionary of computed metrics with "val/" prefix
        """
        from collections import defaultdict

        import numpy as np

        with self._lock:
            trajectories = self._trajectories.copy()

        if not trajectories:
            logger.error("No trajectories to compute metrics")
            raise ValueError("No trajectories to compute metrics")

        # Collect all score values by key
        score_values = defaultdict(list)

        for traj in trajectories:
            score_dict = traj.get("score", {})
            if isinstance(score_dict, dict):
                for key, value in score_dict.items():
                    # Try to convert value to float
                    try:
                        score_values[key].append(float(value))
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Cannot convert score value to float: key={key}, value={value}, error={e}"
                        )
            else:
                logger.warning(f"Trajectory score is not a dict: {score_dict}")

        # Compute mean for each score key and add "val/" prefix
        metrics = {}
        for key, values in score_values.items():
            if values:
                metrics[f"val/{key}"] = float(np.mean(values))

        # Add number of samples
        metrics["val/num_samples"] = len(trajectories)

        # Log metrics via activity tracker
        execute(self._activity_tracker.experiment_logger_post, backend="wandb", data=metrics)

        logger.info(f"Validation metrics: {metrics}")

        self._trajectories.clear()

        return metrics
