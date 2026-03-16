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
SFTRolloutWorker - Pass-through rollout worker for supervised fine-tuning.

Converts pre-tokenized data from StreamingDatasetDataLoader into Trajectories
without performing any LLM inference.
"""

import logging
from typing import Any

from ..nexrl_types import Trajectory
from .base_rollout_worker import BaseRolloutWorker

logger = logging.getLogger(__name__)


class SFTRolloutWorker(BaseRolloutWorker):
    """
    Rollout worker for SFT that converts pre-tokenized data to Trajectories.

    No LLM inference is performed.  The worker simply re-packages the
    ``input_ids`` / ``labels`` dict from the data loader into a
    :class:`Trajectory` with the appropriate ``loss_mask``.

    Config:
        - need_llm_inference: Must be ``false`` (no inference service needed).
        - identifier: Model tag used for trajectory routing (default: "default").
    """

    def rollout(self, task: dict[str, Any]) -> str | None:
        """
        Convert a single data-loader item into a Trajectory.

        Args:
            task: Dict with ``input_ids`` (list[int]) and ``labels`` (list[int]).
                  Labels use -100 for tokens that should be ignored in the loss.

        Returns:
            Result of ``_put_trajectory``: 'success', 'fail', or 're-rollout'.
        """
        input_ids = task.get("input_ids")
        labels = task.get("labels")

        if input_ids is None or labels is None:
            logger.error(
                f"SFTRolloutWorker: task missing 'input_ids' or 'labels': {list(task.keys())}"
            )
            return None

        # Build loss_mask: 1 where the model should compute loss, 0 otherwise
        loss_mask = [0 if label == -100 else 1 for label in labels]

        # identifier serves as model_tag for trajectory routing
        identifier = self._config.get("identifier", "default")

        trajectory = Trajectory(
            tokens=input_ids,
            loss_mask=loss_mask,
            reward=0.0,
            is_val=task.get("is_val", False),
            extra_fields={
                "model_tag": identifier,
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
                "task_id": task.get("task_id", 0),
            },
        )

        return self._put_trajectory(trajectory)
