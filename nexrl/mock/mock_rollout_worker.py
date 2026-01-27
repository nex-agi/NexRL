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
Mock Rollout Worker for testing purposes
"""

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from ..base_module import NexRLModule
from ..nexrl_types import Trajectory
from ..rollout_worker.base_rollout_worker import BaseRolloutWorker
from .mock_inference_service_client import MockInferenceServiceClient

logger = logging.getLogger(__name__)


def _deterministic_reward(*parts: object) -> float:
    """
    Map inputs deterministically to a binary reward (0.0 or 1.0).
    Stable across processes/runs (unlike Python's built-in hash()).
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Use 1 bit -> deterministic binary reward.
    return float(digest[0] & 1)


class MockRolloutWorker(BaseRolloutWorker):
    """
    Mock Rollout Worker for testing purposes.
    Uses MockInferenceServiceClient instead of real inference service to avoid network dependencies.
    """

    def __init__(self, config: DictConfig):  # pylint: disable=super-init-not-called
        # Initialize NexRLModule (grandparent)
        NexRLModule.__init__(self)  # pylint: disable=non-parent-init-called

        # Initialize attributes from BaseRolloutWorker without calling its __init__
        # to avoid creating a real inference service client
        self._config = config
        self._stop_event = threading.Event()
        self._thread: threading.Thread = None  # type: ignore

        # Use MockInferenceServiceClient instead of real client
        self._inference_client: MockInferenceServiceClient = MockInferenceServiceClient(config)  # type: ignore

        # Initialize other attributes that would be set by BaseRolloutWorker
        self._trajectory_pool = None  # type: ignore
        self._dataloader = None  # type: ignore
        self._weight_sync_controller = None  # type: ignore
        self._validate_dataloader = None  # type: ignore
        self._validator = None  # type: ignore
        self._next_task = None
        self._is_running_validate = False

        # Mock-specific attributes
        self._processed_count: int = 0
        self._mock_delay: float = 0.1

        # Trajectory loading from file
        self._trajectory_load_path = config.get("trajectory_load_path", None)
        self._trajectory_format = config.get("trajectory_format", "jsonl")  # "pt" or "jsonl"
        self._loaded_trajectories: list[Trajectory] = []
        self._traj_key_map: dict[tuple, Trajectory] = {}  # Map (group_id, run_id) -> trajectory
        self._trajectory_index: int = 0

        if self._trajectory_load_path and Path(self._trajectory_load_path).exists():
            self._load_trajectories()
            self._build_traj_key_map()
            logger.info(
                f"MockRolloutWorker initialized with {len(self._loaded_trajectories)} loaded trajectories from {self._trajectory_load_path} (format: {self._trajectory_format})"
            )
        else:
            logger.info(
                "MockRolloutWorker initialized with MockInferenceServiceClient (generating mock trajectories)"
            )

    def _load_trajectories(self) -> None:
        """Load trajectories from dump file.

        Supports two formats:
        - "pt": PyTorch tensor format (from DataDumper)
        - "jsonl": JSON Lines format (legacy)
        """
        if not self._trajectory_load_path:
            return

        try:
            if self._trajectory_format == "pt":
                self._load_trajectories_pt()
            elif self._trajectory_format == "jsonl":
                self._load_trajectories_jsonl()
            else:
                raise ValueError(f"Unsupported trajectory format: {self._trajectory_format}")
            logger.info(
                f"Loaded {len(self._loaded_trajectories)} trajectories from {self._trajectory_load_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load trajectories from {self._trajectory_load_path}: {e}")
            self._loaded_trajectories = []

    def _build_traj_key_map(self) -> None:
        """Build mapping from (group_id, run_id) to trajectory for multi-worker replay."""
        for traj in self._loaded_trajectories:
            group_id = traj.get("group_id", None)
            run_id = traj.get("run_id", None)
            if group_id is not None and run_id is not None:
                key = (str(group_id), int(run_id))
                self._traj_key_map[key] = traj
        if self._traj_key_map:
            logger.info(
                f"Built traj_key_map with {len(self._traj_key_map)} entries: {list(self._traj_key_map.keys())}"
            )

    def _load_trajectories_pt(self) -> None:
        """Load trajectories from PyTorch .pt file (DataDumper format)."""
        data = torch.load(self._trajectory_load_path)

        # Handle DataDumper format: {"step": int, "num_trajectories": int, "trajectories": [...]}
        trajectories_data = data.get("trajectories", [])

        for traj_dict in trajectories_data:
            # Convert tensor lists back to Python lists if needed
            tokens = traj_dict["tokens"]
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()

            loss_mask = traj_dict["loss_mask"]
            if isinstance(loss_mask, torch.Tensor):
                loss_mask = loss_mask.tolist()

            reward = traj_dict["reward"]
            if isinstance(reward, torch.Tensor):
                reward = reward.item()

            is_val = traj_dict["is_val"]
            if isinstance(is_val, torch.Tensor):
                is_val = bool(is_val.item())

            # Handle extra_fields - convert any tensors to lists
            extra_fields = traj_dict.get("extra_fields", {})
            if extra_fields:
                extra_fields = self._convert_extra_fields(extra_fields)

            trajectory = Trajectory(
                tokens=tokens,
                loss_mask=loss_mask,
                reward=reward,
                is_val=is_val,
                extra_fields=extra_fields,
            )
            self._loaded_trajectories.append(trajectory)

    def _load_trajectories_jsonl(self) -> None:
        """Load trajectories from JSON Lines file (legacy format)."""
        with open(self._trajectory_load_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                for traj_dict in data.get("trajectories", []):
                    trajectory = Trajectory(
                        tokens=traj_dict["tokens"],
                        loss_mask=traj_dict["loss_mask"],
                        reward=traj_dict["reward"],
                        is_val=traj_dict["is_val"],
                        extra_fields=traj_dict.get("extra_fields", {}),
                    )
                    self._loaded_trajectories.append(trajectory)

    def _convert_extra_fields(self, extra_fields: dict) -> dict:
        """Convert tensor values in extra_fields to Python types."""
        result = {}
        for key, value in extra_fields.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    result[key] = value.item()
                else:
                    result[key] = value.tolist()
            elif isinstance(value, dict):
                result[key] = self._convert_extra_fields(value)
            elif isinstance(value, list):
                result[key] = [v.tolist() if isinstance(v, torch.Tensor) else v for v in value]
            else:
                result[key] = value
        return result

    def rollout(self, task: dict[str, Any]) -> str | None:
        """
        Mock implementation of a single step operation.
        If trajectories are loaded from file, replay them; otherwise generate mock trajectories.

        Args:
            task: Task to process
        """
        # Simulate processing time
        time.sleep(self._mock_delay)

        # If we have loaded trajectories, use them instead of generating
        if self._loaded_trajectories:
            group_id = str(task.get("group_id") or task.get("uid", ""))
            run_id = int(task.get("run_id", 0))
            key = (group_id, run_id)

            # Try to find trajectory by (group_id, run_id)
            if key in self._traj_key_map:
                trajectory = self._traj_key_map[key]
                self._processed_count += 1
                logger.info(
                    f"Mock rollout worker: Replaying trajectory by key={key}, reward={trajectory.reward}"
                )
                return self._put_trajectory(trajectory)

            # Fallback: use sequential index
            if self._trajectory_index >= len(self._loaded_trajectories):
                logger.warning(
                    f"Reached end of loaded trajectories ({len(self._loaded_trajectories)}), wrapping around"
                )
                self._trajectory_index = 0
            trajectory = self._loaded_trajectories[self._trajectory_index]
            self._trajectory_index += 1
            self._processed_count += 1
            logger.info(
                f"Mock rollout worker: Replaying trajectory #{self._processed_count} (sequential, key={key} not found), reward={trajectory.reward}"
            )
            return self._put_trajectory(trajectory)

        # Otherwise, generate mock trajectory
        # GRPO grouping fields: propagate from task (preferred), otherwise generate defaults.
        # - group_id: identifies the group of repeated rollouts for the same prompt
        # - run_id: identifies the sample index within the group
        # - uid: some configs / pools use uid as the grouping key
        mock_group_id = task.get("group_id") or task.get("uid")
        if not mock_group_id:
            # Deterministic fallback to keep mock runs reproducible.
            mock_group_id = (
                f"mock-{hashlib.sha256(task['prompt'].encode('utf-8')).hexdigest()[:16]}"
            )
        run_id = int(task.get("run_id", 0))
        task_id = task.get("task_id")
        uid = task.get("uid") or mock_group_id
        # Generate mock tokens and loss_mask (simple mock: 10 prompt tokens, 5 response tokens)
        prompt_tokens = list(range(100, 110))
        response_tokens = list(range(200, 205))
        tokens = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)

        # Create Trajectory dataclass
        mock_trajectory = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=_deterministic_reward(task.get("is_val", False), run_id, task["prompt"]),
            is_val=task.get("is_val", False),
            extra_fields={
                # GRPO-required fields
                "group_id": mock_group_id,
                "run_id": run_id,
                "uid": uid,
                # Stable ordering key (preferred over uuid-like group_id)
                "task_id": task_id,
                "prompt": task["prompt"],
                "response": f"Mock response (group_id={mock_group_id}, run_id={run_id})",
                "logprobs": [0.0] * len(tokens),
            },
        )

        self._processed_count += 1
        logger.info(f"Mock rollout worker: Successfully processed step #{self._processed_count}")

        return self._put_trajectory(mock_trajectory)
