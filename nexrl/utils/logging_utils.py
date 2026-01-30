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

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np

from ..nexrl_types import Trajectory  # pylint: disable=relative-beyond-top-level

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


def set_logging_basic_config():
    """
    This function sets the global logging format and level. It will be called when import nexrl
    """
    import sys

    # Get logging level from environment variable, default to INFO
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Convert string level to logging level constant
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,  # Common alias
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,  # Common alias
    }

    level = level_mapping.get(log_level_str, logging.INFO)

    # Set the logging level while preserving the original format
    # Format: [timestamp][logger_name][level] - message
    log_stream = os.getenv("LOG_STREAM", "")

    # Explicitly set stream to stdout to ensure Ray captures the logs
    if log_stream == "stdout":
        logging.basicConfig(
            level=level,
            format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            stream=sys.stdout,
            force=True,
        )
    else:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        )

    # Suppress NexAU logs - only show WARNING and above
    # This catches all loggers in the nexau.* hierarchy (nexau.archs.*, nexau.core.*, etc.)
    nexau_log_level = os.getenv("NEXAU_LOG_LEVEL", "WARNING").upper()
    nexau_level = level_mapping.get(nexau_log_level, logging.WARNING)
    logging.getLogger("nexau").setLevel(nexau_level)
    logging.getLogger("tools").setLevel(nexau_level)

    print(f"📋 Logging Configuration:")
    print(f"   ├─ NexRL logging level: {logging.getLevelName(level)} ({level})")
    print(f"   └─ NexAU logging level: {logging.getLevelName(nexau_level)} ({nexau_level})")


def log_rollout_metrics(trajectories: list[Trajectory], metrics: dict[str, Any]) -> None:
    if not trajectories:
        return

    scores = [traj.get("score", {}) for traj in trajectories if "score" in traj]
    if scores:
        all_keys: set[str] = set()
        for score_dict in scores:
            if isinstance(score_dict, dict):
                all_keys.update(score_dict.keys())

        for key in all_keys:
            values = []
            for score_dict in scores:
                if isinstance(score_dict, dict) and key in score_dict:
                    val = score_dict[key]
                    if isinstance(val, bool):
                        val = float(val)
                    elif isinstance(val, (int, float)):
                        val = float(val)
                    else:
                        continue
                    values.append(val)
            if values:
                metrics[f"rollout/{key}"] = np.mean(values)

    rewards = [traj["reward"] for traj in trajectories if "reward" in traj]
    if rewards:
        metrics["rollout/reward_mean"] = np.mean(rewards)
        metrics["rollout/reward_std"] = np.std(rewards)
        metrics["rollout/reward_min"] = min(rewards)
        metrics["rollout/reward_max"] = max(rewards)


def log_grpo_metrics(trajectories: list[Trajectory], metrics: dict[str, Any]) -> None:
    group_to_rewards: dict[str, list[float]] = defaultdict(list)
    for traj in trajectories:
        group_id = traj.get("group_id", "unknown")
        group_to_rewards[group_id].append(traj["reward"])

    group_stds = []
    for _, rewards in group_to_rewards.items():
        if len(rewards) > 1:
            group_stds.append(float(np.std(rewards)))

    if group_stds:
        metrics["critic/grpo_std/mean"] = np.mean(group_stds)
        metrics["critic/grpo_std/min"] = float(min(group_stds))
        metrics["critic/grpo_std/max"] = float(max(group_stds))
        metrics["critic/grpo_num_groups"] = len(group_stds)

    advantages = [traj.get("advantage", 0.0) for traj in trajectories]
    if advantages:
        metrics["critic/advantage_mean"] = np.mean(advantages)
        metrics["critic/advantage_std"] = np.std(advantages)
