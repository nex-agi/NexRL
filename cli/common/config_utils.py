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

"""Common configuration utilities shared between deployment modes."""

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def load_config(config_path: Path) -> dict:
    """Load config using Hydra to properly resolve defaults and interpolations.

    Note: We don't resolve interpolations here (resolve=False) because some values
    reference environment variables (like ${oc.env:EXPERIMENT_PATH}) that aren't
    set yet. They'll be resolved later when the actual training job runs.
    """
    config_path = config_path.resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem

    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Compose the configuration (this resolves defaults but not interpolations)
        cfg = compose(config_name=config_name)
        # Convert to dict, keeping interpolations unresolved
        return OmegaConf.to_container(cfg, resolve=False)


def load_agent_settings(cfg: dict) -> dict[str, Any]:
    """Load agent worker settings from config.

    Supports both old and new config structures:

    Old (deprecated) structure:
        resource.agent.num_workers: 1
        resource.agent.agents_per_worker: 32

    New structure:
        rollout_worker.resource.num_workers: 1
        rollout_worker.resource.agents_per_worker: 32
    """
    # Check for old structure first (backward compatibility)
    agent_cfg = (cfg.get("resource") or {}).get("agent")
    if agent_cfg and isinstance(agent_cfg, dict):
        import warnings

        warnings.warn(
            "The 'resource.agent' config structure is deprecated. "
            "Please migrate to the new 'rollout_worker.resource' structure. "
            "See migration guide in docs/developer-guide/09-recipes/.",
            DeprecationWarning,
            stacklevel=2,
        )

        if "num_workers" not in agent_cfg or "agents_per_worker" not in agent_cfg:
            raise ValueError("resource.agent must define num_workers and agents_per_worker")

        num_workers = agent_cfg.get("num_workers")
        agents_per_worker = agent_cfg.get("agents_per_worker")

        for key, value in {
            "num_workers": num_workers,
            "agents_per_worker": agents_per_worker,
        }.items():
            if not isinstance(value, int):
                raise ValueError(f"resource.agent.{key} must be an integer")

        return {"num_workers": num_workers, "agents_per_worker": agents_per_worker}

    # Use new structure
    rollout_worker = cfg.get("rollout_worker") or {}
    if not isinstance(rollout_worker, dict):
        raise ValueError("rollout_worker must be a mapping")

    agent_resource = rollout_worker.get("resource") or {}
    if not isinstance(agent_resource, dict):
        raise ValueError("rollout_worker.resource must be a mapping when provided")

    if "num_workers" not in agent_resource or "agents_per_worker" not in agent_resource:
        raise ValueError("rollout_worker.resource must define num_workers and agents_per_worker")

    num_workers = agent_resource.get("num_workers")
    agents_per_worker = agent_resource.get("agents_per_worker")
    for key, value in {"num_workers": num_workers, "agents_per_worker": agents_per_worker}.items():
        if not isinstance(value, int):
            raise ValueError(f"rollout_worker.resource.{key} must be an integer")

    return {"num_workers": num_workers, "agents_per_worker": agents_per_worker}
