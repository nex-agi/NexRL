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

"""Common configuration utilities shared across all CLI modes."""

import warnings
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Enable DeprecationWarning to be visible by default in CLI
warnings.simplefilter("always", DeprecationWarning)


def migrate_legacy_config(cfg: dict) -> dict:
    """Centralized migration of all legacy config structures.

    This function handles ALL backward compatibility transformations:
    1. model_tag → identifier (inference_service)
    2. Flat train_service → nested with role
    3. model_tag → identifier (train_service)
    4. Add missing role fields
    5. backend: http/nextrainer → direct-zmq
    6. resource.train → service.train_service.*.resource
    7. resource.inference → service.inference_service.resource
    8. resource.agent → rollout_worker.resource

    To remove backward compatibility in v3.0: just delete this function and its calls.
    """
    migrations_applied = []

    # Ensure service section exists
    if "service" not in cfg:
        cfg["service"] = {}

    # ============================================================================
    # 1. Migrate model_tag → identifier (inference_service)
    # ============================================================================
    if "inference_service" in cfg.get("service", {}):
        inference_service = cfg["service"]["inference_service"]
        if "model_tag" in inference_service and "identifier" not in inference_service:
            inference_service["identifier"] = inference_service["model_tag"]
            migrations_applied.append("inference_service.model_tag → identifier")

    # ============================================================================
    # 2. Migrate flat train_service → nested structure
    # ============================================================================
    if "train_service" in cfg.get("service", {}):
        train_service = cfg["service"]["train_service"]

        # Check where 'backend' is located to determine structure
        has_top_level_backend = "backend" in train_service or "url" in train_service

        # Check if any nested dicts have 'backend' (indicating nested services)
        nested_services_with_backend = [
            k
            for k, v in train_service.items()
            if isinstance(v, dict) and ("backend" in v or "url" in v)
        ]

        # Ambiguous: both top-level and nested backends exist
        if has_top_level_backend and len(nested_services_with_backend) > 0:
            raise ValueError(
                "Cannot auto-migrate: found 'backend' at both top-level and nested levels in train_service. "
                "This configuration is ambiguous. Please manually update to the new format."
            )

        is_flat = has_top_level_backend
        if is_flat:

            old_config = dict(train_service)
            train_service.clear()
            service_name = "main_actor"
            train_service[service_name] = old_config
            train_service[service_name]["role"] = "actor"

            if (
                "model_tag" in train_service[service_name]
                and "identifier" not in train_service[service_name]
            ):
                train_service[service_name]["identifier"] = train_service[service_name]["model_tag"]

            migrations_applied.append(f"train_service: flat → nested ('{service_name}')")

    # ============================================================================
    # 3. Migrate model_tag → identifier (train_service)
    # ============================================================================
    if "train_service" in cfg.get("service", {}):
        for service_name, service_config in cfg["service"]["train_service"].items():
            if not isinstance(service_config, dict):
                continue
            if "model_tag" in service_config and "identifier" not in service_config:
                service_config["identifier"] = service_config["model_tag"]
                migrations_applied.append(f"train_service.{service_name}.model_tag → identifier")

    # ============================================================================
    # 4. Add missing role field
    # ============================================================================
    if "train_service" in cfg.get("service", {}):
        train_service = cfg["service"]["train_service"]
        service_names = [k for k, v in train_service.items() if isinstance(v, dict)]
        services_with_role = [k for k in service_names if train_service[k].get("role")]

        if len(service_names) == 1 and len(services_with_role) == 0:
            service_name = service_names[0]
            train_service[service_name]["role"] = "actor"
            migrations_applied.append(f"train_service.{service_name}: added role='actor'")

    # ============================================================================
    # 5. Migrate train service backend: http/nextrainer → direct-zmq
    # ============================================================================
    if "train_service" in cfg.get("service", {}):
        train_service = cfg["service"]["train_service"]
        for service_name, service_config in train_service.items():
            if not isinstance(service_config, dict):
                continue
            backend = service_config.get("backend")
            if backend in ["http", "nextrainer"]:
                service_config["backend"] = "direct-zmq"
                migrations_applied.append(
                    f"train_service.{service_name}.backend: {backend} → direct-zmq"
                )

    # ============================================================================
    # 6. Migrate resource.train → train_service.*.resource
    # ============================================================================
    old_train_resources = (cfg.get("resource") or {}).get("train")
    if (
        old_train_resources
        and isinstance(old_train_resources, dict)
        and "train_service" in cfg.get("service", {})
    ):
        train_service_items = [
            (k, v) for k, v in cfg["service"]["train_service"].items() if isinstance(v, dict)
        ]

        for service_name, service_config in train_service_items:
            service_identifier = service_config.get("identifier") or service_config.get("model_tag")
            resource_spec = None
            resource_key = None

            # Try exact match first
            if service_identifier and service_identifier in old_train_resources:
                resource_key = service_identifier
                resource_spec = old_train_resources[service_identifier]
            # Fallback: if only one train service and only one resource entry, use it
            elif len(train_service_items) == 1 and len(old_train_resources) == 1:
                resource_key = next(iter(old_train_resources))
                resource_spec = old_train_resources[resource_key]
            # Multiple entries with no match - error
            elif len(train_service_items) > 1 or len(old_train_resources) > 1:
                raise ValueError(
                    f"Cannot auto-migrate resource.train: identifier mismatch. "
                    f"Service '{service_name}' has identifier '{service_identifier}', "
                    f"but resource.train has keys: {list(old_train_resources.keys())}. "
                    f"Please manually update identifiers to match."
                )

            if resource_spec and isinstance(resource_spec, dict):
                if "resource" not in service_config:
                    service_config["resource"] = {}

                for key, value in resource_spec.items():
                    if key not in service_config["resource"]:
                        service_config["resource"][key] = value

                migrations_applied.append(
                    f"resource.train.{resource_key} → train_service.{service_name}.resource"
                )

    # ============================================================================
    # 7. Migrate resource.inference → inference_service.resource
    # ============================================================================
    old_inference = (cfg.get("resource") or {}).get("inference")
    if old_inference and "inference_service" in cfg.get("service", {}):
        inference_service = cfg["service"]["inference_service"]

        # Copy served_model_name to model if missing or if it's an interpolation string
        if "served_model_name" in old_inference:
            existing_model = inference_service.get("model", "")
            is_interpolation = isinstance(existing_model, str) and "${" in existing_model
            if not existing_model or is_interpolation:
                inference_service["model"] = old_inference["served_model_name"]
                migrations_applied.append(
                    "resource.inference.served_model_name → inference_service.model"
                )

        # Copy model_path if missing or if it's an interpolation string
        if "model_path" in old_inference:
            existing_path = inference_service.get("model_path", "")
            is_interpolation = isinstance(existing_path, str) and "${" in existing_path
            if not existing_path or is_interpolation:
                inference_service["model_path"] = old_inference["model_path"]
                migrations_applied.append(
                    "resource.inference.model_path → inference_service.model_path"
                )

        if "resource" not in inference_service:
            inference_service["resource"] = {}

        # Skip top-level fields that are migrated separately (model_path, served_model_name)
        skip_fields = {"model_path", "served_model_name"}

        for field, value in old_inference.items():
            if field not in skip_fields and field not in inference_service["resource"]:
                inference_service["resource"][field] = value
                migrations_applied.append(
                    f"resource.inference.{field} → inference_service.resource.{field}"
                )

    # ============================================================================
    # 8. Migrate resource.agent → rollout_worker.resource
    # ============================================================================
    old_agent = (cfg.get("resource") or {}).get("agent")
    if old_agent:
        if "rollout_worker" not in cfg:
            cfg["rollout_worker"] = {}
        if "resource" not in cfg["rollout_worker"]:
            cfg["rollout_worker"]["resource"] = {}

        for field in ["num_workers", "agents_per_worker"]:
            if field in old_agent and field not in cfg["rollout_worker"]["resource"]:
                cfg["rollout_worker"]["resource"][field] = old_agent[field]
                migrations_applied.append(
                    f"resource.agent.{field} → rollout_worker.resource.{field}"
                )

    # ============================================================================
    # Log migrations
    # ============================================================================
    if migrations_applied:
        warnings.warn(
            "Legacy configuration detected. Applied migrations:\n  - "
            + "\n  - ".join(migrations_applied)
            + "\n\nPlease update your config. See docs/developer-guide/09-recipes/config-migration-guide.md",
            DeprecationWarning,
            stacklevel=3,
        )
        print(f"[Backward Compatibility] Applied {len(migrations_applied)} migrations")

    return cfg


def load_config(config_path: Path) -> dict:
    """Load config using Hydra and apply backward compatibility migrations.

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
        cfg_dict = OmegaConf.to_container(cfg, resolve=False)

        # Apply backward compatibility migrations
        cfg_dict = migrate_legacy_config(cfg_dict)

        return cfg_dict


def load_agent_settings(cfg: dict) -> dict[str, Any]:
    """Load agent worker settings from config.

    Expects new structure (migration handled by migrate_legacy_config):
        rollout_worker.resource.num_workers: 1
        rollout_worker.resource.agents_per_worker: 32
    """
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


def load_identifier_world_sizes(cfg: dict) -> dict[str, int]:
    """Load GPU worker world sizes keyed by identifier from the config file.

    Expects new structure (migration handled by migrate_legacy_config):
        service.train_service.<service-name>.identifier: "student"
        service.train_service.<service-name>.resource.world_size: 1
    """
    train_service = (cfg.get("service") or {}).get("train_service") or {}
    if not isinstance(train_service, dict) or not train_service:
        raise ValueError("service.train_service must contain service group configs")

    world_sizes: dict[str, int] = {}
    for service_name, service_spec in train_service.items():
        if not isinstance(service_spec, dict):
            raise ValueError(f"service.train_service.{service_name} must be a mapping")

        identifier = service_spec.get("identifier")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(
                f"service.train_service.{service_name}.identifier must be a non-empty string"
            )

        resource = service_spec.get("resource") or {}
        if not isinstance(resource, dict):
            raise ValueError(f"service.train_service.{service_name}.resource must be a mapping")

        ws = resource.get("world_size")
        if not isinstance(ws, int) or ws < 1:
            raise ValueError(
                f"service.train_service.{service_name}.resource.world_size must be a positive integer"
            )
        world_sizes[identifier] = ws
    return world_sizes


def load_inference_resource(cfg: dict) -> dict[str, Any]:
    """Load inference resource defaults from config.

    Expects new structure (migration handled by migrate_legacy_config):
        service.inference_service.model: "model-name"
        service.inference_service.model_path: "/path/to/model"
        service.inference_service.resource.replicas: 4
        service.inference_service.resource.gpus_per_replica: 2
        service.inference_service.resource.backend: "sglang"
        service.inference_service.resource.extra_args: ""
    """
    inference_service = (cfg.get("service") or {}).get("inference_service") or {}
    if not isinstance(inference_service, dict) or not inference_service:
        raise ValueError("service.inference_service must be a mapping")

    # Get model name and path from inference_service level
    served_model_name = inference_service.get("model", "")
    model_path = inference_service.get("model_path", "")
    if not isinstance(served_model_name, str) or not served_model_name:
        raise ValueError("service.inference_service.model must be a non-empty string")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("service.inference_service.model_path must be a non-empty string")

    # Get resource config
    inference_resource = inference_service.get("resource") or {}
    if not isinstance(inference_resource, dict) or not inference_resource:
        raise ValueError("service.inference_service.resource must be a mapping")

    replicas = inference_resource.get("replicas")
    gpus_per_replica = inference_resource.get("gpus_per_replica")
    backend = inference_resource.get("backend", "sglang")
    extra_args = inference_resource.get("extra_args", "")

    for key, value in {
        "replicas": replicas,
        "gpus_per_replica": gpus_per_replica,
    }.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"service.inference_service.resource.{key} must be a positive integer")

    return {
        "replicas": replicas,
        "gpus_per_replica": gpus_per_replica,
        "served_model_name": served_model_name,
        "model_path": model_path,
        "backend": backend,
        "extra_args": extra_args,
    }


def load_model_name(cfg: dict) -> str:
    """Load model name from service.inference_service.model in config."""
    service_cfg = cfg.get("service") or {}
    if not isinstance(service_cfg, dict):
        raise ValueError("service must be a mapping")

    inference_service_cfg = service_cfg.get("inference_service") or {}
    if not isinstance(inference_service_cfg, dict):
        raise ValueError("service.inference_service must be a mapping")

    model_name = inference_service_cfg.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("service.inference_service.model must be a non-empty string")

    return model_name
