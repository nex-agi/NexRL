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

"""Self-hosted mode specific configuration utilities."""

# Import common utilities
import sys
import warnings
from pathlib import Path
from typing import Any

# Add parent directory to path to import common utilities
CLI_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CLI_DIR))


def load_identifier_world_sizes(cfg: dict) -> dict[str, int]:
    """Load GPU worker world sizes keyed by identifier from the config file.

    Supports both old and new config structures:

    Old (deprecated) structure:
        resource.train.<identifier>.world_size: 1

    New structure:
        service.train_service.<service-name>.identifier: "student"
        service.train_service.<service-name>.resource.world_size: 1
    """
    # Check for old structure first (backward compatibility)
    train_resources = (cfg.get("resource") or {}).get("train")
    if train_resources and isinstance(train_resources, dict):
        warnings.warn(
            "The 'resource.train' config structure is deprecated. "
            "Please migrate to the new 'service.train_service' structure. "
            "See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )

        world_sizes_old: dict[str, int] = {}
        for identifier, spec in train_resources.items():
            if not isinstance(spec, dict):
                raise ValueError(f"resource.train.{identifier} must be a mapping")
            ws = spec.get("world_size")
            if not isinstance(ws, int) or ws < 1:
                raise ValueError(
                    f"resource.train.{identifier}.world_size must be a positive integer"
                )
            world_sizes_old[identifier] = ws
        return world_sizes_old

    # Use new structure
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

    Supports both old and new config structures:

    Old (deprecated) structure:
        resource.inference.replicas: 4
        resource.inference.gpus_per_replica: 2
        resource.inference.served_model_name: "model-name"
        resource.inference.model_path: "/path/to/model"
        resource.inference.backend: "bp-sglang"
        resource.inference.extra_args: ""

    New structure:
        service.inference_service.model: "model-name"
        service.inference_service.model_path: "/path/to/model"
        service.inference_service.resource.replicas: 4
        service.inference_service.resource.gpus_per_replica: 2
        service.inference_service.resource.backend: "bp-sglang"
        service.inference_service.resource.extra_args: ""
    """
    # Check for old structure first (backward compatibility)
    inference_resources = (cfg.get("resource") or {}).get("inference")
    if inference_resources and isinstance(inference_resources, dict):
        warnings.warn(
            "The 'resource.inference' config structure is deprecated. "
            "Please migrate to the new 'service.inference_service' structure. "
            "See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )

        spec = inference_resources
        replicas = spec.get("replicas")
        gpus_per_replica = spec.get("gpus_per_replica")
        served_model_name = spec.get("served_model_name")
        model_path = spec.get("model_path")
        backend = spec.get("backend")
        extra_args = spec.get("extra_args", "")

        for key, value in {
            "replicas": replicas,
            "gpus_per_replica": gpus_per_replica,
        }.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"resource.inference.{key} must be a positive integer")

        if not isinstance(served_model_name, str) or not served_model_name:
            raise ValueError("resource.inference.served_model_name must be a non-empty string")
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("resource.inference.model_path must be a non-empty string")

        return {
            "replicas": replicas,
            "gpus_per_replica": gpus_per_replica,
            "served_model_name": served_model_name,
            "model_path": model_path,
            "backend": backend,
            "extra_args": extra_args,
        }

    # Use new structure
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
    backend = inference_resource.get("backend", "bp-sglang")
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
