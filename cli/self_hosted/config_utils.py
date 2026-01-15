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
from pathlib import Path
from typing import Any

# Add parent directory to path to import common utilities
CLI_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CLI_DIR))


def load_identifier_world_sizes(cfg: dict) -> dict[str, int]:
    """Load GPU worker world sizes keyed by identifier from the config file."""
    train_resources = (cfg.get("resource") or {}).get("train") or {}
    if not isinstance(train_resources, dict) or not train_resources:
        raise ValueError("resource.train must map identifiers to resource configs")

    world_sizes: dict[str, int] = {}
    for identifier, spec in train_resources.items():
        if not isinstance(spec, dict):
            raise ValueError(f"resource.train.{identifier} must be a mapping")
        ws = spec.get("world_size")
        if not isinstance(ws, int) or ws < 1:
            raise ValueError(f"resource.train.{identifier}.world_size must be a positive integer")
        world_sizes[identifier] = ws
    return world_sizes


def load_inference_resource(cfg: dict) -> dict[str, Any]:
    """Load inference resource defaults from config."""
    inference_resources = (cfg.get("resource") or {}).get("inference") or {}
    if not isinstance(inference_resources, dict) or not inference_resources:
        raise ValueError("resource.inference must be a mapping")

    spec = inference_resources

    replicas = spec.get("replicas")
    gpus_per_replica = spec.get("gpus_per_replica")
    served_model_name = spec.get("served_model_name")
    model_path = spec.get("model_path")
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
        "extra_args": extra_args,
    }
