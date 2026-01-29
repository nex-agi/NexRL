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
Configuration utilities for handling OmegaConf DictConfig operations.
"""

from omegaconf import DictConfig, OmegaConf


def insert_config(
    target: DictConfig,
    key: str,
    value: DictConfig,
    restore_struct: bool = True,
) -> None:
    """
    Insert a DictConfig value into a target DictConfig at the specified key.

    This function temporarily disables struct mode to allow dynamic key insertion,
    then optionally restores the original struct mode setting.

    Args:
        target: The target DictConfig to insert into
        key: The key name to insert the value at
        value: The DictConfig value to insert
        restore_struct: Whether to restore struct mode after insertion (default: True)

    Example:
        >>> target_config = OmegaConf.create({"a": 1, "b": 2})
        >>> source_config = OmegaConf.create({"x": 10, "y": 20})
        >>> insert_config(target_config, "new_key", source_config)
        >>> print(target_config.new_key.x)  # 10
    """
    # Store original struct mode setting
    original_struct = OmegaConf.is_struct(target)

    # Temporarily disable struct mode to allow insertion
    OmegaConf.set_struct(target, False)

    try:
        # Insert the value at the specified key
        target[key] = value
    finally:
        # Restore struct mode if requested
        if restore_struct:
            OmegaConf.set_struct(target, original_struct)


def get_train_service_config_by_role(train_service: DictConfig, role: str) -> DictConfig:
    """
    Get a train service configuration by its role.

    Args:
        train_service: The train_service configuration dict
        role: The role to search for (e.g., "actor", "teacher")

    Returns:
        The train service config with the matching role

    Raises:
        ValueError: If no service with the specified role is found
    """
    # Note: With OmegaConf, nested configs are DictConfig objects
    for service_config in train_service.values():
        if isinstance(service_config, (dict, DictConfig)):
            if service_config.get("role") == role:
                return service_config

    available_services = [
        k for k in train_service.keys() if isinstance(train_service[k], (dict, DictConfig))
    ]
    raise ValueError(
        f"Train service with role '{role}' not found in train_service. "
        f"Available services: {available_services}"
    )


def get_actor_train_service_config(config: DictConfig) -> DictConfig:
    """
    Get the actor (main) train service configuration from full config.
    Searches for a service with role="actor".

    Args:
        config: The full configuration with service.train_service

    Returns:
        The actor train service config

    Raises:
        ValueError: If actor train service is not found
    """
    return get_train_service_config_by_role(config.service.train_service, "actor")


def use_tinker(config: DictConfig) -> bool:
    """
    Check if Tinker is used in the configuration.

    Args:
        config: The configuration

    Returns:
        True if Tinker is used, False otherwise
    """
    return config.service.inference_service.backend == "tinker"


def use_weaver(config: DictConfig) -> bool:
    """
    Check if Weaver is used in the configuration.
    """
    return config.service.inference_service.backend == "weaver"
