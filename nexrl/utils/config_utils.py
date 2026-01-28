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


def get_actor_train_service_config_by_name(
    train_service: DictConfig, actor_train_service_name: str
) -> DictConfig:
    """
    Get the actor (main) train service configuration by name.

    Args:
        train_service: The train_service configuration dict
        actor_train_service_name: The name of the actor train service (e.g., "student")

    Returns:
        The actor train service config

    Raises:
        ValueError: If actor train service is not found
    """
    if not actor_train_service_name:
        raise ValueError("actor_train_service_name must be specified")

    # Validate that the actor service exists
    if actor_train_service_name not in train_service:
        # Note: With OmegaConf, nested configs are DictConfig objects, not regular dicts
        available_services = [
            k for k in train_service.keys() if isinstance(train_service[k], (dict, DictConfig))
        ]
        raise ValueError(
            f"Actor train service '{actor_train_service_name}' not found in train_service. "
            f"Available services: {available_services}"
        )

    return train_service[actor_train_service_name]


def get_actor_train_service_config(config: DictConfig) -> DictConfig:
    """
    Get the actor (main) train service configuration from full config.

    Args:
        config: The full configuration with service.actor_train_service and service.train_service

    Returns:
        The actor train service config

    Raises:
        ValueError: If actor train service is not specified or not found
    """
    actor_train_service_name = config.service.get("actor_train_service")
    if not actor_train_service_name:
        raise ValueError("service.actor_train_service must be specified")

    return get_actor_train_service_config_by_name(
        config.service.train_service, actor_train_service_name
    )


def use_tinker(config: DictConfig) -> bool:
    """
    Check if Tinker is used in the configuration.

    Args:
        config: The configuration

    Returns:
        True if Tinker is used, False otherwise
    """
    actor_train_service = get_actor_train_service_config(config)
    return actor_train_service.backend == "tinker"


def use_weaver(config: DictConfig) -> bool:
    """
    Check if Weaver is used in the configuration.
    """
    actor_train_service = get_actor_train_service_config(config)
    return actor_train_service.backend == "weaver"
