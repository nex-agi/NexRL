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

    Supports backward compatibility:
    - Old flat structure: train_service is directly a config (no nested services)
    - Missing role field: If only one service, assume it's the requested role

    Args:
        train_service: The train_service configuration dict
        role: The role to search for (e.g., "actor", "teacher")

    Returns:
        The train service config with the matching role

    Raises:
        ValueError: If no service with the specified role is found
    """
    import warnings

    # Check if this is the old flat structure (has backend/url directly)
    # Old structure: config.service.train_service.backend exists
    # New structure: config.service.train_service.<name>.backend exists
    if "backend" in train_service or "url" in train_service:
        warnings.warn(
            "The flat train_service config structure is deprecated. "
            "Please migrate to the new nested structure with explicit role field. "
            "See migration guide in docs/developer-guide/09-recipes/.",
            DeprecationWarning,
            stacklevel=3,
        )
        # In old structure, there's only one service, so return it for any role
        return train_service

    # New structure: search for service with matching role
    # Note: With OmegaConf, nested configs are DictConfig objects
    services_with_role = []
    services_without_role = []

    for service_name, service_config in train_service.items():
        if isinstance(service_config, (dict, DictConfig)):
            if service_config.get("role") == role:
                services_with_role.append((service_name, service_config))
            elif "backend" in service_config or "url" in service_config:
                # This looks like a service config without role
                services_without_role.append((service_name, service_config))

    # If we found services with the matching role, return the first one
    if services_with_role:
        return services_with_role[0][1]

    # If no services have roles, but there's only one service, assume it's the requested role
    if not services_with_role and len(services_without_role) == 1:
        warnings.warn(
            f"Train service '{services_without_role[0][0]}' is missing 'role' field. "
            f"Assuming role='{role}'. Please add explicit role field. "
            "See migration guide in docs/developer-guide/09-recipes/.",
            DeprecationWarning,
            stacklevel=3,
        )
        return services_without_role[0][1]

    # Multiple services without roles - cannot determine which is which
    if len(services_without_role) > 1:
        service_names = [name for name, _ in services_without_role]
        raise ValueError(
            f"Multiple train services found without 'role' field: {service_names}. "
            f"Cannot determine which is '{role}'. Please add explicit 'role' field to each service."
        )

    # No matching services found
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
