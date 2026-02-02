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

import logging

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


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


def migrate_legacy_config(config: DictConfig):  # pylint: disable=protected-access
    """Centralized migration of all legacy config structures to new format.

    This function handles ALL backward compatibility transformations in one place:
    1. model_tag → identifier
    2. resource.train → service.train_service.*.resource
    3. resource.inference → service.inference_service.resource
    4. resource.agent → rollout_worker.resource
    5. Flat train_service → nested with role
    6. Add missing role fields

    To remove backward compatibility in future: just delete this function and its call.
    """
    # pylint: disable=protected-access
    import warnings

    from omegaconf import OmegaConf  # pylint: disable=reimported,redefined-outer-name,unused-import

    # Disable struct mode globally to allow adding new keys
    OmegaConf.set_struct(config, False)

    migrations_applied = []

    # ============================================================================
    # 1. Migrate model_tag → identifier (inference_service)
    # ============================================================================
    inference_service = config.service.inference_service
    if "model_tag" in inference_service and "identifier" not in inference_service:
        inference_service["identifier"] = inference_service["model_tag"]
        migrations_applied.append("inference_service.model_tag → identifier")

    # ============================================================================
    # 2. Migrate flat train_service → nested structure with role
    # ============================================================================
    train_service = config.service.train_service

    # Check where 'backend' is located to determine structure
    has_top_level_backend = "backend" in train_service or "url" in train_service

    # Check if any nested dicts have 'backend' (indicating nested services)
    nested_services_with_backend = [
        k
        for k, v in train_service.items()
        if (isinstance(v, dict) or OmegaConf.is_dict(v)) and ("backend" in v or "url" in v)
    ]

    # Ambiguous: both top-level and nested backends exist
    if has_top_level_backend and len(nested_services_with_backend) > 0:
        raise ValueError(
            "Cannot auto-migrate: found 'backend' at both top-level and nested levels in train_service. "
            "This configuration is ambiguous. Please manually update to the new format."
        )

    is_flat_structure = has_top_level_backend
    if is_flat_structure:

        # Migrate flat structure to nested
        old_config = dict(train_service)
        train_service.clear()

        # Create nested structure with default name
        service_name = "main_actor"
        train_service[service_name] = old_config
        train_service[service_name]["role"] = "actor"

        # Migrate model_tag to identifier if exists
        if (
            "model_tag" in train_service[service_name]
            and "identifier" not in train_service[service_name]
        ):
            train_service[service_name]["identifier"] = train_service[service_name]["model_tag"]
            del train_service[service_name]["model_tag"]

        migrations_applied.append(f"train_service: flat → nested ('{service_name}')")

    # ============================================================================
    # 3. Migrate model_tag → identifier (train_service)
    # ============================================================================
    for service_name, service_config in train_service.items():
        if not (isinstance(service_config, dict) or OmegaConf.is_dict(service_config)):
            continue
        if "model_tag" in service_config and "identifier" not in service_config:
            service_config["identifier"] = service_config["model_tag"]
            migrations_applied.append(f"train_service.{service_name}.model_tag → identifier")

    # ============================================================================
    # 4. Add missing role field (default to actor if single service)
    # ============================================================================
    service_names = [
        k for k, v in train_service.items() if isinstance(v, dict) or OmegaConf.is_dict(v)
    ]
    services_with_role = [k for k in service_names if train_service[k].get("role")]

    if len(service_names) == 1 and len(services_with_role) == 0:
        service_name = service_names[0]
        train_service[service_name]["role"] = "actor"
        migrations_applied.append(f"train_service.{service_name}: added role='actor'")

    # ============================================================================
    # 5. Migrate resource.train → train_service.*.resource
    # ============================================================================
    old_train_resources = config.get("resource", {}).get("train") if "resource" in config else None
    logger.info(f"[DEBUG] old_train_resources: {old_train_resources}")
    logger.info(f"[DEBUG] old_train_resources type: {type(old_train_resources)}")
    logger.info(f"[DEBUG] train_service keys: {list(train_service.keys())}")

    if old_train_resources and (
        isinstance(old_train_resources, dict) or OmegaConf.is_dict(old_train_resources)
    ):
        train_service_items = [
            (k, v) for k, v in train_service.items() if isinstance(v, dict) or OmegaConf.is_dict(v)
        ]
        logger.info(f"[DEBUG] train_service_items: {[name for name, _ in train_service_items]}")

        for service_name, service_config in train_service_items:
            # Get identifier from service config
            service_identifier = service_config.get("identifier")
            logger.info(
                f"[DEBUG] Processing service '{service_name}' with identifier '{service_identifier}'"
            )
            resource_spec = None
            resource_key = None

            # Try exact match first
            if service_identifier and service_identifier in old_train_resources:
                resource_key = service_identifier
                resource_spec = old_train_resources[service_identifier]
                logger.info(f"[DEBUG] Exact match found for '{service_identifier}'")
            # Fallback: if only one train service and only one resource entry, use it
            elif len(train_service_items) == 1 and len(old_train_resources) == 1:
                resource_key = next(iter(old_train_resources))
                resource_spec = old_train_resources[resource_key]
                logger.info(
                    f"[DEBUG] Fallback match: using '{resource_key}' for service '{service_name}'"
                )
            # Multiple entries with no match - error
            elif len(train_service_items) > 1 or len(old_train_resources) > 1:
                raise ValueError(
                    f"Cannot auto-migrate resource.train: identifier mismatch. "
                    f"Service '{service_name}' has identifier '{service_identifier}', "
                    f"but resource.train has keys: {list(old_train_resources.keys())}. "
                    f"Please manually update identifiers to match."
                )

            if resource_spec and (
                isinstance(resource_spec, dict) or OmegaConf.is_dict(resource_spec)
            ):
                logger.info(f"[DEBUG] Adding resource to service '{service_name}': {resource_spec}")
                # Ensure resource section exists
                if "resource" not in service_config:
                    service_config["resource"] = {}

                # Copy resource fields that don't already exist
                for key, value in resource_spec.items():
                    if key not in service_config["resource"]:
                        service_config["resource"][key] = value

                migrations_applied.append(
                    f"resource.train.{resource_key} → " f"train_service.{service_name}.resource"
                )
            else:
                logger.info(f"[DEBUG] No resource_spec found for service '{service_name}'")

    # ============================================================================
    # 6. Migrate resource.inference → inference_service.resource
    # ============================================================================
    old_inference_resource = (config.get("resource") or {}).get("inference")
    if old_inference_resource and (
        isinstance(old_inference_resource, dict) or OmegaConf.is_dict(old_inference_resource)
    ):
        # Migrate model/path fields
        if "served_model_name" in old_inference_resource:
            if "model" not in inference_service:
                inference_service["model"] = old_inference_resource["served_model_name"]
                migrations_applied.append(
                    "resource.inference.served_model_name → inference_service.model"
                )

        if "model_path" in old_inference_resource:
            if "model_path" not in inference_service:
                inference_service["model_path"] = old_inference_resource["model_path"]
                migrations_applied.append(
                    "resource.inference.model_path → inference_service.model_path"
                )

        # Migrate resource fields - copy all fields dynamically
        if "resource" not in inference_service:
            inference_service["resource"] = {}

        # Skip top-level fields that are migrated separately (model_path, served_model_name)
        skip_fields = {"model_path", "served_model_name"}

        for field, value in old_inference_resource.items():
            if field not in skip_fields and field not in inference_service["resource"]:
                inference_service["resource"][field] = value
                migrations_applied.append(
                    f"resource.inference.{field} → inference_service.resource.{field}"
                )

    # ============================================================================
    # 7. Migrate resource.agent → rollout_worker.resource
    # ============================================================================
    old_agent_resource = (config.get("resource") or {}).get("agent")
    if old_agent_resource and (
        isinstance(old_agent_resource, dict) or OmegaConf.is_dict(old_agent_resource)
    ):
        if "rollout_worker" not in config:
            config["rollout_worker"] = {}

        if "resource" not in config.rollout_worker:
            config.rollout_worker["resource"] = {}

        agent_fields = ["num_workers", "agents_per_worker"]
        for field in agent_fields:
            if field in old_agent_resource:
                if field not in config.rollout_worker.resource:
                    config.rollout_worker.resource[field] = old_agent_resource[field]
                    migrations_applied.append(
                        f"resource.agent.{field} → rollout_worker.resource.{field}"
                    )

    # ============================================================================
    # Note: We keep struct mode disabled after migration because the dynamically
    # added keys (identifier, role, resource, etc.) are not part of the original
    # YAML schema. Re-enabling struct mode would prevent access to these keys.
    # ============================================================================

    # ============================================================================
    # Log all migrations
    # ============================================================================
    if migrations_applied:
        warnings.warn(
            "Legacy configuration detected. Applied backward compatibility migrations:\n  - "
            + "\n  - ".join(migrations_applied)
            + "\n\nPlease update your config to the new format. "
            "See docs/developer-guide/09-recipes/config-migration-guide.md",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info(f"[Backward Compatibility] Applied {len(migrations_applied)} migrations")
