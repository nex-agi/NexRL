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

"""Configuration loader with support for Hydra-style config composition"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def resolve_config_references(
    config: DictConfig, config_dir: Path, config_root: Path | None = None
) -> DictConfig:
    """
    Resolve configuration references (Hydra-style defaults)

    Args:
        config: The main configuration
        config_dir: Directory containing configuration files (for relative paths)
        config_root: Root directory for configuration files (for absolute paths starting with /)

    Returns:
        Resolved configuration with references expanded
    """
    # If config_root not provided, use config_dir as root
    if config_root is None:
        config_root = config_dir

    # Check for defaults section
    if "defaults" in config:
        defaults = config.defaults
        resolved_configs = []

        for default in defaults:
            if isinstance(default, str):
                # Simple reference: load the config file
                if default == "_self_":
                    continue  # Skip self reference

                # Handle absolute paths (starting with /)
                if default.startswith("/"):
                    # Absolute path from config root
                    config_path = config_root / f"{default[1:]}.yaml"
                else:
                    # Relative path from config_dir
                    config_path = config_dir / f"{default}.yaml"

                if config_path.exists():
                    sub_config = OmegaConf.load(config_path)
                    # Recursively resolve references in the sub-config
                    sub_config = resolve_config_references(
                        sub_config, config_path.parent, config_root
                    )
                    resolved_configs.append(sub_config)
                else:
                    # Log warning but don't fail - config might be optional
                    logger.warning(f"Config file not found: {config_path}")

            elif isinstance(default, dict):
                # Named reference: e.g., data: torch
                for key, value in default.items():
                    # Handle absolute paths
                    if value.startswith("/"):
                        config_path = config_root / f"{value[1:]}.yaml"
                    else:
                        config_path = config_dir / key / f"{value}.yaml"

                    if config_path.exists():
                        sub_config = OmegaConf.load(config_path)
                        # Recursively resolve references
                        sub_config = resolve_config_references(
                            sub_config, config_path.parent, config_root
                        )
                        # Store under the specified key
                        resolved_configs.append({key: sub_config})
                    else:
                        logger.warning(f"Config file not found: {config_path}")

        # Merge all configs (defaults first, then main config)
        base_config = OmegaConf.create({})
        for cfg in resolved_configs:
            base_config = OmegaConf.merge(base_config, cfg)

        # Remove defaults from main config before merging
        config_without_defaults = OmegaConf.create(config)
        if "defaults" in config_without_defaults:
            del config_without_defaults["defaults"]

        # Merge main config last (it has priority)
        config = OmegaConf.merge(base_config, config_without_defaults)

    return config


def interpolate_config_values(config: DictConfig) -> DictConfig:
    """
    Interpolate configuration values (resolve ${...} references)

    Args:
        config: Configuration with potential interpolations

    Returns:
        Configuration with interpolated values
    """
    # OmegaConf automatically handles interpolation when accessing values
    # But we need to resolve all interpolations explicitly
    OmegaConf.resolve(config)
    return config


def load_config_with_references(config_path: str, config_root: Path | None = None) -> DictConfig:
    """
    Load configuration file with support for references and interpolation

    Args:
        config_path: Path to the main configuration file
        config_root: Root directory for configuration files (auto-detected if not provided)

    Returns:
        Fully resolved configuration
    """
    config_path_obj: Path = Path(config_path)
    config_dir: Path = config_path_obj.parent

    # Auto-detect config root by looking for common structure
    # If we're in a subdirectory like worker/, go up to find the root
    if config_root is None:
        # Try to find config root by looking for base/ directory
        search_path: Path = config_dir
        for _ in range(5):  # Search up to 5 levels
            if (search_path / "base").exists():
                config_root = search_path
                break
            if search_path.parent == search_path:  # Reached filesystem root
                break
            search_path = search_path.parent

        # If not found, use config_dir as root
        if config_root is None:
            config_root = config_dir

    # Load main configuration
    config = OmegaConf.load(config_path_obj)

    # Resolve references (Hydra-style defaults)
    config = resolve_config_references(config, config_dir, config_root)

    # Interpolate values
    config = interpolate_config_values(config)

    return config


# merge_megatron_config removed - Megatron support removed


def load_nextrainer_config(
    config_path: str, overrides: dict[str, Any] | None = None, config_root: Path | None = None
) -> DictConfig:
    """
    Load NexTrainer configuration with full support for references and overrides

    Args:
        config_path: Path to configuration file
        overrides: Dictionary of values to override in the config
        config_root: Root directory for configuration files (auto-detected if not provided)

    Returns:
        Fully resolved configuration
    """
    # Load config with references
    config = load_config_with_references(config_path, config_root)

    # Apply overrides if provided
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)

    # Final resolution of any remaining interpolations
    try:
        OmegaConf.resolve(config)
    except Exception as e:
        # Provide helpful error message if interpolation fails
        logger.error(f"Failed to resolve config interpolations: {e}")
        logger.error("This usually means a referenced key doesn't exist in the config")
        raise

    return config
