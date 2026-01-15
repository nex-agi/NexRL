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

"""Path utilities for NexRL framework."""

import logging
import os

logger = logging.getLogger(__name__)


def resolve_path_from_config(path: str | None, config_file_path: str | None = None) -> str | None:
    """
    Resolve a path relative to the config file's location.

    This is the preferred method for resolving paths in recipe configs,
    as it makes recipes self-contained and independent of working directory.

    Priority:
    1. Absolute path - use as-is
    2. Relative path - resolve relative to config file's directory

    Args:
        path: Path to resolve (can be None, absolute, or relative)
        config_file_path: Path to the config file (to determine base directory)

    Returns:
        Resolved absolute path, or None if input is None

    Raises:
        ValueError: If path cannot be resolved
    """
    if path is None:
        return None

    # Convert to string if Path object
    path_str = str(path)

    # If absolute path, return as-is
    if os.path.isabs(path_str):
        if not os.path.exists(path_str):
            logger.warning(f"Absolute path does not exist: {path_str}")
        return path_str

    # Relative path - resolve relative to config file's directory
    if config_file_path is None:
        raise ValueError(
            f"Cannot resolve relative path '{path_str}': " "config_file_path not provided"
        )

    config_dir = os.path.dirname(os.path.abspath(config_file_path))
    resolved = os.path.join(config_dir, path_str)
    resolved = os.path.abspath(resolved)

    if not os.path.exists(resolved):
        logger.warning(
            f"Resolved path does not exist: {resolved} (from config: {config_file_path})"
        )

    return resolved


def resolve_agent_config_path(
    config_path: str | None, recipe_config_path: str | None = None
) -> str | None:
    """
    Resolve agent config path relative to the recipe config file.

    Args:
        config_path: Agent config path from configuration
        recipe_config_path: Path to the recipe config file (required for relative paths)

    Returns:
        Resolved absolute path to agent config

    Raises:
        ValueError: If relative path provided without recipe_config_path
    """
    return resolve_path_from_config(config_path, recipe_config_path)


def resolve_evaluator_module_path(
    module_path: str | None,
    config_file_path: str | None = None,
) -> str | None:
    """
    Resolve evaluator module path relative to the config file.

    Supports formats:
    - Absolute: "/path/to/evaluator.py:ClassName"
    - Relative to config file: "agent_workspace/evaluator.py:ClassName"

    Args:
        module_path: Evaluator module path in format "path/to/file.py:ClassName"
        config_file_path: Path to the config file (required for relative paths)

    Returns:
        Resolved module path with absolute file path

    Raises:
        ValueError: If relative path provided without config_file_path
    """
    if module_path is None:
        return None

    if ":" not in module_path:
        raise ValueError(
            f"Invalid evaluator module path format. "
            f"Expected 'path:ClassName', got: {module_path}"
        )

    file_path, class_name = module_path.rsplit(":", 1)

    # Resolve file path
    resolved_file: str | None
    if os.path.isabs(file_path):
        resolved_file = file_path
    else:
        # Relative to config file
        resolved_file = resolve_path_from_config(file_path, config_file_path)
        if resolved_file is None:
            raise ValueError(f"Could not resolve path: {file_path}")

    if not os.path.exists(resolved_file):
        raise FileNotFoundError(f"Evaluator file does not exist: {resolved_file}")

    return f"{resolved_file}:{class_name}"
