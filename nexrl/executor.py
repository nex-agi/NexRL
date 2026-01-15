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

import os
from typing import Any

import ray


def _is_ray_remote_method(func) -> bool:
    """
    Detect if a function is a Ray remote method by inspecting its attributes.
    Ray remote methods have specific attributes that distinguish them.

    Args:
        func: The function to check.

    Returns:
        True if the function is a Ray remote method, False otherwise.
    """
    # Check if it's a bound method of a Ray actor
    if hasattr(func, "__self__"):
        obj = func.__self__
        # Ray actors have _ray_actor_id attribute
        if hasattr(obj, "_ray_actor_id"):
            return True
        # Alternative: check if it's a Ray ObjectRef
        if hasattr(obj, "_ray_object_id"):
            return True

    # Check if it's a Ray remote function directly
    if hasattr(func, "remote"):
        return True

    return False


def execute(func: Any, *args, **kwargs) -> Any:
    # TODO: check transferred data size and add warning if it's too large  # pylint: disable=fixme
    """
    Execute a function, automatically detecting whether it's local or Ray remote.

    In local mode: Always execute locally
    In ray mode: Auto-detect based on function type (hybrid support)

    Args:
        func: The function to execute. It can be a local function or a Ray remote function.
        *args: The arguments to pass to the function.
        **kwargs: The keyword arguments to pass to the function.

    Returns:
        The result of the function execution.
    """
    launch_mode = os.getenv("NEXRL_LAUNCH_MODE", "local")
    if launch_mode == "local":
        # Pure local mode - everything is local
        return func(*args, **kwargs)
    elif launch_mode == "ray":
        # Ray mode with hybrid support - auto-detect
        if _is_ray_remote_method(func):
            # Add timeout to prevent indefinite blocking
            # Default timeout is 300 seconds (5 minutes), configurable via environment variable
            timeout = 40
            try:
                return ray.get(func.remote(*args, **kwargs), timeout=timeout)
            except ray.exceptions.GetTimeoutError as exc:
                raise TimeoutError(
                    f"Ray remote task timed out after {timeout} seconds. "
                    f"Function: {func}, Args: {args}, Kwargs: {kwargs}. "
                    f"This may indicate a deadlock, resource starvation, or unresponsive actor. "
                    f"Adjust NEXRL_RAY_GET_TIMEOUT environment variable if needed."
                ) from exc
        else:
            return func(*args, **kwargs)
    else:
        raise ValueError(f"Invalid launch mode: {launch_mode}")


def execute_async(func: Any, *args, **kwargs) -> Any:
    """
    Execute a function asynchronously.

    For local functions: Execute immediately (since we don't have async local execution)
    For Ray functions: Return ObjectRef for async execution

    Args:
        func: The function to execute. It can be a local function or a Ray remote function.
        *args: The arguments to pass to the function.
        **kwargs: The keyword arguments to pass to the function.

    """
    launch_mode = os.getenv("NEXRL_LAUNCH_MODE", "local")
    if launch_mode == "local":
        return func(*args, **kwargs)
    elif launch_mode == "ray":
        if _is_ray_remote_method(func):
            return func.remote(*args, **kwargs)  # Return ObjectRef
        else:
            return func(*args, **kwargs)  # Execute immediately for local
    else:
        raise ValueError(f"Invalid launch mode: {launch_mode}")
