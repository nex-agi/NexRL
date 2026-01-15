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
Simplified Ray Resource Management for NexRL
Focuses on actor colocation and creation without placement group complexity
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, cast

import ray
from ray.actor import ActorHandle

from .nexrl_types import NexRLRole

logger = logging.getLogger(__name__)


def _get_minimal_env_vars() -> dict[str, str]:
    """
    Get minimal required environment variables for Ray actors.

    This avoids copying the entire os.environ which can be huge in HPC environments,
    causing timeouts and serialization issues.

    Returns:
        dict: Minimal set of environment variables needed by NexRLModules
    """
    env_vars = {
        "NEXRL_LAUNCH_MODE": "ray",
        # Logging configuration
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
        "LOG_STREAM": os.environ.get("LOG_STREAM", ""),
        "NX_PPO_LOGGING_LEVEL": os.environ.get("NX_PPO_LOGGING_LEVEL", "WARN"),
        # Tracking/logging backends
        "NEXRL_USER": os.environ.get("NEXRL_USER", os.environ.get("USER", "")),
        # Experiment paths
        "EXPERIMENT_PATH": os.environ.get("EXPERIMENT_PATH", ""),
    }

    # Add WandB credentials if present
    if "WANDB_HOST" in os.environ:
        env_vars["WANDB_HOST"] = os.environ["WANDB_HOST"]
    if "WANDB_KEY" in os.environ:
        env_vars["WANDB_KEY"] = os.environ["WANDB_KEY"]

    # Add SwanLab credentials if present
    if "SWANLAB_API_KEY" in os.environ:
        env_vars["SWANLAB_API_KEY"] = os.environ["SWANLAB_API_KEY"]
    if "SWANLAB_LOG_DIR" in os.environ:
        env_vars["SWANLAB_LOG_DIR"] = os.environ["SWANLAB_LOG_DIR"]
    if "SWANLAB_MODE" in os.environ:
        env_vars["SWANLAB_MODE"] = os.environ["SWANLAB_MODE"]

    # Add distributed training vars if present
    if "RANK" in os.environ:
        env_vars["RANK"] = os.environ["RANK"]
    if "WORLD_SIZE" in os.environ:
        env_vars["WORLD_SIZE"] = os.environ["WORLD_SIZE"]
    if "LOCAL_RANK" in os.environ:
        env_vars["LOCAL_RANK"] = os.environ["LOCAL_RANK"]
    if "LOCAL_WORLD_SIZE" in os.environ:
        env_vars["LOCAL_WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"]
    if "MASTER_ADDR" in os.environ:
        env_vars["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
    if "MASTER_PORT" in os.environ:
        env_vars["MASTER_PORT"] = os.environ["MASTER_PORT"]
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_vars["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

    # Add LLM judge configuration if present (used by reward scoring)
    if "LLM_JUDGE_URL" in os.environ:
        env_vars["LLM_JUDGE_URL"] = os.environ["LLM_JUDGE_URL"]
    if "LLM_JUDGE_MODEL" in os.environ:
        env_vars["LLM_JUDGE_MODEL"] = os.environ["LLM_JUDGE_MODEL"]

    return env_vars


class ClassWithInitArgs:
    """Stores a class with its initialization arguments (no Ray-specific logic)."""

    cls: type
    args: tuple
    kwargs: dict

    def __init__(self, cls: type, *args, **kwargs) -> None:
        self.cls = cls
        self.args = args
        self.kwargs = kwargs


class RayActorWrapper:
    """
    A wrapper for Ray actors to support elegant colocation.

    When multiple modules are colocated in a single Ray actor, this wrapper
    provides a clean interface to call methods on a specific module by
    automatically handling method name prefixing.

    Example:
        For example, if we have the following modules with their methods:
        - Alice: forward()
        - Bob: evaluate()

        If Alice and Bob are colocated, the actor has methods:
        - alice_forward()
        - bob_evaluate()

        Then with the created wrapper, we have:
        wrapper_alice.forward() -> calls actor.alice_forward()
        wrapper_bob.evaluate() -> calls actor.bob_evaluate()
    """

    def __init__(
        self, actor: ActorHandle, actor_class: type, role: NexRLRole, is_colocated: bool = True
    ):
        self._actor = actor
        self._actor_class = actor_class
        self._role = role
        self._role_prefix = role.value
        self._is_colocated = is_colocated

        # Rebind public methods from the actor to this wrapper
        self._rebind_public_methods()

    def _rebind_public_methods(self):
        """
        Rebind public methods from the actor to this wrapper.
        If the actor is colocated, bind methods with role prefix removed.
        If the actor is not colocated, bind all public methods as-is.
        """
        if self._is_colocated:
            # Bind methods with role prefix
            for attr_name in dir(self._actor_class):
                if attr_name.startswith(self._role_prefix + "_"):
                    role_method_name = attr_name.replace(self._role_prefix + "_", "")
                    actor_method = getattr(self._actor, attr_name)
                    setattr(self, role_method_name, actor_method)
        else:
            # Bind all public methods
            for attr_name in dir(self._actor_class):
                if not attr_name.startswith("_") and callable(
                    getattr(self._actor_class, attr_name, None)
                ):
                    actor_method = getattr(self._actor, attr_name)
                    setattr(self, attr_name, actor_method)


@dataclass
class RayResourceManager:
    """
    Ray resource manager for NexRL.

    Responsibilities:
    1. Register roles with their classes, configs, and instance counts
    2. Group roles into colocation groups
    3. Create Ray actors (colocated or standalone)
    4. Provide actor wrappers for each role
    """

    def __init__(self):
        # Store role registrations: {role: (cls, config, count, colocation_group)}
        self._role_registry: dict[NexRLRole, tuple[type, Any, int, str | None]] = {}

        # Store created actors for each colocation group
        self._colocation_group_actors: dict[str, list[ActorHandle]] = {}

        # Store actor wrappers for each role
        self._actor_wrappers: dict[NexRLRole, list[RayActorWrapper]] = {}

    def register_role(
        self,
        role: NexRLRole,
        cls: type,
        config: Any,
        count: int = 1,
        colocation_group: str | None = None,
    ):
        """
        Register a role with its class, configuration, and instance count.

        Args:
            role: The NexRL role
            cls: The class to instantiate
            config: Configuration object to pass to the class constructor
            count: Number of instances to create
            colocation_group: Name of colocation group. Roles with the same
                            group name will be colocated in the same actors.
                            None means standalone (one actor per instance).
        """
        if role in self._role_registry:
            logger.warning(f"Role {role} already registered, overwriting")

        self._role_registry[role] = (cls, config, count, colocation_group)

    def create_all_actors(self):
        """
        Create all Ray actors based on registered roles and colocation groups.

        Process:
        1. Group roles by colocation group
        2. For each group, create colocated actor class if multiple roles
        3. Create the specified number of actor instances
        4. Create wrappers for each role
        """
        # Group roles by colocation group
        colocation_groups: dict[str | None, list[NexRLRole]] = {}

        for role, (_, _, _, group) in self._role_registry.items():
            if group not in colocation_groups:
                colocation_groups[group] = []
            colocation_groups[group].append(role)

        logger.info(f"Creating actors for colocation groups: {list(colocation_groups.keys())}")

        # Process each colocation group
        for group_name, roles in colocation_groups.items():
            if group_name is None:
                # Standalone actors - create separately for each role
                for role in roles:
                    self._create_standalone_actors(role)
            else:
                # Colocated actors - create one actor type containing all roles
                self._create_colocated_actors(group_name, roles)

        logger.info("All actors created successfully")

    def _create_standalone_actors(self, role: NexRLRole):
        """Create standalone (non-colocated) actors for a role."""
        cls, config, count, _ = self._role_registry[role]

        # Apply ray.remote() to the original class
        ray_actor_cls = ray.remote(cls)

        # Get minimal required environment variables (avoids huge env in HPC)
        env_vars = _get_minimal_env_vars()

        ray_options = {
            "num_cpus": 1,
            "runtime_env": {"env_vars": env_vars},
            "max_concurrency": 100,
        }

        # Create actors (Ray only appears here)
        actors = []
        for _ in range(count):
            actor = ray_actor_cls.options(**ray_options).remote(config)
            actors.append(actor)

        # Create wrappers (use original class for inspection)
        self._actor_wrappers[role] = []
        for actor in actors:
            wrapper = RayActorWrapper(
                actor=actor,
                actor_class=cls,  # Use original class
                role=role,
                is_colocated=False,
            )
            self._actor_wrappers[role].append(wrapper)

        logger.info(f"Created {count} standalone actors for {role.value}")

    def _create_colocated_actors(self, group_name: str, roles: list[NexRLRole]):
        """Create colocated actors containing multiple roles."""
        # Determine the number of instances (all roles in group must have same count)
        counts = [self._role_registry[role][2] for role in roles]
        if len(set(counts)) != 1:
            raise ValueError(
                f"All roles in colocation group '{group_name}' must have the same count. "
                f"Got: {dict(zip([r.value for r in roles], counts))}"
            )
        count = counts[0]

        # Build class dict for colocation (pure Python classes)
        class_dict = {}
        for role in roles:
            cls, config, _, _ = self._role_registry[role]
            class_dict[role.value] = ClassWithInitArgs(cls, config)

        # Create colocated class (pure Python, no Ray yet)
        colocated_class = create_colocated_class(class_dict)

        # Now apply ray.remote() to the colocated class
        ray_actor_cls = ray.remote(colocated_class)

        # Get minimal required environment variables (avoids huge env in HPC)
        env_vars = _get_minimal_env_vars()

        ray_options = {
            "num_cpus": 1,
            "runtime_env": {"env_vars": env_vars},
            "max_concurrency": 10,
        }

        # Create actor instances (Ray only appears here)
        actors = []
        for i in range(count):
            actor = ray_actor_cls.options(**ray_options).remote()
            actors.append(actor)

            # Check readiness
            try:
                ray.get(actor.__ray_ready__.remote(), timeout=600)
            except Exception as e:
                logger.error(f"Colocated actor {i} in group '{group_name}' failed: {e}")
                raise

        self._colocation_group_actors[group_name] = actors

        # Create wrappers for each role
        for role in roles:
            self._actor_wrappers[role] = []
            for actor in actors:
                wrapper = RayActorWrapper(
                    actor=actor,
                    actor_class=colocated_class,  # Use the pure Python colocated class
                    role=role,
                    is_colocated=True,
                )
                self._actor_wrappers[role].append(wrapper)

        logger.info(
            f"Created {count} colocated actors for group '{group_name} with roles: {[r.value for r in roles]}'"
        )

    def get_actor_wrapper(self, role: NexRLRole) -> list[RayActorWrapper]:
        """Get actor wrappers for a specific role."""
        if role not in self._actor_wrappers:
            raise ValueError(f"No actor wrappers found for role {role}")
        return self._actor_wrappers[role]


def create_colocated_class(class_dict: dict[str, ClassWithInitArgs]) -> type:
    """
    Create a colocated class that contains multiple worker types.

    This is pure Python class composition - no Ray-specific logic.
    Ray wrapping happens later in the caller.

    Args:
        class_dict: Dictionary mapping role names to ClassWithInitArgs

    Returns:
        A pure Python class (ColocatedWorker) that contains all the roles
    """
    cls_dict = {}
    init_args_dict = {}

    # Import the common base class
    from .base_module import NexRLModule

    # Store original classes and their init args
    for key, cls_with_args in class_dict.items():
        cls_dict[key] = cls_with_args.cls
        init_args_dict[key] = {"args": cls_with_args.args, "kwargs": cls_with_args.kwargs}

    # Create the colocated worker class (pure Python)
    class ColocatedWorker(NexRLModule):
        def __init__(self):
            logger.info(
                f"ColocatedWorker.__init__ started, creating workers for keys: {list(cls_dict.keys())}"
            )
            super().__init__()
            self.worker_dict = {}

            for key, original_cls in cls_dict.items():
                # Create worker instance
                args = cast(tuple[Any, ...], init_args_dict[key].get("args", ()))
                kwargs = cast(dict[str, Any], init_args_dict[key].get("kwargs", {}))
                self.worker_dict[key] = original_cls(*args, **kwargs)
                logger.info(f"ColocatedWorker.__init__ completed for key: {key}")

    # Bind methods with prefixes (e.g., 'data_loader_get_batch' for DataLoader.get_batch)
    for key, original_cls in cls_dict.items():
        for method_name in dir(original_cls):
            if not method_name.startswith("_") and callable(getattr(original_cls, method_name)):

                def create_delegated_method(worker_key, method):
                    def delegated_method(self, *args, **kwargs):
                        return getattr(self.worker_dict[worker_key], method)(*args, **kwargs)

                    return delegated_method

                # Add method with prefix: e.g., 'data_loader_get_batch'
                prefixed_method_name = f"{key}_{method_name}"
                setattr(
                    ColocatedWorker, prefixed_method_name, create_delegated_method(key, method_name)
                )

    # Return the pure Python class (no ray.remote here)
    return ColocatedWorker
