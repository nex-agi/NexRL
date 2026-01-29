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
NexRL Controller - Main controller for the RL training framework
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import ray
from omegaconf import DictConfig

from .activity_tracker import ActivityTracker, ActivityTrackerProxy
from .data_loader import BaseDataLoader, TorchDataLoader
from .executor import execute
from .mock.mock_data_loader import MockDataLoader
from .mock.mock_rollout_worker import MockRolloutWorker
from .nexrl_types import ModelTag, NexRLRole
from .ray_resource_manager import RayResourceManager, _get_minimal_env_vars
from .rollout_worker import (
    AgentRolloutWorker,
    BaseRolloutWorker,
    DefaultNexAURolloutWorker,
    PigLatinRolloutWorker,
    SimpleRolloutWorker,
)
from .trainer import BaseTrainer
from .trainer.remote_api_cross_entropy_trainer import RemoteApiCrossEntropyTrainer
from .trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from .trainer.self_hosted_grpo_trainer import SelfHostedGrpoTrainer
from .trainer.self_hosted_opd_trainer import SelfHostedOpdTrainer
from .trajectory_pool import TrajectoryPool
from .utils.config_utils import (
    get_actor_train_service_config,
    insert_config,
    use_tinker,
    use_weaver,
)
from .utils.init_utils import create_train_service_client
from .utils.logging_utils import set_logging_basic_config
from .validator import Validator
from .weight_sync.weight_sync_controller import WeightSyncController

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """
    Set the seed of each random module.
    `torch.manual_seed` will set seed on all devices.
    Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NexRLController:
    """
    NexRL controller is the main process of the RL training framework.
    Responsible for starting the entire experiment, initializing various modules,
    and synchronizing configuration information.

    In NexRL, all components are managed through Ray, i.e., each component is a Ray actor.
    When created, components will get references to other modules they need to interact with,
    used to call functions of other modules.

    Interaction relationship: DataLoader <- RolloutWorker -> TrajectoryPool <- Trainer
    """

    # Public interfaces - accessible to external components
    trajectory_pool: TrajectoryPool
    weight_sync_controller: WeightSyncController
    dataloader: BaseDataLoader
    validate_dataloader: BaseDataLoader
    validator: Validator
    trainer: BaseTrainer
    rollout_workers: list[BaseRolloutWorker]
    activity_tracker: ActivityTracker

    def __init__(self, config: DictConfig):
        """
        Initialize all members of NexRL.
        Calls all other init() functions of all other modules in `_init_modules()`.

        Args:
            config: Configuration file
        """
        # Set up logging configuration for Ray actors
        set_logging_basic_config()

        seed_everything(config.data.seed)

        self._config = config

        # Initialize all modules
        self._launch_mode: str = config.launch_mode
        logger.info(f"Controller initializing with launch mode: {self._launch_mode}")
        os.environ["NEXRL_LAUNCH_MODE"] = config.launch_mode

        # Initialize activity tracker and error reporter
        self._init_activity_tracker()

        self.use_tinker = use_tinker(config)
        self.use_weaver = use_weaver(config)
        if self.use_tinker:
            self._init_tinker_service()
        if self.use_weaver:
            self._init_weaver_service()

        # Initialize modules using unified approach
        self._init_modules()

        logger.info("NexRL Controller initialized successfully")

    def run(self) -> None:
        """Start the training process"""
        logger.info("Starting NexRL training process...")

        execute(self.trainer.initialize_workers)

        # Load initial checkpoint
        self._load_initial_checkpoint()

        if self._config.validate.validate_before_train:
            # identifier serves as model_tag for weight sync coordination
            self._start_validate(self._config.service.inference_service.identifier)

        # Start all components
        execute(self.trainer.run)
        for worker in self.rollout_workers:
            execute(worker.run)

        # Start progress monitor (integrated in activity tracker)
        execute(self.activity_tracker.start_progress_monitor)

        if self._config.validate.validate_before_train:
            # identifier serves as model_tag for weight sync coordination
            self._end_validate(self._config.service.inference_service.identifier)

        # Main monitoring loop
        runtime_config = self._config.get("runtime_monitor", {})
        health_config = runtime_config.get("health_check", {})
        exception_config = runtime_config.get("exception_handling", {})

        health_check_counter = 0
        exception_check_counter = 0
        health_check_interval = health_config.get("check_interval", 10.0)
        exception_check_interval = exception_config.get("check_interval", 1.0)

        while True:
            # Check if weight sync is waiting for validation
            # identifier serves as model_tag for weight sync coordination
            identifier = self._config.service.inference_service.identifier
            if execute(self.weight_sync_controller.is_waiting_for_validation):
                self._run_validate(identifier)

            if self._check_finish():
                self.activity_tracker.experiment_logger_post(
                    backend="feishu",
                    content=f'"Experiment: {self._config.project_name}/{self._config.experiment_name}"'
                    f"training completed, wandb: {self.activity_tracker.experiment_logger.wandb_url} ",
                    title="Complete!",
                )
                break

            # Check module liveness at configured interval
            if health_check_counter % health_check_interval == 0:
                is_all_modules_alive = self._check_module_liveness()
                if not is_all_modules_alive:
                    logger.error("Module liveness check detected dead modules. Terminating system.")
                    self.activity_tracker.experiment_logger_post(
                        backend="feishu",
                        content=f'"Experiment: {self._config.project_name}/{self._config.experiment_name}"'
                        f"training failed, module liveness check detected dead modules. wandb: {self.activity_tracker.experiment_logger.wandb_url} ",
                        title="Error!",
                    )
                    break
                logger.info(f"Module liveness check passed at {time.time()}")

            # Check system health at configured interval
            if exception_check_counter >= exception_check_interval:
                is_system_healthy = self._check_module_exceptions()
                if not is_system_healthy:
                    logger.error(
                        "System health check detected critical issues. Terminating system."
                    )
                    self.activity_tracker.experiment_logger_post(
                        backend="feishu",
                        content=f'"Experiment: {self._config.project_name}/{self._config.experiment_name}"'
                        f"training failed, system health check detected critical issues. wandb: {self.activity_tracker.experiment_logger.wandb_url} ",
                        title="Error!",
                    )
                    break
                exception_check_counter = 0

            time.sleep(1)
            health_check_counter += 1
            exception_check_counter += 1

        self._stop()
        logger.info("NexRL training process finished")

    def _stop(self):
        """Stop all components gracefully"""
        logger.info("Stopping NexRL training process...")

        # Stop progress monitor first
        execute(self.activity_tracker.stop_progress_monitor)

        # Signal all workers to stop gracefully
        execute(self.trainer.stop)
        for worker in self.rollout_workers:
            execute(worker.stop)

        # Wait for all activities to complete with timeout
        logger.info("Waiting for all activities to complete...")
        logger.info(f"Current status: {execute(self.activity_tracker.get_running_status_summary)}")

        if execute(self.activity_tracker.wait_quiescent, timeout=30.0):
            logger.info("All activities completed gracefully")
        else:
            logger.warning("Timeout reached - some activities may not have completed")
            logger.warning(
                f"Remaining activities: {execute(self.activity_tracker.get_running_status_summary)}"
            )
            logger.warning("Workers should eventually stop when they check their stop_event")

        logger.info("NexRL training process stopped")

    # ---- Load checkpoint functions ----
    def _load_initial_checkpoint(self):
        if self.use_tinker or self.use_weaver:
            return

        # Get the actor (main) train service config
        actor_train_service = get_actor_train_service_config(self._config)

        # Create train service client for the actor (main) train service
        self._train_service_client = create_train_service_client(
            actor_train_service.backend,
            actor_train_service.url,
            actor_train_service.get("identifier", None),
            tinker_service_holder=getattr(self, "_tinker_service_holder", None),
            config=actor_train_service.get("config", {}),
        )

        if self._config.resume.mode != "disable":
            self._load_resume_checkpoint()

        self._train_service_client.save_checkpoint(
            self._config.trainer.sync_weight_path,
            global_step=0,
            saved_fully_shared_ckpt=False,
            save_weight_only=True,
            remove_previous_ckpt=False,
        )
        execute(
            self.weight_sync_controller.sync_weight_to_rollout_service,
            actor_train_service.identifier,
        )

    def _load_resume_checkpoint(self):
        """Load checkpoint based on resume configuration"""
        if self._config.resume.mode == "disable":
            logger.info("Resume mode is disabled, training from scratch")
            return

        if self.use_tinker or self.use_weaver:
            raise NotImplementedError("Tinker/Weaver backend does not support checkpoint loading")

        checkpoint_folder = self._config.trainer.checkpoint_path
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

        global_step_folder = None

        # Determine checkpoint path based on resume mode
        if self._config.resume.mode == "auto":
            # Find the latest checkpoint automatically
            global_step_folder = self._find_latest_checkpoint(checkpoint_folder)
            if global_step_folder is None:
                logger.info("No checkpoint found for auto resume, training from scratch")
                return
        elif self._config.resume.mode == "from_path":
            # Use specific checkpoint path from config
            resume_path = self._config.resume.resume_path
            if not resume_path:
                raise ValueError("resume_path must be specified when resume_mode is 'from_path'")

            if not os.path.isabs(resume_path):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, resume_path)
            else:
                global_step_folder = resume_path

            if not os.path.exists(global_step_folder):
                raise FileNotFoundError(
                    f"Resume checkpoint path does not exist: {global_step_folder}"
                )
        else:
            raise ValueError(f"Unsupported resume mode: {self._config.resume.mode}")

        logger.info(f"Loading checkpoint from: {global_step_folder}")

        # Extract global step from folder name
        if "global_step_" in global_step_folder:
            global_step = int(global_step_folder.split("global_step_")[-1].split("/")[0])
            logger.info(f"Resuming from global step: {global_step}")
        else:
            logger.warning(
                "Could not extract global step from checkpoint path, starting from step 0"
            )
            global_step = 0

        # Load checkpoint using train service client
        with self._train_service_client.actor_context():
            result = self._train_service_client.load_checkpoint(
                path=global_step_folder,
                del_local_after_load=False,
                load_weight_only=False,
            )
            logger.info(f"Checkpoint loaded successfully: {result}")

        # Update trainer's global step
        execute(self.trainer.set_train_step, global_step)
        logger.info(f"Training will resume from step {global_step}")

        # Resume dataloader by skipping already-consumed batches
        resume_dataloader = self._config.resume.get("resume_dataloader", True)
        if resume_dataloader and global_step > 0:
            logger.info(
                f"Resuming dataloader: skipping first {global_step} batches "
                f"(corresponding to {global_step} training steps)"
            )
            try:
                execute(self.dataloader.skip_batches, global_step)
                logger.info(f"Successfully resumed dataloader from step {global_step}")
            except Exception as e:
                logger.error(
                    f"Failed to skip batches in dataloader: {e}. "
                    f"Training will continue but may process duplicate data.",
                    exc_info=True,
                )
        elif not resume_dataloader:
            logger.info(
                "Dataloader resume is disabled (resume.resume_dataloader=false). "
                "Dataloader will start from the beginning - data may be processed again."
            )
        else:
            logger.info("No batches to skip (global_step=0)")

    def _find_latest_checkpoint(self, checkpoint_folder: str) -> str | None:
        """Find the latest checkpoint in the given folder"""
        if not os.path.exists(checkpoint_folder):
            logger.warning(f"Checkpoint folder does not exist: {checkpoint_folder}")
            return None

        # Find all global_step_* directories
        checkpoint_dirs = []
        try:
            for item in os.listdir(checkpoint_folder):
                item_path = os.path.join(checkpoint_folder, item)
                if os.path.isdir(item_path) and item.startswith("global_step_"):
                    try:
                        step_num = int(item.split("global_step_")[-1])
                        checkpoint_dirs.append((step_num, item_path))
                    except ValueError:
                        logger.warning(f"Invalid checkpoint directory name: {item}")
                        continue
        except Exception as e:
            logger.error(f"Error listing checkpoint directory: {e}")
            return None

        if not checkpoint_dirs:
            return None

        # Sort by step number and return the latest
        checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
        latest_checkpoint = checkpoint_dirs[0][1]
        logger.info(f"Found latest checkpoint: {latest_checkpoint} (step {checkpoint_dirs[0][0]})")
        return latest_checkpoint

    # ---- Check functions ----
    def _check_finish(self) -> bool:
        """
        Check if the training process should be stopped.

        Returns:
            True if training should be stopped, False otherwise
        """
        # Hard stop: reached or exceeded max training steps
        steps_reached = (
            execute(self.trainer.get_train_step) >= self._config.trainer.total_train_steps
        )
        if steps_reached:
            logger.info("Finish check: steps_reached=True -> stopping")
            return True

        # Quiescence stop: only when all sources/sinks are drained and no work is in-flight
        rollout_pool_empty = execute(self.trajectory_pool.is_empty)
        dataloader_done = execute(self.dataloader.is_finished)
        tracker_idle = execute(self.activity_tracker.is_quiescent)

        logger.debug(
            f"Finish check: rollout_pool_empty={rollout_pool_empty}, "
            f"dataloader_done={dataloader_done}, tracker_idle={tracker_idle}"
        )

        return rollout_pool_empty and dataloader_done and tracker_idle

    def _check_module_liveness(self) -> bool:
        """
        Check module liveness and return whether all modules are alive.

        Returns:
            bool: True if all modules are alive, False if any modules are dead
        """
        runtime_config = self._config.get("runtime_monitor", {})
        health_config = runtime_config.get("health_check", {})
        # Skip if health checking is disabled - assume all modules are alive
        if not health_config.get("enabled", True):
            return True
        health_timeout = health_config.get("timeout", 20.0)

        return execute(self.activity_tracker.check_module_liveness, health_timeout)

    def _check_module_exceptions(self) -> bool:
        """
        Check system health based on exceptions and return whether the system is healthy.

        Returns:
            bool: True if system is healthy, False if system has critical issues
        """
        runtime_config = self._config.get("runtime_monitor", {})
        exception_config = runtime_config.get("exception_handling", {})

        # Skip if exception handling is disabled - assume system is healthy
        if not exception_config.get("enabled", True):
            return True

        policy = exception_config.get("policy", "stop_on_error")

        # Get recent errors from the error reporter
        try:
            health_status = execute(self.activity_tracker.get_error_health_status)
            error_count = health_status.get("error_level_count", 0)

            # Apply policy to determine system health
            if policy == "stop_on_error" and error_count > 0:
                logger.error(
                    f"System health check: policy '{policy}' detected {error_count} errors. System is unhealthy."
                )
                return False
            elif policy == "stop_on_critical" and error_count > 0:
                # For now, treat any error as critical - can be extended later
                # For example, if only one of hundreds of agent workers died and the system is still healthy, we should continue the system.
                logger.error(
                    f"System health check: policy '{policy}' detected {error_count} critical errors. System is unhealthy."
                )
                return False
            elif policy == "continue":
                # Always consider system healthy regardless of exceptions
                if error_count > 0:
                    logger.warning(
                        f"Detected {error_count} errors, but policy is 'continue'. System remains healthy."
                    )
                return True

            return True

        except Exception as e:
            logger.error(f"Error while checking system health: {e}")
            return False  # If we can't check health, assume unhealthy

    def _start_validate(self, model_tag: ModelTag):
        """Start validation"""
        current_version = execute(
            self.weight_sync_controller.get_rollout_model_version,
            model_tag,
        )
        logger.info(f"Version {current_version}: Starting validation...")

        # Switch workers to validation mode
        for worker in self.rollout_workers:
            execute(worker.begin_validate)

    def _end_validate(self, model_tag: ModelTag):
        """End validation"""
        # Wait for completion
        while not execute(self.validator.is_complete):
            time.sleep(1.0)
        logger.info("Validator is complete")

        execute(self.validator.compute_and_log_metrics)

        # Switch back to training mode
        for worker in self.rollout_workers:
            execute(worker.end_validate)

        # Tell weight sync controller validation is done - unlock now
        execute(self.weight_sync_controller.end_validate, model_tag)
        execute(self.validate_dataloader.reset)

        logger.info("Validation complete")

    # ---- Validation functions ----
    def _run_validate(self, model_tag: ModelTag):
        """Run validation right after a weight sync"""
        self._start_validate(model_tag)
        self._end_validate(model_tag)

    # ---- Module initialization functions ----
    def _load_custom_module(
        self, module_path: str, class_name: str, module_type: str = "module"
    ) -> type:
        """
        Dynamically load a custom module class from a recipe directory.

        Args:
            module_path: Path to the Python module (e.g., "agent_workspace/my_custom.py")
            class_name: Name of the class to load
            module_type: Type of module being loaded (for logging/error messages)

        Returns:
            The loaded class
        """
        import importlib.util
        import sys

        # Resolve the full path relative to config file
        config_file_path = getattr(self._config, "_config_file_path", None)
        if not config_file_path:
            raise ValueError(
                "Config file path not available for resolving custom module path. "
                "Ensure the config has _config_file_path attribute set."
            )

        from .utils.path_utils import resolve_path_from_config

        full_path_str = resolve_path_from_config(module_path, config_file_path)

        if full_path_str is None:
            raise ValueError(f"Could not resolve module path: {module_path}")

        full_path = Path(full_path_str)

        if not full_path.exists():
            raise FileNotFoundError(f"Custom {module_type} module not found: {full_path}")

        # Create a unique module name
        module_name = f"nexrl.custom_{module_type}.{full_path.stem}"

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {full_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Get the class
        if not hasattr(module, class_name):
            raise AttributeError(f"Module {module_path} does not have class {class_name}")

        loaded_class = getattr(module, class_name)
        logger.info(f"Loaded custom {module_type}: {class_name} from {module_path}")

        return loaded_class

    def _get_module_class_for_role(
        self, role: NexRLRole, config_type: str, config: DictConfig | None = None
    ) -> type:
        """
        Get the class for a specific module type and config type.

        For ROLLOUT_WORKER and TRAINER roles, supports custom class loading via config.

        Args:
            role: The NexRL role
            config_type: The type string from config (e.g., "nexau", "mock")
            config: Full config object (optional, needed for custom class loading)

        Returns:
            The module class to instantiate
        """
        # Check for custom rollout worker before registry lookup
        if role == NexRLRole.ROLLOUT_WORKER and config is not None:
            if "custom_rollout_worker_module_path" in config:
                custom_path = config.custom_rollout_worker_module_path
                custom_class_name = config.get(
                    "custom_rollout_worker_class_name", "CustomRolloutWorker"
                )
                return self._load_custom_module(custom_path, custom_class_name, "rollout_worker")

        # Check for custom trainer before registry lookup
        if role == NexRLRole.TRAINER and config is not None:
            if "custom_trainer_module_path" in config:
                custom_path = config.custom_trainer_module_path
                custom_class_name = config.get("custom_trainer_class_name", "CustomTrainer")
                return self._load_custom_module(custom_path, custom_class_name, "trainer")

        # Standard registry lookup
        registry: dict[NexRLRole, dict[str, type]] = {
            NexRLRole.DATA_LOADER: {
                "mock": MockDataLoader,
                "torch": TorchDataLoader,
            },
            NexRLRole.ROLLOUT_WORKER: {
                "mock": MockRolloutWorker,
                "simple": SimpleRolloutWorker,
                "agent": AgentRolloutWorker,
                "pig_latin": PigLatinRolloutWorker,
                "nexau": DefaultNexAURolloutWorker,  # Default NexAU worker with self-contained implementation
                # Note: Custom workers use type="nexau" + custom_rollout_worker_module_path
            },
            NexRLRole.TRAINER: {
                "self_hosted_grpo": SelfHostedGrpoTrainer,
                "self_hosted_opd": SelfHostedOpdTrainer,
                "remote_api_grpo": RemoteApiGrpoTrainer,
                "remote_api_cross_entropy": RemoteApiCrossEntropyTrainer,
            },
            NexRLRole.TRAJECTORY_POOL: {
                "default": TrajectoryPool,
            },
            NexRLRole.WEIGHT_SYNC_CONTROLLER: {
                "default": WeightSyncController,
            },
            NexRLRole.VALIDATE_DATALOADER: {
                "mock": MockDataLoader,
                "torch": TorchDataLoader,
            },
            NexRLRole.VALIDATOR: {
                "default": Validator,
            },
        }

        if role not in registry:
            raise ValueError(f"Unsupported module type: {role}")

        if config_type not in registry[role]:
            available_types = list(registry[role].keys())
            raise ValueError(
                f"Unsupported {role} type: {config_type}. Available types: {available_types}"
            )

        return registry[role][config_type]

    def _init_module_for_role(
        self, role: NexRLRole, config: DictConfig, num_workers: int, *args, **kwargs
    ) -> list[Any]:
        """
        Generic module instantiation supporting local, Ray, and hybrid modes

        Args:
            role: The NexRL role
            config: The configuration for the module
            num_workers: The number of workers to initialize
            *args, **kwargs: Arguments to pass to the module constructor
        Returns:
            list of module instances
        """
        module_type = config.get("type", "default")
        module_class = self._get_module_class_for_role(role, module_type, config)
        ret = []

        if self._launch_mode == "local":

            def init_worker(_):
                return module_class(config, *args, **kwargs)

            # Initialize workers in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                ret = list(executor.map(init_worker, range(num_workers)))
        elif self._launch_mode == "ray":
            assert self._ray_resource_manager is not None, "Resource manager not initialized"
            # Get the actor wrappers for this role (actors should already be created)
            ret = self._ray_resource_manager.get_actor_wrapper(role)
        else:
            raise ValueError(f"Unsupported launch mode: {self._launch_mode}")

        for i, module in enumerate(ret):
            if hasattr(module, "_actor"):
                try:
                    ray.get(
                        module._actor.__ray_ready__.remote(),  # pylint: disable=protected-access
                        timeout=60,
                    )
                    logger.debug(f"Module {i} with role {role.value} is ready")
                except Exception as e:
                    logger.error(
                        f"Module {i} with role {role.value} failed readiness check after 60s timeout: {e}"
                    )
                    raise
            proxy = ActivityTrackerProxy(self.activity_tracker)
            execute(module.set_activity_tracker, proxy)
            module_name = role.value + f"-{i}"
            execute(module.set_module_name, module_name)
            execute(
                self.activity_tracker.register_module,
                module_name=module_name,
                module_ref=ret,
                is_rollout_worker=(role == NexRLRole.ROLLOUT_WORKER),
            )
        logger.info(f"Module {role.value} initialization completed successfully")

        return ret

    def _init_rollout_worker_inference_clients(self):
        """Initialize inference service clients for all rollout workers"""
        # Get backend from inference service
        backend = self._config.service.inference_service.backend
        for worker in self.rollout_workers:
            if backend == "tinker":
                execute(worker.init_inference_service_client, self._tinker_service_holder)
            elif backend == "weaver":
                execute(worker.init_inference_service_client, self._weaver_service_holder)
            else:
                execute(worker.init_inference_service_client)

        logger.info(
            f"Initialized inference service clients for {len(self.rollout_workers)} rollout workers"
        )

    def _init_modules(self):
        """Initialize all modules using unified approach (supports local, Ray, and hybrid modes)"""
        launch_mode = self._config.launch_mode
        logger.info(f"Initializing modules in {launch_mode} mode...")

        # Define module configurations - single source of truth using roles as keys
        module_configs = {
            NexRLRole.TRAJECTORY_POOL: self._config.trajectory_pool,
            NexRLRole.WEIGHT_SYNC_CONTROLLER: self._config.weight,
            NexRLRole.DATA_LOADER: self._config.data,
            NexRLRole.TRAINER: self._config.trainer,
            NexRLRole.ROLLOUT_WORKER: self._config.rollout_worker,
            NexRLRole.VALIDATE_DATALOADER: self._config.validate.data,
            NexRLRole.VALIDATOR: self._config.validate.eval,
        }

        # TODO: use reference in config instead to insert_config  # pylint: disable=fixme

        # Add inference_service config to weight and rollout_worker config
        insert_config(
            self._config.weight, "inference_service", self._config.service.inference_service
        )
        insert_config(
            self._config.rollout_worker, "inference_service", self._config.service.inference_service
        )
        config_file_path = getattr(self._config, "_config_file_path", None)
        if not config_file_path:
            raise ValueError(
                "Config file path not available. Ensure the config has _config_file_path attribute set."
            )
        insert_config(self._config.rollout_worker, "_config_file_path", config_file_path)

        if self._config.trainer.get("algorithm", None) is not None:
            # Add inference_service and train_service config to trainer's algorithm config
            insert_config(
                self._config.trainer.algorithm,
                "inference_service",
                self._config.service.inference_service,
            )
            insert_config(
                self._config.trainer.algorithm, "train_service", self._config.service.train_service
            )

        insert_config(self._config.trainer, "_config_file_path", config_file_path)

        # Add train_service config directly to trainer
        insert_config(self._config.trainer, "train_service", self._config.service.train_service)

        if launch_mode == "ray":
            self._init_ray_resources(module_configs)

        # Explicit module initialization for better readability
        self.dataloader = self._init_module_for_role(
            NexRLRole.DATA_LOADER,
            module_configs[NexRLRole.DATA_LOADER],
            num_workers=1,
            is_validate=False,
        )[0]
        self.trajectory_pool = self._init_module_for_role(
            NexRLRole.TRAJECTORY_POOL,
            module_configs[NexRLRole.TRAJECTORY_POOL],
            num_workers=1,
        )[0]
        self.weight_sync_controller = self._init_module_for_role(
            NexRLRole.WEIGHT_SYNC_CONTROLLER,
            module_configs[NexRLRole.WEIGHT_SYNC_CONTROLLER],
            num_workers=1,
        )[0]
        self.trainer = self._init_module_for_role(
            NexRLRole.TRAINER,
            module_configs[NexRLRole.TRAINER],
            num_workers=1,
        )[0]

        self.rollout_workers = self._init_module_for_role(
            NexRLRole.ROLLOUT_WORKER,
            module_configs[NexRLRole.ROLLOUT_WORKER],
            num_workers=self._config.rollout_worker.num_workers,
        )
        self.validate_dataloader = self._init_module_for_role(
            NexRLRole.VALIDATE_DATALOADER,
            module_configs[NexRLRole.VALIDATE_DATALOADER],
            num_workers=1,
            is_validate=True,
        )[0]
        self.validator = self._init_module_for_role(
            NexRLRole.VALIDATOR,
            module_configs[NexRLRole.VALIDATOR],
            num_workers=1,
        )[0]

        # Initialize inference service clients
        self._init_rollout_worker_inference_clients()

        # Set module references
        self._set_module_references()

        logger.info("Module initialization completed successfully")

    def _set_module_references(self):
        """Set module references consistently across local and Ray modes"""

        execute(
            self.dataloader.set_module_references,
            weight_sync_controller=self.weight_sync_controller,
        )
        for worker in self.rollout_workers:
            execute(
                worker.set_module_references,
                trajectory_pool=self.trajectory_pool,
                dataloader=self.dataloader,
                weight_sync_controller=self.weight_sync_controller,
                validate_dataloader=self.validate_dataloader,
                validator=self.validator,
            )
        execute(
            self.trajectory_pool.set_module_references,
            dataloader=self.dataloader,
            weight_sync_controller=self.weight_sync_controller,
        )
        execute(
            self.trainer.set_module_references,
            trajectory_pool=self.trajectory_pool,
            weight_sync_controller=self.weight_sync_controller,
        )
        execute(
            self.weight_sync_controller.set_module_references,
            dataloader=self.dataloader,
            trajectory_pool=self.trajectory_pool,
        )
        execute(
            self.validator.set_module_references,
            validate_dataloader=self.validate_dataloader,
        )
        if self.use_tinker:
            execute(
                self.weight_sync_controller.set_tinker_service_holder, self._tinker_service_holder
            )
            # Set service holder for RemoteApiTrainer
            if hasattr(self.trainer, "set_service_holder"):
                execute(self.trainer.set_service_holder, self._tinker_service_holder)
        if self.use_weaver:
            execute(
                self.weight_sync_controller.set_weaver_service_holder, self._weaver_service_holder
            )
            # Set service holder for RemoteApiTrainer
            if hasattr(self.trainer, "set_service_holder"):
                execute(self.trainer.set_service_holder, self._weaver_service_holder)

    def _init_ray_resources(self, module_configs):
        """Initialize Ray resources and register all roles"""

        self._ray_resource_manager = RayResourceManager()

        # Register all roles with their classes and colocation groups
        # Roles in the same colocation group will share the same Ray actor

        # Main coordination group - lightweight, can be colocated
        self._ray_resource_manager.register_role(
            role=NexRLRole.DATA_LOADER,
            cls=self._get_module_class_for_role(
                NexRLRole.DATA_LOADER, module_configs[NexRLRole.DATA_LOADER].type
            ),
            config=module_configs[NexRLRole.DATA_LOADER],
            count=1,
            colocation_group="main",
        )

        self._ray_resource_manager.register_role(
            role=NexRLRole.TRAJECTORY_POOL,
            cls=self._get_module_class_for_role(
                NexRLRole.TRAJECTORY_POOL, module_configs[NexRLRole.TRAJECTORY_POOL].type
            ),
            config=module_configs[NexRLRole.TRAJECTORY_POOL],
            count=1,
            colocation_group="main",
        )

        self._ray_resource_manager.register_role(
            role=NexRLRole.WEIGHT_SYNC_CONTROLLER,
            cls=self._get_module_class_for_role(
                NexRLRole.WEIGHT_SYNC_CONTROLLER,
                module_configs[NexRLRole.WEIGHT_SYNC_CONTROLLER].type,
            ),
            config=module_configs[NexRLRole.WEIGHT_SYNC_CONTROLLER],
            count=1,
            colocation_group="main",
        )

        # Validation group - can be colocated
        self._ray_resource_manager.register_role(
            role=NexRLRole.VALIDATE_DATALOADER,
            cls=self._get_module_class_for_role(
                NexRLRole.VALIDATE_DATALOADER, module_configs[NexRLRole.VALIDATE_DATALOADER].type
            ),
            config=module_configs[NexRLRole.VALIDATE_DATALOADER],
            count=1,
            colocation_group="validation",
        )

        self._ray_resource_manager.register_role(
            role=NexRLRole.VALIDATOR,
            cls=self._get_module_class_for_role(
                NexRLRole.VALIDATOR, module_configs[NexRLRole.VALIDATOR].type
            ),
            config=module_configs[NexRLRole.VALIDATOR],
            count=1,
            colocation_group="validation",
        )

        # Standalone actors - each needs its own process
        # Rollout workers - multiple instances, each standalone
        rollout_worker_count = self._config.rollout_worker.num_workers
        self._ray_resource_manager.register_role(
            role=NexRLRole.ROLLOUT_WORKER,
            cls=self._get_module_class_for_role(
                NexRLRole.ROLLOUT_WORKER,
                module_configs[NexRLRole.ROLLOUT_WORKER].type,
                module_configs[
                    NexRLRole.ROLLOUT_WORKER
                ],  # Pass full config for custom worker support
            ),
            config=module_configs[NexRLRole.ROLLOUT_WORKER],
            count=rollout_worker_count,
            colocation_group=None,  # None = standalone
        )

        # Trainer - needs its own dedicated process
        self._ray_resource_manager.register_role(
            role=NexRLRole.TRAINER,
            cls=self._get_module_class_for_role(
                NexRLRole.TRAINER, module_configs[NexRLRole.TRAINER].type
            ),
            config=module_configs[NexRLRole.TRAINER],
            count=1,
            colocation_group=None,  # None = standalone
        )

        # Create all actors based on registrations
        self._ray_resource_manager.create_all_actors()

        logger.info("Ray resources and actors initialized successfully")

    def _init_activity_tracker(self):
        """Initialize activity tracker and error reporter, then start all workers"""
        # Initialize error reporting system for both local and Ray modes

        # Initialize activity tracker (standalone Ray actor for Ray mode, local for local mode)
        if self._launch_mode == "local":
            self.activity_tracker = ActivityTracker(max_errors=1000, config=self._config)
        elif self._launch_mode == "ray":
            # Make ActivityTracker multi-threaded to handle concurrent requests from multiple modules
            # Get minimal required environment variables (avoids huge env in HPC)
            env_vars = _get_minimal_env_vars()
            ActivityTrackerActor = ray.remote(ActivityTracker).options(
                max_concurrency=100,
                runtime_env={"env_vars": env_vars},
            )
            self.activity_tracker = ActivityTrackerActor.remote(config=self._config, max_errors=1000)  # type: ignore  # mypy does not recognize this
            try:
                ray.get(self.activity_tracker.__ray_ready__.remote(), timeout=180)  # type: ignore[attr-defined]
                logger.info(f"Activity tracker is ready")
            except Exception as e:
                logger.error(f"Activity tracker failed readiness check after 60s timeout: {e}")
                raise
        else:
            raise ValueError(f"Unsupported launch mode: {self._launch_mode}")

    def _init_tinker_service(self):
        """Initialize Tinker service holder for Tinker backend"""
        from .tinker import TinkerServiceHolder

        tinker_config = self._config.service.get("tinker_service", {})
        # Get base_url from environment, fallback to config
        base_url = os.environ.get("TINKER_BASE_URL", None)
        if base_url is None:
            base_url = tinker_config.get("base_url", None)

        # Get tinker_api_key from environment, fallback to config
        tinker_api_key = os.environ.get("TINKER_API_KEY", None)
        if tinker_api_key is None:
            tinker_api_key = tinker_config.get("api_key", None)

        # Report error if tinker_api_key is not provided in either environment or config
        if not tinker_api_key:
            error_msg = (
                "TINKER_API_KEY is required but not found in environment variables or config"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        os.environ["TINKER_API_KEY"] = tinker_api_key

        base_model = self._config.service.inference_service.model
        tokenizer_path = self._config.service.inference_service.get("tokenizer", None)
        lora_rank = tinker_config.get("lora_rank", 32)

        # Create TinkerServiceHolder (can be Ray actor or local)
        if self._launch_mode == "local":
            self._tinker_service_holder = TinkerServiceHolder(
                base_model=base_model,
                lora_rank=lora_rank,
                base_url=base_url,
                tokenizer_path=tokenizer_path,
            )
        elif self._launch_mode == "ray":
            # Create as Ray actor for shared access
            env_vars = _get_minimal_env_vars()
            env_vars["TINKER_API_KEY"] = tinker_api_key
            TinkerServiceActor = ray.remote(TinkerServiceHolder).options(
                runtime_env={"env_vars": env_vars},
            )
            self._tinker_service_holder = TinkerServiceActor.remote(
                base_model=base_model,
                lora_rank=lora_rank,
                base_url=base_url,
                tokenizer_path=tokenizer_path,
            )
            # Wait for initialization
            try:
                ray.get(self._tinker_service_holder.__ray_ready__.remote(), timeout=180)  # type: ignore
                logger.info("TinkerServiceHolder Ray actor is ready")
            except Exception as e:
                logger.error(f"TinkerServiceHolder failed initialization: {e}")
                raise

        logger.info("TinkerServiceHolder initialized successfully")

    def _init_weaver_service(self):
        """Initialize Weaver service holder for Weaver backend"""
        from .weaver import WeaverServiceHolder

        weaver_config = self._config.service.get("weaver_service", {})

        # Get base_url from environment, fallback to config
        base_url = os.environ.get("WEAVER_BASE_URL", None)
        if base_url is None:
            base_url = weaver_config.get("base_url", None)

        # Get weaver_api_key from environment, fallback to config
        weaver_api_key = os.environ.get("WEAVER_API_KEY", None)
        if weaver_api_key is None:
            weaver_api_key = weaver_config.get("api_key", None)

        # Report error if weaver_api_key is not provided in either environment or config
        if not weaver_api_key:
            error_msg = (
                "WEAVER_API_KEY is required but not found in environment variables or config"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Set weaver_api_key in environment for downstream usage
        os.environ["WEAVER_API_KEY"] = weaver_api_key

        base_model = self._config.service.inference_service.model
        tokenizer_path = self._config.service.inference_service.get("tokenizer", None)
        lora_rank = weaver_config.get("lora_rank", 32)

        training_mode = weaver_config.get("training_mode", "lora")

        logger.info(f"Initializing WeaverServiceHolder: base_model={base_model}, rank={lora_rank}")

        if self._launch_mode == "local":
            self._weaver_service_holder = WeaverServiceHolder(
                base_model=base_model,
                lora_rank=lora_rank,
                base_url=base_url,
                tokenizer_path=tokenizer_path,
                training_mode=training_mode,
            )
        elif self._launch_mode == "ray":
            env_vars = _get_minimal_env_vars()
            env_vars["WEAVER_API_KEY"] = weaver_api_key
            if base_url:
                env_vars["WEAVER_BASE_URL"] = base_url
            WeaverServiceActor = ray.remote(WeaverServiceHolder).options(
                runtime_env={"env_vars": env_vars},
            )
            self._weaver_service_holder = WeaverServiceActor.remote(
                base_model=base_model,
                lora_rank=lora_rank,
                base_url=base_url,
                training_mode=training_mode,
            )
            if hasattr(self._weaver_service_holder, "__ray_ready__"):
                try:
                    ray.get(self._weaver_service_holder.__ray_ready__.remote(), timeout=180)  # type: ignore[attr-defined]
                    logger.info("WeaverServiceHolder Ray actor is ready")
                except Exception as e:
                    logger.error(f"WeaverServiceHolder failed initialization: {e}")
                    raise

        logger.info("WeaverServiceHolder initialized successfully")
