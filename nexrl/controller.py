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
from typing import Any

import ray
from omegaconf import DictConfig

from .activity_tracker import ActivityTracker, ActivityTrackerProxy
from .algorithm_processor.grpo_processor import GRPOProcessor
from .data_loader import BaseDataLoader, TorchDataLoader
from .executor import execute
from .mock.mock_algorithm_processor import (
    BaseAlgorithmProcessor,
    MockAlgorithmProcessor,
)
from .mock.mock_data_loader import MockDataLoader
from .mock.mock_rollout_worker import MockRolloutWorker
from .nexrl_types import ModelTag, NexRLRole
from .ray_resource_manager import RayResourceManager, _get_minimal_env_vars
from .rollout_worker.base_rollout_worker import BaseRolloutWorker
from .rollout_worker.simple_rollout_worker import SimpleRolloutWorker
from .rollout_worker.single_turn_math import SingleTurnMathAgent
from .train_batch_pool import TrainBatchPool
from .train_worker import TrainWorker
from .trajectory_pool import TrajectoryPool
from .utils.config_utils import insert_config
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

    Interaction relationship: DataLoader <- RolloutWorkerManager -> RolloutResultsPool <- AlgorithmProcessor -> TrainBatchPool <- ... (other training components)
    """

    # Public interfaces - accessible to external components
    trajectory_pool: TrajectoryPool
    train_batch_pool: TrainBatchPool
    weight_sync_controller: WeightSyncController
    dataloader: BaseDataLoader
    validate_dataloader: BaseDataLoader
    validator: Validator
    algorithm_processor: BaseAlgorithmProcessor
    rollout_workers: list[BaseRolloutWorker]
    train_worker: TrainWorker
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

        # Initialize modules using unified approach
        self._init_modules()

        logger.info("NexRL Controller initialized successfully")

    def run(self) -> None:
        """Start the training process"""
        logger.info("Starting NexRL training process...")

        execute(self.train_worker.initialize_workers)

        # Load initial checkpoint
        self._load_initial_checkpoint()

        if self._config.validate.validate_before_train:
            self._start_validate(self._config.service.inference_service.model_tag)

        # Start all components
        execute(self.train_worker.run)
        for worker in self.rollout_workers:
            execute(worker.run)
        execute(self.algorithm_processor.run)

        if self._config.validate.validate_before_train:
            self._end_validate(self._config.service.inference_service.model_tag)

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

            model_tag = self._config.service.inference_service.model_tag
            if execute(self.weight_sync_controller.is_waiting_for_validation):
                self._run_validate(model_tag)

            if self._check_finish():
                logger.info(
                    f"Training job finish, wandb: {self.activity_tracker.experiment_logger.wandb_url} "
                )
                break

            # Check module liveness at configured interval
            if health_check_counter % health_check_interval == 0:
                is_all_modules_alive = self._check_module_liveness()
                if not is_all_modules_alive:
                    logger.error("Module liveness check detected dead modules. Terminating system.")
                    break
                logger.info(f"Module liveness check passed at {time.time()}")

            # Check system health at configured interval
            if exception_check_counter >= exception_check_interval:
                is_system_healthy = self._check_module_exceptions()
                if not is_system_healthy:
                    logger.error(
                        "System health check detected critical issues. Terminating system."
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

        # Signal all workers to stop gracefully
        execute(self.algorithm_processor.stop)
        for worker in self.rollout_workers:
            execute(worker.stop)
        execute(self.train_worker.stop)

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
        self._train_service_client = create_train_service_client(
            self._config.service.train_service.backend,
            self._config.service.train_service.url,
            self._config.service.train_service.get("identifier", None),
        )

        if self._config.resume.mode != "disable":
            self._load_resume_checkpoint()

        with self._train_service_client.actor_context():
            self._train_service_client.save_checkpoint(
                self._config.train_worker.sync_weight_path,
                global_step=0,
                saved_fully_shared_ckpt=False,
                save_weight_only=True,
                remove_previous_ckpt=False,
            )
        execute(
            self.weight_sync_controller.sync_weight_to_rollout_service,
            self._config.service.train_service.model_tag,
        )

    def _load_resume_checkpoint(self):
        """Load checkpoint based on resume configuration"""
        if self._config.resume.mode == "disable":
            logger.info("Resume mode is disabled, training from scratch")
            return

        checkpoint_folder = self._config.train_worker.checkpoint_path
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

        # Update train worker's global step
        execute(self.train_worker.set_train_step, global_step)
        logger.info(f"Training will resume from step {global_step}")

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
            execute(self.train_worker.get_train_step) >= self._config.train_worker.total_train_steps
        )
        if steps_reached:
            logger.info("Finish check: steps_reached=True -> stopping")
            return True

        # Quiescence stop: only when all sources/sinks are drained and no work is in-flight
        rollout_pool_empty = execute(self.trajectory_pool.is_empty)
        train_pool_empty = execute(self.train_batch_pool.is_empty)
        dataloader_done = execute(self.dataloader.is_finished)
        tracker_idle = execute(self.activity_tracker.is_quiescent)

        logger.debug(
            f"Finish check: rollout_pool_empty={rollout_pool_empty}, "
            f"train_pool_empty={train_pool_empty}, dataloader_done={dataloader_done}, "
            f"tracker_idle={tracker_idle}"
        )

        return rollout_pool_empty and train_pool_empty and dataloader_done and tracker_idle

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
    def _get_module_class_for_role(self, role: NexRLRole, config_type: str) -> type:
        """Get the class for a specific module type and config type"""
        registry: dict[NexRLRole, dict[str, type]] = {
            NexRLRole.DATA_LOADER: {
                "mock": MockDataLoader,
                "torch": TorchDataLoader,
            },
            NexRLRole.ALGORITHM_PROCESSOR: {
                "mock": MockAlgorithmProcessor,
                "grpo": GRPOProcessor,
            },
            NexRLRole.ROLLOUT_WORKER: {
                "mock": MockRolloutWorker,
                "simple": SimpleRolloutWorker,
                "single_turn_math": SingleTurnMathAgent,
            },
            NexRLRole.TRAIN_WORKER: {
                "default": TrainWorker,
            },
            NexRLRole.TRAJECTORY_POOL: {
                "default": TrajectoryPool,
            },
            NexRLRole.TRAIN_BATCH_POOL: {
                "default": TrainBatchPool,
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
        module_class = self._get_module_class_for_role(role, module_type)
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
                    ray.get(module._actor.__ray_ready__.remote(), timeout=60)
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

    def _init_modules(self):
        """Initialize all modules using unified approach (supports local, Ray, and hybrid modes)"""
        launch_mode = self._config.launch_mode
        logger.info(f"Initializing modules in {launch_mode} mode...")

        # Define module configurations - single source of truth using roles as keys
        module_configs = {
            NexRLRole.TRAJECTORY_POOL: self._config.trajectory_pool,
            NexRLRole.TRAIN_BATCH_POOL: self._config.train_batch_pool,
            NexRLRole.WEIGHT_SYNC_CONTROLLER: self._config.weight,
            NexRLRole.DATA_LOADER: self._config.data,
            NexRLRole.ALGORITHM_PROCESSOR: self._config.algorithm,
            NexRLRole.ROLLOUT_WORKER: self._config.rollout_worker,
            NexRLRole.TRAIN_WORKER: self._config.train_worker,
            NexRLRole.VALIDATE_DATALOADER: self._config.validate.data,
            NexRLRole.VALIDATOR: self._config.validate.eval,
        }

        # Add inference_service config to weight and rollout_worker config
        insert_config(
            self._config.weight, "inference_service", self._config.service.inference_service
        )
        insert_config(
            self._config.rollout_worker, "inference_service", self._config.service.inference_service
        )
        insert_config(
            self._config.algorithm, "inference_service", self._config.service.inference_service
        )
        insert_config(self._config.algorithm, "train_service", self._config.service.train_service)
        insert_config(
            self._config.train_worker, "train_service", self._config.service.train_service
        )

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
        self.train_batch_pool = self._init_module_for_role(
            NexRLRole.TRAIN_BATCH_POOL,
            module_configs[NexRLRole.TRAIN_BATCH_POOL],
            num_workers=1,
        )[0]
        self.algorithm_processor = self._init_module_for_role(
            NexRLRole.ALGORITHM_PROCESSOR,
            module_configs[NexRLRole.ALGORITHM_PROCESSOR],
            num_workers=1,
        )[0]

        self.rollout_workers = self._init_module_for_role(
            NexRLRole.ROLLOUT_WORKER,
            module_configs[NexRLRole.ROLLOUT_WORKER],
            num_workers=self._config.rollout_worker.num_workers,
        )
        self.train_worker = self._init_module_for_role(
            NexRLRole.TRAIN_WORKER,
            module_configs[NexRLRole.TRAIN_WORKER],
            num_workers=self._config.train_worker.num_workers,  # usually 1
        )[0]
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
            self.algorithm_processor.set_module_references,
            trajectory_pool=self.trajectory_pool,
            train_batch_pool=self.train_batch_pool,
        )
        execute(
            self.train_worker.set_module_references,
            train_batch_pool=self.train_batch_pool,
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
            role=NexRLRole.TRAIN_BATCH_POOL,
            cls=self._get_module_class_for_role(
                NexRLRole.TRAIN_BATCH_POOL, module_configs[NexRLRole.TRAIN_BATCH_POOL].type
            ),
            config=module_configs[NexRLRole.TRAIN_BATCH_POOL],
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

        self._ray_resource_manager.register_role(
            role=NexRLRole.ALGORITHM_PROCESSOR,
            cls=self._get_module_class_for_role(
                NexRLRole.ALGORITHM_PROCESSOR, module_configs[NexRLRole.ALGORITHM_PROCESSOR].type
            ),
            config=module_configs[NexRLRole.ALGORITHM_PROCESSOR],
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
                NexRLRole.ROLLOUT_WORKER, module_configs[NexRLRole.ROLLOUT_WORKER].type
            ),
            config=module_configs[NexRLRole.ROLLOUT_WORKER],
            count=rollout_worker_count,
            colocation_group=None,  # None = standalone
        )

        # Train worker - needs its own dedicated process
        self._ray_resource_manager.register_role(
            role=NexRLRole.TRAIN_WORKER,
            cls=self._get_module_class_for_role(
                NexRLRole.TRAIN_WORKER, module_configs[NexRLRole.TRAIN_WORKER].type
            ),
            config=module_configs[NexRLRole.TRAIN_WORKER],
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
