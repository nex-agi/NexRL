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

import logging
import os
import threading
import time
import uuid
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .error_reporter import ErrorReporter, ErrorSeverity
from .utils.tracking import Tracking

logger = logging.getLogger(__name__)


class ActivityTracker:
    """
    Centralized tracker of in-flight work across modules.

    Modules communicate with this tracker through ActivityTrackerProxy instances.
    The tracker provides start/end methods for activity tracking and is used by
    the controller to monitor system quiescence and module health.
    """

    experiment_logger: Tracking

    def __init__(self, config: DictConfig, max_errors: int = 1000) -> None:
        self._cv = threading.Condition()
        self._count: int = 0
        self._by_module: dict[str, int] = {}
        self._tokens: dict[str, str] = {}
        self._error_reporter: ErrorReporter = ErrorReporter(max_errors=max_errors)
        self._module_refs: dict[str, Any] = (
            {}
        )  # module_name -> module_reference (for Ray health checking)
        self._rollout_worker_modules: set[str] = set()  # Track which modules are rollout workers
        self._current_training_step: int = 0  # Track current training step
        self._config = config

        self.experiment_logger = Tracking(
            project_name=self._config.project_name,
            experiment_name=self._config.experiment_name,
            default_backend=self._config.logger.backend,
            config=OmegaConf.to_container(config, resolve=True),
        )

        # Rollout progress monitoring (nested under runtime_monitor)
        runtime_monitor_config = config.get("runtime_monitor", {})
        progress_config = runtime_monitor_config.get("rollout_progress_monitor", {})
        self._progress_enabled = progress_config.get("enabled", True)
        self._progress_update_interval = progress_config.get("update_interval", 1.0)
        self._progress_log_interval = progress_config.get("log_interval", 10)
        self._progress_use_tqdm = progress_config.get("use_tqdm", True)
        self._progress_thread: threading.Thread | None = None
        self._progress_stop_event = threading.Event()
        self._progress_update_count = 0
        self._progress_last_metrics: dict[str, Any] | None = None

    def start(self, module: str, work: str) -> str:  # pylint: disable=unused-argument
        token = str(uuid.uuid4())
        with self._cv:
            self._count += 1
            self._by_module[module] = self._by_module.get(module, 0) + 1
            self._tokens[token] = module
        return token

    def end(self, token: str) -> None:
        with self._cv:
            module: str | None = self._tokens.pop(token, None)
            if module is None:
                return
            self._count -= 1
            self._by_module[module] = max(0, self._by_module.get(module, 0) - 1)
            if self._count == 0:
                self._cv.notify_all()

    def is_quiescent(self) -> bool:
        with self._cv:
            return self._count == 0

    def is_rollout_worker_quiescent(self) -> bool:
        """Check if all registered rollout workers are currently idle (no active work)

        Returns:
            bool: True if all rollout workers have no active work, False otherwise
        """
        with self._cv:
            for module_name in self._rollout_worker_modules:
                if self._by_module.get(module_name, 0) > 0:
                    return False
            return True

    def wait_quiescent(self, timeout: float | None = None) -> bool:
        with self._cv:
            end_time = None if timeout is None else (time.time() + timeout)
            while self._count != 0:
                remaining = None if end_time is None else max(0.0, end_time - time.time())
                if end_time is not None and remaining is not None and remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
            return True

    def register_module(
        self, module_name: str, module_ref: Any, is_rollout_worker: bool = False
    ) -> None:
        """Register a module reference for health checking

        Args:
            module_name: Name of the module
            module_ref: Reference to the module (for health checking)
            is_rollout_worker: Whether this module is a rollout worker (for special monitoring)
        """
        with self._cv:
            self._module_refs[module_name] = module_ref
            if is_rollout_worker:
                self._rollout_worker_modules.add(module_name)
            self._by_module[module_name] = 0

    def report_exception(
        self,
        module: str,
        work: str,
        exception: Exception,
        severity: ErrorSeverity | None = None,
    ) -> str:
        """
        Report an exception through the error reporter

        Args:
            module: Module name where the exception occurred
            work: Work context where the exception occurred
            exception: The exception that was raised
            severity: Severity level (defaults to ERROR)

        Returns:
            str: Error ID of the reported exception
        """
        severity = severity or ErrorSeverity.ERROR
        return self._error_reporter.report_exception(module, work, exception, severity)

    # -- module liveness checking --
    def check_module_liveness(self, timeout: float = 5.0) -> bool:
        """
        Check module liveness and return whether all modules are alive.

        In local mode, error reporting relies on exceptions and logging which are
        already visible in the controller process, so no liveness checking is needed.

        Args:
            timeout: Timeout for liveness check operations (Ray mode only)

        Returns:
            bool: True if all modules are alive, False if any modules are dead
        """
        launch_mode = os.getenv("NEXRL_LAUNCH_MODE", "local")
        if launch_mode == "ray":
            return self._check_ray_module_liveness(timeout)
        # Local mode: rely on exception reporting and logging - assume all modules are alive
        return True

    def _check_ray_module_liveness(self, timeout: float = 5.0) -> bool:
        """
        Check Ray actor liveness using Ray's actor monitoring and health check.

        Args:
            timeout: Timeout for liveness check operations

        Returns:
            bool: True if all modules are alive, False if any modules are dead
        """
        try:
            import ray
        except ImportError:
            return True  # If Ray is not available, assume modules are alive

        with self._cv:
            modules_to_check = list(self._module_refs.items())

        dead_modules = []
        for module_name, module_ref in modules_to_check:
            try:
                # For Ray actors, check if the actor is still alive
                if hasattr(module_ref, "_ray_actor_id"):
                    # First check if actor is ready (basic liveness)
                    ray.get(module_ref.__ray_ready__.remote(), timeout=timeout)

                    # Then perform health check to ensure responsiveness
                    # This calls the health_check() method we added to NexRLModule
                    try:
                        health_result = ray.get(module_ref.health_check.remote(), timeout=timeout)
                        if not health_result:
                            self._report_module_death(
                                module_name, "Module health check returned False"
                            )
                            dead_modules.append(module_name)
                    except ray.exceptions.GetTimeoutError:
                        self._report_module_death(
                            module_name, f"Module health check timed out after {timeout}s"
                        )
                        dead_modules.append(module_name)
                else:
                    # This might be a local object in Ray mode, skip active checking
                    continue
            except ray.exceptions.RayActorError as e:
                self._report_module_death(module_name, f"Ray actor died: {e}")
                dead_modules.append(module_name)
            except Exception as e:
                if "actor is dead" in str(e).lower() or "actor died" in str(e).lower():
                    self._report_module_death(module_name, f"Ray actor unresponsive: {e}")
                    dead_modules.append(module_name)

        # Return False if any modules died, True if all are alive
        if dead_modules:
            logger.error(f"Liveness check detected dead modules: {dead_modules}.")
            return False
        return True

    def _report_module_death(self, module_name: str, reason: str) -> None:
        """Report a confirmed module death"""
        self._error_reporter.report_error(
            module_name=module_name,
            work_context="system_monitoring",
            message=f"Module died: {reason}",
            severity=ErrorSeverity.ERROR,
            details={"death_reason": reason},
        )

        # Clean up references to dead module
        with self._cv:
            self._module_refs.pop(module_name, None)
            self._rollout_worker_modules.discard(module_name)

    # -- module running status summary --
    def get_running_status_summary(self) -> str:
        """Get a human-readable summary of current activity status"""
        with self._cv:
            if self._count == 0:
                return "System is quiescent (no active work)"

            module_running_status = []
            for module, count in self._by_module.items():
                if count > 0:
                    module_running_status.append(f"{module}: {count}")

            return f"Active work items: {self._count} total ({', '.join(module_running_status)})"

    def get_error_health_status(self) -> dict[str, Any]:
        """Get health status from the error reporter"""
        return self._error_reporter.get_health_status()

    # -- training step tracking --
    def set_training_step(self, step: int) -> None:
        """
        Set the current training step

        Args:
            step: The current training step
        """
        with self._cv:
            self._current_training_step = step

        if step == 1 and self.experiment_logger.feishu_logger is not None:
            self.experiment_logger.feishu_logger.post(
                content=f'"Experiment: {self._config.project_name}/{self._config.experiment_name}"'
                f"begin training, wandb: {self.experiment_logger.wandb_url} ",
                title="Begin!",
            )

    def get_training_step(self) -> int:
        """
        Get the current training step

        Returns:
            int: The current training step
        """
        with self._cv:
            return self._current_training_step

    def experiment_logger_post(self, backend: str, **kwargs) -> None:
        if backend == "feishu":
            if self.experiment_logger.feishu_logger is not None:
                self.experiment_logger.feishu_logger.post(
                    content=kwargs["content"], title=kwargs["title"]
                )
        elif backend == "wandb":
            if "step" not in kwargs:
                kwargs["step"] = self._current_training_step
            self.experiment_logger.log(data=kwargs["data"], step=kwargs["step"])
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # -- progress monitoring --
    def get_progress_metrics(self) -> dict[str, Any]:
        """
        Collect progress metrics from all modules for progress monitoring.
        Uses _module_refs to find DataLoader, TrajectoryPool, and WeightSyncController.

        Returns:
            dict with keys:
                - active_rollout_workers: Number of rollout workers currently active
                - dataloader: Progress info from dataloader (if available)
                - trajectory_pool: Progress info from trajectory pool (if available)
                - sync_mode: Current weight sync mode
        """
        from .executor import execute

        metrics: dict[str, Any] = {}

        # Get active rollout worker count
        with self._cv:
            active_workers = sum(
                self._by_module.get(module_name, 0) for module_name in self._rollout_worker_modules
            )
            metrics["active_rollout_workers"] = active_workers

        # Find singleton modules from _module_refs (they are registered with "-0" suffix)
        # The value in _module_refs is the ret list, which contains the actual module as the only element
        # Module names use lowercase with underscores (from role enum values)
        dataloader_list = self._module_refs.get("data_loader-0")
        trajectory_pool_list = self._module_refs.get("trajectory_pool-0")
        weight_sync_controller_list = self._module_refs.get("weight_sync_controller-0")

        # Extract the actual module (first element of the list)
        dataloader = dataloader_list[0] if dataloader_list and len(dataloader_list) > 0 else None
        trajectory_pool = (
            trajectory_pool_list[0]
            if trajectory_pool_list and len(trajectory_pool_list) > 0
            else None
        )
        weight_sync_controller = (
            weight_sync_controller_list[0]
            if weight_sync_controller_list and len(weight_sync_controller_list) > 0
            else None
        )

        # Get dataloader progress (if available)
        if dataloader is not None:
            try:
                dataloader_info = execute(dataloader.get_progress_info)
                metrics["dataloader"] = dataloader_info
            except Exception as e:
                logger.warning(f"Could not get dataloader progress: {e}", exc_info=True)
                metrics["dataloader"] = None
        else:
            logger.debug("DataLoader not found in module registry")
            metrics["dataloader"] = None

        # Get trajectory pool progress (if available)
        if trajectory_pool is not None:
            try:
                pool_info = execute(trajectory_pool.get_progress_info)
                metrics["trajectory_pool"] = pool_info
            except Exception as e:
                logger.warning(f"Could not get trajectory pool progress: {e}", exc_info=True)
                metrics["trajectory_pool"] = None
        else:
            logger.debug("TrajectoryPool not found in module registry")
            metrics["trajectory_pool"] = None

        # Get sync mode (if available)
        if weight_sync_controller is not None:
            try:
                sync_mode = execute(
                    lambda: weight_sync_controller._sync_mode  # pylint: disable=protected-access
                )
                metrics["sync_mode"] = sync_mode
            except Exception as e:
                logger.warning(f"Could not get sync mode: {e}", exc_info=True)
                metrics["sync_mode"] = None
        else:
            logger.debug("WeightSyncController not found in module registry")
            metrics["sync_mode"] = None

        return metrics

    def start_progress_monitor(self) -> None:
        """Start the rollout progress monitoring thread"""
        if not self._progress_enabled:
            logger.info("Rollout progress monitoring is disabled")
            return

        if self._progress_thread is not None and self._progress_thread.is_alive():
            logger.warning("Rollout progress monitor already running")
            return

        # Log available modules for debugging
        with self._cv:
            available_modules = list(self._module_refs.keys())
            logger.info(
                f"Starting rollout progress monitor. Available modules: {available_modules}"
            )

        self._progress_stop_event.clear()
        self._progress_thread = threading.Thread(
            target=self._progress_monitor_loop, daemon=True, name="RolloutProgressMonitor"
        )
        self._progress_thread.start()
        logger.info("Rollout progress monitor started")

    def stop_progress_monitor(self) -> None:
        """Stop the rollout progress monitoring thread"""
        if not self._progress_enabled:
            return

        if self._progress_thread is None or not self._progress_thread.is_alive():
            return

        logger.info("Stopping rollout progress monitor...")
        self._progress_stop_event.set()

        if self._progress_thread is not None:
            self._progress_thread.join(timeout=5.0)
            if self._progress_thread.is_alive():
                logger.warning("Rollout progress monitor thread did not stop gracefully")

        logger.info("Rollout progress monitor stopped")

    def _progress_monitor_loop(self) -> None:
        """Main rollout progress monitoring loop - runs in separate thread"""
        try:
            while not self._progress_stop_event.is_set():
                try:
                    # Collect metrics
                    metrics = self.get_progress_metrics()
                    self._progress_last_metrics = metrics

                    # Update display
                    self._update_progress_display(metrics)

                    self._progress_update_count += 1

                except Exception as e:
                    logger.error(f"Error in rollout progress monitor loop: {e}", exc_info=True)

                # Sleep until next update
                time.sleep(self._progress_update_interval)

        except Exception as e:
            logger.error(f"Fatal error in rollout progress monitor loop: {e}", exc_info=True)

    def _update_progress_display(self, metrics: dict[str, Any]) -> None:
        """
        Update progress display with latest metrics

        Args:
            metrics: Progress metrics
        """
        # Extract metrics
        active_workers = metrics.get("active_rollout_workers", 0)
        dataloader_info = metrics.get("dataloader")
        pool_info = metrics.get("trajectory_pool")
        sync_mode = metrics.get("sync_mode")

        # Determine if we should show batch progress
        show_batch_progress = sync_mode in ["sync", "batch-async"] and dataloader_info is not None

        # Format display strings
        display_parts = []

        # 1. DataLoader batch progress (only in sync/batch-async modes)
        if show_batch_progress and dataloader_info:
            batch_size = dataloader_info.get("batch_size", 0)
            batch_remaining = dataloader_info.get("batch_remaining", 0)
            batch_processed = dataloader_info.get("batch_processed", 0)
            display_parts.append(
                f"Batch: {batch_processed}/{batch_size} (remaining: {batch_remaining})"
            )

        # 2. Active rollout workers
        display_parts.append(f"Workers: {active_workers} active")

        # 3. Trajectories received
        if pool_info:
            finished_trajectories = pool_info.get("finished_trajectories", 0)
            display_parts.append(f"Trajectories: {finished_trajectories} finished")

        # Log periodically
        if self._progress_update_count % self._progress_log_interval == 0:
            if self._progress_use_tqdm:
                try:
                    from tqdm import tqdm

                    message = f"Progress: {' | '.join(display_parts)}"
                    tqdm.write(message)
                except ImportError:
                    logger.info(f"Progress: {' | '.join(display_parts)}")
            else:
                logger.info(f"Progress: {' | '.join(display_parts)}")


class ActivityTrackerProxy:
    """
    Local proxy for ActivityTracker that provides the same interface but communicates
    with a centralized ActivityTracker (which may be local or a Ray actor).

    This proxy is stateless and simply forwards all calls to the central tracker
    using the execute() function for Ray compatibility.
    """

    def __init__(self, central_tracker: Any):
        """
        Initialize the proxy with a reference to the central tracker.

        Args:
            central_tracker: Reference to the central ActivityTracker (local or Ray actor)
        """
        self._central_tracker: ActivityTracker = central_tracker

    def track(self, module: str, work: str, auto_report_errors: bool = True) -> "_ProxyTrackCtx":
        """
        Create a context manager for tracking work and optionally reporting errors.

        Args:
            module: Module name doing the work
            work: Description of the work being done
            auto_report_errors: Whether to automatically report exceptions to error reporter

        Returns:
            Context manager that tracks activity and optionally reports errors
        """
        return ActivityTrackerProxy._ProxyTrackCtx(
            self._central_tracker, module, work, auto_report_errors
        )

    class _ProxyTrackCtx:
        """
        Context manager that forwards tracking calls to the central ActivityTracker.
        """

        def __init__(
            self,
            central_tracker: Any,
            module: str,
            work: str,
            auto_report_errors: bool = True,
        ) -> None:
            self._central_tracker = central_tracker
            self._token: str | None = None
            self._module = module
            self._work = work
            self._auto_report_errors = auto_report_errors

        def __enter__(self):
            # Import here to avoid circular imports
            from .executor import execute

            self._token = execute(self._central_tracker.start, self._module, self._work)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            # Import here to avoid circular imports
            from .executor import execute

            # Report exception automatically if enabled and exception occurred
            if self._auto_report_errors and exc_type is not None and exc_value is not None:
                try:
                    execute(
                        self._central_tracker.report_exception, self._module, self._work, exc_value
                    )
                except Exception:
                    # Don't let error reporting interfere with original exception
                    pass

            # End activity tracking
            if self._token is not None:
                execute(self._central_tracker.end, self._token)

            # Don't suppress the original exception
            return False

    def is_rollout_worker_quiescent(self) -> bool:
        """Check if all registered rollout workers are currently idle (no active work)

        Returns:
            bool: True if all rollout workers have no active work, False otherwise
        """
        from .executor import execute

        return execute(self._central_tracker.is_rollout_worker_quiescent)

    def set_training_step(self, step: int) -> None:
        """
        Set the current training step in the central tracker

        Args:
            step: The current training step
        """
        from .executor import execute

        execute(self._central_tracker.set_training_step, step)

    def get_training_step(self) -> int:
        """
        Get the current training step from the central tracker

        Returns:
            int: The current training step
        """
        from .executor import execute

        return execute(self._central_tracker.get_training_step)

    def experiment_logger_post(self, backend: str, **kwargs) -> None:
        from .executor import execute

        if backend == "wandb":
            assert "data" in kwargs, "data is required for wandb"
        elif backend == "feishu":
            assert "content" in kwargs, "content is required for feishu"
            assert "title" in kwargs, "title is required for feishu"
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        execute(self._central_tracker.experiment_logger_post, backend, **kwargs)

    def get_progress_metrics(self) -> dict[str, Any]:
        """
        Get progress metrics from the central tracker

        Returns:
            dict: Progress metrics from all modules
        """
        from .executor import execute

        return execute(self._central_tracker.get_progress_metrics)

    def start_progress_monitor(self) -> None:
        """Start the rollout progress monitor in the central tracker"""
        from .executor import execute

        execute(self._central_tracker.start_progress_monitor)

    def stop_progress_monitor(self) -> None:
        """Stop the rollout progress monitor in the central tracker"""
        from .executor import execute

        execute(self._central_tracker.stop_progress_monitor)
