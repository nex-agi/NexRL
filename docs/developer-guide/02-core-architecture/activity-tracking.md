# Activity Tracking System

The Activity Tracking system provides comprehensive monitoring of in-flight work, module health checking, error reporting, and experiment logging across the NexRL framework.

## Overview

The activity tracking system consists of two main components:

1. **ActivityTracker**: Centralized tracker that monitors all module activity
2. **ActivityTrackerProxy**: Local proxies that forward tracking calls to the central tracker

## ActivityTracker

**Location**: `nexrl/activity_tracker.py`

**Purpose**: Centralized tracker of in-flight work across all modules

### Constructor

```python
def __init__(self, config: DictConfig, max_errors: int = 1000)
```

**Parameters**:
- `config`: Configuration containing project and experiment names for logging
- `max_errors`: Maximum number of errors to retain in memory

**Initializes**:
- Threading primitives for tracking
- Error reporter for exception management
- Experiment logger (Tracking instance) for W&B/etc.
- Module references for health checking

### Activity Tracking Methods

#### start(module: str, work: str) -> str

Registers the start of work.

```python
token = activity_tracker.start("RolloutWorker-1", "processing_task")
```

**Parameters**:
- `module`: Name of the module performing work
- `work`: Description of the work being performed

**Returns**: Unique token for this work item

**Usage**: Called automatically by `ActivityTrackerProxy.track()` context manager

#### end(token: str) -> None

Registers the completion of work.

```python
activity_tracker.end(token)
```

**Parameters**:
- `token`: Token from corresponding `start()` call

**Usage**: Called automatically by `ActivityTrackerProxy.track()` context manager

### Status Monitoring Methods

#### is_quiescent() -> bool

Checks if all work has completed.

```python
if activity_tracker.is_quiescent():
    print("System is idle")
```

**Returns**: `True` if no work is currently in progress across all modules

**Usage**: Controller uses this to detect training completion

#### is_rollout_worker_quiescent() -> bool

Checks if all rollout workers are idle.

```python
if activity_tracker.is_rollout_worker_quiescent():
    print("All rollout workers are idle")
```

**Returns**: `True` if all registered rollout workers have no active work

**Usage**: Validator uses this to detect completion of validation

#### wait_quiescent(timeout: float | None = None) -> bool

Waits for system to become quiescent.

```python
if activity_tracker.wait_quiescent(timeout=30.0):
    print("System became quiescent")
else:
    print("Timeout waiting for quiescence")
```

**Parameters**:
- `timeout`: Maximum time to wait in seconds, or `None` for indefinite

**Returns**: `True` if quiescence achieved, `False` if timeout

**Usage**: Controller uses this during graceful shutdown

#### get_running_status_summary() -> str

Gets a human-readable summary of current activity.

```python
status = activity_tracker.get_running_status_summary()
print(status)
# Output: "RolloutWorker-1: 3 tasks, Trainer: 1 batch, ..."
```

**Returns**: String describing active work by module

**Usage**: Debugging and monitoring

### Module Health Tracking

#### register_module(module_name: str, module_ref: Any, is_rollout_worker: bool = False)

Registers a module for health checking.

```python
activity_tracker.register_module(
    module_name="RolloutWorker-1",
    module_ref=worker_actor,
    is_rollout_worker=True
)
```

**Parameters**:
- `module_name`: Name for identification
- `module_ref`: Reference to module (local object or Ray actor)
- `is_rollout_worker`: Whether this is a rollout worker (for specialized monitoring)

**Usage**: Controller registers all modules during initialization

#### check_module_liveness(timeout: float = 5.0) -> bool

Checks if all registered modules are alive and responsive.

```python
if not activity_tracker.check_module_liveness(timeout=5.0):
    logger.error("Some modules are dead")
```

**Parameters**:
- `timeout`: Timeout for Ray operations in seconds

**Returns**: `True` if all modules are alive, `False` if any are dead

**Process**:
1. Call `health_check()` on each registered module
2. For Ray actors, use `ray.get()` with timeout
3. For local modules, call directly
4. Return `False` if any fail or timeout

**Usage**: Controller monitors module health periodically

### Error Reporting

#### report_exception(module: str, work: str, exception: Exception, severity: ErrorSeverity | None = None) -> str

Reports an exception through the error reporter.

```python
try:
    process_data()
except Exception as e:
    error_id = activity_tracker.report_exception(
        module="RolloutWorker-1",
        work="processing_task",
        exception=e,
        severity=ErrorSeverity.ERROR
    )
```

**Parameters**:
- `module`: Module where exception occurred
- `work`: Work context
- `exception`: The exception that was raised
- `severity`: Error severity (defaults to ERROR)

**Returns**: Unique error ID

**Severity Levels**:
- `ErrorSeverity.INFO`: Informational
- `ErrorSeverity.WARNING`: Warning
- `ErrorSeverity.ERROR`: Error (default)

#### get_error_health_status() -> dict[str, Any]

Gets error health status information.

```python
status = activity_tracker.get_error_health_status()
# {
#     "status": "healthy"|"warning"|"error",
#     "message": "...",
#     "recent_error_count": 0,
#     "error_level_count": 0,
#     "warning_level_count": 0
# }
```

**Returns**: Dictionary with health status and error counts

**Usage**: Controller checks this to decide if system should stop

### Training Step Tracking

#### set_training_step(step: int)

Updates the current training step.

```python
activity_tracker.set_training_step(100)
```

**Parameters**:
- `step`: Current training step number

**Usage**: Trainer updates this after each training step

#### get_training_step() -> int

Gets the current training step.

```python
current_step = activity_tracker.get_training_step()
```

**Returns**: Current training step number

**Usage**: Used for logging and experiment tracking

### Experiment Logging

#### experiment_logger_post(backend: str, **kwargs)

Posts metrics or messages to the experiment logging backend.

```python
# Log metrics
activity_tracker.experiment_logger_post(
    backend="wandb",
    data={"loss": 0.5, "reward": 1.0},
    step=100
)

# Log message
activity_tracker.experiment_logger_post(
    backend="feishu",
    content="Training completed!",
    title="Success"
)
```

**Parameters**:
- `backend`: Logging backend ("wandb", "feishu", etc.)
- `**kwargs`: Backend-specific parameters

**Common Parameters**:
- W&B: `data` (dict), `step` (int)
- Feishu: `content` (str), `title` (str)

**Usage**: All modules can log through this unified interface

## ActivityTrackerProxy

**Location**: `nexrl/activity_tracker.py`

**Purpose**: Local proxy that forwards tracking calls to central ActivityTracker

### Constructor

```python
def __init__(self, central_tracker: Any)
```

**Parameters**:
- `central_tracker`: Reference to central ActivityTracker (local or Ray actor)

**Usage**: Controller creates proxies and distributes to modules

### Context Manager Pattern

#### track(module: str, work: str, auto_report_errors: bool = True)

Context manager for automatic activity tracking.

```python
with activity_tracker.track("MyModule", "processing_batch"):
    # Perform work
    process_batch(batch)
    # Automatic activity tracking and error reporting
```

**Parameters**:
- `module`: Module name performing work
- `work`: Description of work
- `auto_report_errors`: Whether to automatically report exceptions (default: True)

**Features**:
- Automatically calls `start()` on entry
- Automatically calls `end()` on exit
- Automatically reports exceptions if `auto_report_errors=True`
- Safe exception re-raising

**Example with Manual Error Handling**:

```python
with activity_tracker.track("MyModule", "risky_work", auto_report_errors=False):
    try:
        risky_operation()
    except SpecificError as e:
        # Custom error handling
        logger.warning(f"Expected error: {e}")
        # Don't re-raise
```

### Forwarding Methods

The proxy forwards these methods to the central tracker:

```python
# Status checking
is_quiescent = activity_tracker_proxy.is_quiescent()
is_workers_idle = activity_tracker_proxy.is_rollout_worker_quiescent()

# Training step
activity_tracker_proxy.set_training_step(step)
current_step = activity_tracker_proxy.get_training_step()

# Logging
activity_tracker_proxy.experiment_logger_post(
    backend="wandb",
    data=metrics,
    step=step
)
```

## Usage Patterns

### Basic Activity Tracking

```python
class MyModule(NexRLModule):
    def process_work(self, item):
        with self._activity_tracker.track("MyModule", "process_work"):
            # Do work
            result = process(item)
            return result
```

### Multiple Work Items

```python
def process_batch(self, batch):
    with self._activity_tracker.track("MyModule", f"batch_{batch_id}"):
        for item in batch:
            # Each item tracked separately
            with self._activity_tracker.track("MyModule", f"item_{item_id}"):
                process_item(item)
```

### Error Reporting with Context

```python
def risky_operation(self):
    with self._activity_tracker.track("MyModule", "risky_op"):
        try:
            dangerous_work()
        except CriticalError as e:
            # Manual reporting with specific severity
            self._activity_tracker.report_exception(
                module=self._module_name,
                work="risky_op",
                exception=e,
                severity=ErrorSeverity.ERROR
            )
            # Decide whether to re-raise
            raise
```

### Logging Metrics

```python
def after_training_step(self, metrics, step):
    # Log to W&B
    self._activity_tracker.experiment_logger_post(
        backend="wandb",
        data={
            "train/loss": metrics["loss"],
            "train/reward": metrics["reward"],
        },
        step=step
    )

    # Update training step
    self._activity_tracker.set_training_step(step)
```

### Checking Quiescence

```python
# In controller
def wait_for_completion(self):
    if self.activity_tracker.wait_quiescent(timeout=60.0):
        logger.info("All work completed")
    else:
        status = self.activity_tracker.get_running_status_summary()
        logger.warning(f"Timeout. Active work: {status}")
```

## Integration with Controller

The controller uses the activity tracker throughout the system:

```python
class NexRLController:
    def __init__(self, config):
        # Create central tracker
        self.activity_tracker = ActivityTracker(config)

        # Create and distribute proxies to all modules
        for module in all_modules:
            proxy = ActivityTrackerProxy(self.activity_tracker)
            module.set_activity_tracker(proxy)

    def run(self):
        # Monitor system health
        while not self._check_finish():
            # Check quiescence
            if self.activity_tracker.is_quiescent():
                if self._all_data_processed():
                    break

            # Check module health
            if not self.activity_tracker.check_module_liveness():
                logger.error("Module health check failed")
                break

            # Check for errors
            health = self.activity_tracker.get_error_health_status()
            if health["status"] == "error":
                logger.error("System errors detected")
                break
```

## Best Practices

### Activity Tracking

1. **Always use the context manager** for automatic tracking
2. **Choose descriptive work names** for easy debugging
3. **Track significant work** (not trivial operations)
4. **Avoid excessive tracking** (hurts performance)

### Error Reporting

1. **Let auto_report_errors=True** handle most cases
2. **Use manual reporting** only for special cases
3. **Set appropriate severity levels**
4. **Include context** in error messages

### Logging

1. **Use consistent metric names** (e.g., "train/loss" not "loss")
2. **Log at appropriate intervals** (not every iteration)
3. **Include step numbers** for time series
4. **Aggregate before logging** (don't log individual samples)

### Monitoring

1. **Check quiescence** before shutting down
2. **Monitor health** periodically
3. **Review error status** regularly
4. **Use timeout values** appropriate for your workload

## Configuration

```yaml
runtime_monitor:
  exception_handling:
    enabled: true
    check_interval: 1.0  # Check every 1 second
    policy: "stop_on_error"  # or "continue", "stop_on_critical"

  health_check:
    enabled: true
    check_interval: 30.0  # Check every 30 seconds
    timeout: 5.0  # Timeout for health checks

logger:
  backend: "wandb"  # or "feishu", etc.

project_name: "NexRL-Project"
experiment_name: "experiment-v1"
```

## Next Steps

- Understand [Controller](./controller.md) usage of activity tracking
- Learn about [Error Reporting](../08-features/error-handling.md)
- Explore [Module Development](../13-best-practices/module-development.md) for integration
- Review [Distributed Execution](../11-distributed-execution/ray-integration.md) for Ray mode
