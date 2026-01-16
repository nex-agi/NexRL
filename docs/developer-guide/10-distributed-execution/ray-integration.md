# Ray Integration

Ray integration enables NexRL to scale across distributed compute resources with minimal code changes.

## Overview

NexRL supports two execution modes:
- **Local Mode**: All components run in a single process
- **Ray Mode**: Components run as distributed Ray actors

The framework automatically handles execution differences through the `executor` module.

## Launch Modes

### Local Mode

Single-process execution for development and debugging.

```python
import hydra
from nexrl import NexRLController

@hydra.main(config_path="config", config_name="train")
def main(config):
    config.launch_mode = "local"
    controller = NexRLController(config)
    controller.run()
```

**Characteristics:**
- All components in same process
- Direct method calls (no serialization)
- Easy debugging with standard tools
- Limited scalability

### Ray Mode

Distributed execution across multiple nodes/GPUs.

```python
import ray
import hydra
from nexrl import NexRLController

@hydra.main(config_path="config", config_name="train")
def main(config):
    config.launch_mode = "ray"
    ray.init(address="auto")  # Connect to Ray cluster

    # Create controller as Ray actor
    ControllerActor = ray.remote(NexRLController)
    controller = ControllerActor.remote(config)
    ray.get(controller.run.remote())

    ray.shutdown()
```

**Characteristics:**
- Components distributed as Ray actors
- Parallel execution across resources
- Automatic fault tolerance
- Scalable to large clusters

## Execution Interface

The `executor` module provides a unified interface for both modes.

### execute()

Synchronous execution with automatic mode detection.

```python
from nexrl.executor import execute

# Works in both local and Ray modes
result = execute(module.method, arg1, arg2)
```

**Implementation (`nexrl/executor.py`):**

```python
def execute(func: Any, *args, **kwargs) -> Any:
    """Execute synchronously, detecting local vs Ray remote."""
    launch_mode = os.getenv("NEXRL_LAUNCH_MODE", "local")

    if launch_mode == "local":
        return func(*args, **kwargs)
    elif launch_mode == "ray":
        if _is_ray_remote_method(func):
            return ray.get(func.remote(*args, **kwargs), timeout=40)
        else:
            return func(*args, **kwargs)
```

**Features:**
- Automatic Ray remote method detection
- Configurable timeout (default 40 seconds)
- Timeout protection against deadlocks
- Transparent fallback to local execution

### execute_async()

Asynchronous execution for parallel operations.

```python
from nexrl.executor import execute_async

# Launch multiple operations in parallel
refs = [execute_async(worker.step, task) for task in tasks]

# Wait for completion (Ray mode)
if launch_mode == "ray":
    results = ray.get(refs)
else:
    results = refs  # Already executed in local mode
```

**Implementation:**

```python
def execute_async(func: Any, *args, **kwargs) -> Any:
    """Execute asynchronously."""
    launch_mode = os.getenv("NEXRL_LAUNCH_MODE", "local")

    if launch_mode == "local":
        return func(*args, **kwargs)  # Immediate execution
    elif launch_mode == "ray":
        if _is_ray_remote_method(func):
            return func.remote(*args, **kwargs)  # Return ObjectRef
        else:
            return func(*args, **kwargs)
```

## RayResourceManager

Manages Ray actor creation and colocation.

### Initialization

```python
from nexrl.ray_resource_manager import RayResourceManager
from nexrl.nexrl_types import NexRLRole

resource_manager = RayResourceManager()
```

### Role Registration

Register roles with their classes and configurations.

```python
# Register rollout workers (standalone actors)
resource_manager.register_role(
    role=NexRLRole.ROLLOUT_WORKER,
    cls=SimpleRolloutWorker,
    config=config.rollout_worker,
    count=config.rollout_worker.num_workers,
    colocation_group=None  # Standalone
)

# Register trainer (colocated with trajectory pool)
resource_manager.register_role(
    role=NexRLRole.TRAINER,
    cls=SelfHostedGrpoTrainer,
    config=config.trainer,
    count=1,
    colocation_group="training_group"  # Colocated
)

# Register trajectory pool (colocated with trainer)
resource_manager.register_role(
    role=NexRLRole.TRAJECTORY_POOL,
    cls=TrajectoryPool,
    config=config.trajectory_pool,
    count=1,
    colocation_group="training_group"  # Same group
)
```

### Actor Creation

Create all registered actors.

```python
# Create actors based on registrations
resource_manager.create_all_actors()

# Access actor wrappers
workers = resource_manager.get_actor_wrapper(NexRLRole.ROLLOUT_WORKER)
trainers = resource_manager.get_actor_wrapper(NexRLRole.TRAINER)
pools = resource_manager.get_actor_wrapper(NexRLRole.TRAJECTORY_POOL)
```

## Environment Variables

Ray actors receive minimal environment variables to avoid HPC serialization issues.

### Required Variables

From `_get_minimal_env_vars()` in `ray_resource_manager.py`:

```python
env_vars = {
    "NEXRL_LAUNCH_MODE": "ray",
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
    "NEXRL_USER": os.environ.get("NEXRL_USER", ""),
    "EXPERIMENT_PATH": os.environ.get("EXPERIMENT_PATH", ""),
}
```

### Optional Variables

**Experiment Tracking:**
- `WANDB_HOST`, `WANDB_KEY` - WandB integration
- `SWANLAB_API_KEY`, `SWANLAB_LOG_DIR`, `SWANLAB_MODE` - SwanLab integration

**Distributed Training:**
- `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`
- `MASTER_ADDR`, `MASTER_PORT`
- `CUDA_VISIBLE_DEVICES`

**LLM Judge:**
- `LLM_JUDGE_URL`, `LLM_JUDGE_MODEL`

## Actor Options

### Standalone Actors

```python
ray_options = {
    "num_cpus": 1,
    "runtime_env": {"env_vars": env_vars},
    "max_concurrency": 100,  # High for workers
}
actor = ray_actor_cls.options(**ray_options).remote(config)
```

### Colocated Actors

```python
ray_options = {
    "num_cpus": 1,
    "runtime_env": {"env_vars": env_vars},
    "max_concurrency": 10,  # Lower for colocated
}
actor = ray_actor_cls.options(**ray_options).remote()
```

## Controller Integration

The controller creates Ray actors in distributed mode:

```python
def _initialize_ray_mode(self):
    """Initialize components in Ray mode."""
    # Create resource manager
    self._resource_manager = RayResourceManager()

    # Register all roles with colocation groups
    self._register_roles()

    # Create all actors
    self._resource_manager.create_all_actors()

    # Get actor references
    self._rollout_workers = self._resource_manager.get_actor_wrapper(
        NexRLRole.ROLLOUT_WORKER
    )
    self._trainer = self._resource_manager.get_actor_wrapper(
        NexRLRole.TRAINER
    )[0]
    # ... other components
```

## Best Practices

### 1. Minimize Data Transfer

Ray serializes all arguments and return values:

```python
# Bad: Transferring large data
result = execute(worker.process, large_dataset)

# Good: Use Ray object store
dataset_ref = ray.put(large_dataset)
result = execute(worker.process, dataset_ref)
```

### 2. Handle Timeouts

Ray operations have timeouts to prevent deadlocks:

```python
try:
    result = execute(actor.method, args)
except TimeoutError as e:
    logger.error(f"Operation timed out: {e}")
    # Handle timeout (retry, skip, fail)
```

### 3. Use Appropriate Concurrency

Match `max_concurrency` to workload:

```python
# High concurrency for I/O-bound workers
ray_options = {"max_concurrency": 100}

# Lower concurrency for compute-intensive actors
ray_options = {"max_concurrency": 10}
```

### 4. Monitor Actor Health

Check actor liveness regularly:

```python
# In controller monitoring loop
is_alive = self._activity_tracker.check_module_liveness(timeout=5.0)
if not is_alive:
    logger.error("Some actors are dead")
    # Handle failure
```

## Troubleshooting

### Actor Creation Failures

**Symptom:** Actors fail to initialize

**Solutions:**
- Check Ray cluster resources: `ray status`
- Verify environment variables are set
- Check actor logs for initialization errors
- Increase actor readiness timeout

### Communication Timeouts

**Symptom:** `GetTimeoutError` during execution

**Solutions:**
- Increase timeout in `execute()` calls
- Check for deadlocks in actor methods
- Verify actors are responsive via health checks
- Review actor concurrency limits

### Serialization Errors

**Symptom:** `SerializationError` or `PicklingError`

**Solutions:**
- Avoid passing non-serializable objects
- Use Ray object store for large data
- Check custom class serialization
- Minimize environment variable size

### Resource Contention

**Symptom:** Actors waiting for resources

**Solutions:**
- Review resource specifications
- Check cluster capacity: `ray status`
- Adjust actor placement constraints
- Use colocation to reduce resource usage

## Related Documentation

- [Colocation Patterns](./colocation.md) - Actor colocation strategies
- [Resource Allocation](./resource-allocation.md) - GPU/CPU allocation
- [Core Architecture](../02-core-architecture/overview.md) - System architecture
