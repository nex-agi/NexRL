# Actor Colocation

Actor colocation allows multiple NexRL modules to share a single Ray actor, reducing resource usage and communication overhead.

## Overview

**Colocation** groups multiple modules into a single Ray actor, enabling:
- Reduced resource consumption (fewer actors)
- Lower communication latency (in-process calls)
- Shared state and memory
- Coordinated lifecycle management

## Colocation Groups

### Defining Groups

Modules with the same `colocation_group` name share an actor.

```python
# Trainer and trajectory pool share an actor
resource_manager.register_role(
    role=NexRLRole.TRAINER,
    cls=SelfHostedGrpoTrainer,
    config=config.trainer,
    count=1,
    colocation_group="training_group"
)

resource_manager.register_role(
    role=NexRLRole.TRAJECTORY_POOL,
    cls=TrajectoryPool,
    config=config.trajectory_pool,
    count=1,
    colocation_group="training_group"  # Same group
)

# Data loader runs standalone
resource_manager.register_role(
    role=NexRLRole.DATA_LOADER,
    cls=TorchDataLoader,
    config=config.data,
    count=1,
    colocation_group=None  # Standalone
)
```

### Requirements

All roles in a colocation group must have the **same count**:

```python
# Valid: Both have count=2
register_role(role_a, count=2, colocation_group="group1")
register_role(role_b, count=2, colocation_group="group1")

# Invalid: Different counts
register_role(role_a, count=1, colocation_group="group1")  # Error!
register_role(role_b, count=2, colocation_group="group1")
```

## RayActorWrapper

The wrapper provides transparent access to colocated modules.

### Architecture

When modules are colocated, their methods are prefixed with the role name:

```python
# Original methods
trainer.train(trajectories)
pool.get_batch(batch_size)

# Colocated actor methods
actor.trainer_train(trajectories)
actor.trajectory_pool_get_batch(batch_size)

# Wrapper provides clean interface
trainer_wrapper.train(trajectories)      # Calls actor.trainer_train()
pool_wrapper.get_batch(batch_size)       # Calls actor.trajectory_pool_get_batch()
```

### Implementation

From `nexrl/ray_resource_manager.py`:

```python
class RayActorWrapper:
    """
    Wrapper for Ray actors supporting elegant colocation.

    Automatically handles method name prefixing for colocated actors.
    """

    def __init__(
        self,
        actor: ActorHandle,
        actor_class: type,
        role: NexRLRole,
        is_colocated: bool = True
    ):
        self._actor = actor
        self._actor_class = actor_class
        self._role = role
        self._role_prefix = role.value
        self._is_colocated = is_colocated

        # Rebind public methods
        self._rebind_public_methods()

    def _rebind_public_methods(self):
        """Bind methods with role prefix removed."""
        if self._is_colocated:
            # Find methods with role prefix
            for attr_name in dir(self._actor_class):
                if attr_name.startswith(self._role_prefix + "_"):
                    # Remove prefix
                    role_method_name = attr_name.replace(
                        self._role_prefix + "_", ""
                    )
                    # Bind actor method to wrapper
                    actor_method = getattr(self._actor, attr_name)
                    setattr(self, role_method_name, actor_method)
        else:
            # Standalone: bind all public methods as-is
            for attr_name in dir(self._actor_class):
                if not attr_name.startswith("_"):
                    actor_method = getattr(self._actor, attr_name)
                    setattr(self, attr_name, actor_method)
```

### Usage Example

```python
# Get wrappers from resource manager
trainers = resource_manager.get_actor_wrapper(NexRLRole.TRAINER)
pools = resource_manager.get_actor_wrapper(NexRLRole.TRAJECTORY_POOL)

trainer_wrapper = trainers[0]
pool_wrapper = pools[0]

# Use like local objects
from nexrl.executor import execute

batch = execute(pool_wrapper.get_batch, batch_size=32)
metrics = execute(trainer_wrapper.train, trajectories)
```

## Colocated Class Creation

The `create_colocated_class()` function dynamically creates a class containing multiple modules.

### Implementation

From `ray_resource_manager.py`:

```python
def create_colocated_class(class_dict: dict[str, ClassWithInitArgs]) -> type:
    """
    Create a class that contains multiple module instances.

    Args:
        class_dict: {role_name: ClassWithInitArgs(cls, config)}

    Returns:
        Colocated class with prefixed methods
    """
    # Store module instances
    module_instances = {}

    def __init__(self):
        """Initialize all modules."""
        for role_name, class_with_args in class_dict.items():
            cls = class_with_args.cls
            args = class_with_args.args
            kwargs = class_with_args.kwargs

            # Create instance
            instance = cls(*args, **kwargs)
            module_instances[role_name] = instance

    # Create method wrappers with role prefix
    methods = {"__init__": __init__}

    for role_name, class_with_args in class_dict.items():
        cls = class_with_args.cls

        # Wrap each public method
        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue

            attr = getattr(cls, attr_name)
            if not callable(attr):
                continue

            # Create prefixed method
            prefixed_name = f"{role_name}_{attr_name}"

            def make_method(role, method_name):
                def method(self, *args, **kwargs):
                    instance = module_instances[role]
                    return getattr(instance, method_name)(*args, **kwargs)
                return method

            methods[prefixed_name] = make_method(role_name, attr_name)

    # Create and return the class
    return type("ColocatedClass", (), methods)
```

### Example Result

Given roles `trainer` and `trajectory_pool`:

```python
class ColocatedClass:
    def __init__(self):
        self._trainer = SelfHostedGrpoTrainer(config.trainer)
        self._trajectory_pool = TrajectoryPool(config.trajectory_pool)

    # Trainer methods (prefixed)
    def trainer_train(self, *args, **kwargs):
        return self._trainer.train(*args, **kwargs)

    def trainer_run(self, *args, **kwargs):
        return self._trainer.run(*args, **kwargs)

    # TrajectoryPool methods (prefixed)
    def trajectory_pool_get_batch(self, *args, **kwargs):
        return self._trajectory_pool.get_batch(*args, **kwargs)

    def trajectory_pool_put_trajectory(self, *args, **kwargs):
        return self._trajectory_pool.put_trajectory(*args, **kwargs)
```

## Common Colocation Patterns

### Pattern 1: Training Group

Colocate trainer with trajectory pool for fast batch access.

```python
# Trainer needs frequent access to trajectory pool
colocation_group = "training_group"

register_role(NexRLRole.TRAINER, SelfHostedGrpoTrainer,
              count=1, colocation_group=colocation_group)
register_role(NexRLRole.TRAJECTORY_POOL, TrajectoryPool,
              count=1, colocation_group=colocation_group)
```

**Benefits:**
- No serialization for batch retrieval
- Shared memory for trajectories
- Atomic batch operations

### Pattern 2: Data Management Group

Colocate data loader with validator for shared data state.

```python
colocation_group = "data_group"

register_role(NexRLRole.DATA_LOADER, TorchDataLoader,
              count=1, colocation_group=colocation_group)
register_role(NexRLRole.VALIDATE_DATALOADER, TorchDataLoader,
              count=1, colocation_group=colocation_group)
register_role(NexRLRole.VALIDATOR, Validator,
              count=1, colocation_group=colocation_group)
```

**Benefits:**
- Shared data loading infrastructure
- Coordinated validation cycles
- Reduced memory overhead

### Pattern 3: Standalone Workers

Keep rollout workers standalone for parallel execution.

```python
# Each worker runs independently
register_role(NexRLRole.ROLLOUT_WORKER, SimpleRolloutWorker,
              count=16, colocation_group=None)
```

**Benefits:**
- Maximum parallelism
- Independent failure isolation
- Flexible resource allocation

### Pattern 4: Weight Sync Group

Colocate weight sync controller with trainer.

```python
colocation_group = "training_group"

register_role(NexRLRole.TRAINER, SelfHostedGrpoTrainer,
              count=1, colocation_group=colocation_group)
register_role(NexRLRole.WEIGHT_SYNC_CONTROLLER, WeightSyncController,
              count=1, colocation_group=colocation_group)
register_role(NexRLRole.TRAJECTORY_POOL, TrajectoryPool,
              count=1, colocation_group=colocation_group)
```

**Benefits:**
- Direct weight access
- Atomic checkpoint operations
- Coordinated synchronization

## Choosing Colocation Strategy

### Colocate When:

1. **High Communication Frequency**
   - Components call each other frequently
   - Example: Trainer ↔ TrajectoryPool

2. **Shared State**
   - Components need shared memory access
   - Example: DataLoader ↔ Validator

3. **Coordination Required**
   - Components must coordinate actions
   - Example: WeightSync ↔ Trainer

4. **Resource Conservation**
   - Limited actor slots available
   - Multiple lightweight components

### Keep Standalone When:

1. **Parallel Execution**
   - Need multiple instances running simultaneously
   - Example: Rollout workers

2. **Independent Failure**
   - Failures should not affect other components
   - Example: Separate experiment loggers

3. **Different Resource Needs**
   - Components require different GPU/CPU allocation
   - Example: GPU trainer vs CPU data loader

4. **Load Balancing**
   - Need to distribute work across nodes
   - Example: Distributed rollout workers

## Debugging Colocated Actors

### Check Method Names

List available methods on colocated actor:

```python
actor = resource_manager._colocation_group_actors["training_group"][0]
methods = [m for m in dir(actor) if not m.startswith("_")]
print("Available methods:", methods)
# ['trainer_train', 'trainer_run', 'trajectory_pool_get_batch', ...]
```

### Verify Wrapper Binding

Check wrapper has correct methods:

```python
trainer_wrapper = resource_manager.get_actor_wrapper(NexRLRole.TRAINER)[0]
print("Wrapper methods:", [m for m in dir(trainer_wrapper) if not m.startswith("_")])
# ['train', 'run', 'stop', ...]  # Prefixes removed
```

### Test Actor Health

Health check colocated actors:

```python
for role in [NexRLRole.TRAINER, NexRLRole.TRAJECTORY_POOL]:
    wrappers = resource_manager.get_actor_wrapper(role)
    for i, wrapper in enumerate(wrappers):
        try:
            result = execute(wrapper.health_check)
            print(f"{role.value}[{i}]: {'healthy' if result else 'unhealthy'}")
        except Exception as e:
            print(f"{role.value}[{i}]: error - {e}")
```

## Best Practices

### 1. Group by Access Patterns

Colocate components that frequently communicate:

```python
# Good: Trainer accesses pool frequently
colocation_group = "training_group"

# Bad: Workers rarely interact with pool directly
# (they go through trajectory_pool.put_trajectory)
```

### 2. Respect Resource Boundaries

Don't colocate GPU and CPU-only components:

```python
# Bad: GPU trainer with CPU-only data loader
colocation_group = "mixed_group"  # Wastes GPU resources

# Good: Keep separate
trainer_group = "gpu_group"
data_group = "cpu_group"
```

### 3. Monitor Concurrency

Colocated actors share concurrency limits:

```python
# Lower concurrency for colocated actors
ray_options = {
    "max_concurrency": 10  # Shared across all colocated modules
}

# Higher for standalone
ray_options = {
    "max_concurrency": 100  # Dedicated to single module
}
```

### 4. Test Colocation Changes

Validate colocation changes don't introduce deadlocks:

```python
# Before: Standalone (works)
register_role(role_a, colocation_group=None)
register_role(role_b, colocation_group=None)

# After: Colocated (test for deadlocks!)
register_role(role_a, colocation_group="group1")
register_role(role_b, colocation_group="group1")
```

## Troubleshooting

### Method Not Found

**Symptom:** `AttributeError: 'RayActorWrapper' has no attribute 'method_name'`

**Causes:**
- Method prefix mismatch
- Method not public
- Class not registered correctly

**Solutions:**
- Check method exists in original class
- Verify method is public (no leading `_`)
- Print available methods on wrapper

### Colocation Count Mismatch

**Symptom:** `ValueError: All roles in colocation group must have same count`

**Solution:** Ensure all roles in group have identical `count`:

```python
# Fix count mismatch
register_role(role_a, count=1, colocation_group="group1")
register_role(role_b, count=1, colocation_group="group1")  # Must be 1
```

### Actor Initialization Failure

**Symptom:** Actor fails to initialize with colocated modules

**Causes:**
- Module initialization order dependencies
- Shared resource conflicts
- Configuration incompatibilities

**Solutions:**
- Check module `__init__` for dependencies
- Review resource requirements
- Verify configurations are compatible

## Related Documentation

- [Ray Integration](./ray-integration.md) - Ray execution modes
- [Resource Allocation](./resource-allocation.md) - GPU/CPU allocation
- [Controller](../02-core-architecture/controller.md) - Module initialization
