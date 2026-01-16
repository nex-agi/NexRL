# Resource Allocation

Resource allocation in NexRL manages CPU and GPU resources across distributed components.

## Overview

NexRL uses Ray's resource management to:
- Allocate CPU/GPU to actors
- Control actor placement
- Manage resource pools
- Optimize resource utilization

## Ray Actor Options

### Basic Options

Actors are created with resource specifications:

```python
ray_options = {
    "num_cpus": 1,              # CPU cores
    "num_gpus": 0,              # GPU devices
    "runtime_env": {            # Environment configuration
        "env_vars": env_vars
    },
    "max_concurrency": 100,     # Max concurrent tasks
}

actor = ray_actor_cls.options(**ray_options).remote(config)
```

### Current Implementation

From `nexrl/ray_resource_manager.py`:

**Standalone Actors:**
```python
ray_options = {
    "num_cpus": 1,
    "runtime_env": {"env_vars": env_vars},
    "max_concurrency": 100,  # High for workers
}
```

**Colocated Actors:**
```python
ray_options = {
    "num_cpus": 1,
    "runtime_env": {"env_vars": env_vars},
    "max_concurrency": 10,   # Lower for shared actors
}
```

## Resource Requirements by Component

### Rollout Workers

**Profile:** CPU-bound, high concurrency

```python
ray_options = {
    "num_cpus": 1,
    "num_gpus": 0,              # CPU-only
    "max_concurrency": 100,     # Many concurrent requests
}
```

**Characteristics:**
- No GPU needed (inference service handles that)
- High I/O for LLM API calls
- Minimal memory per worker
- Scale horizontally

### Trainer (Self-Hosted)

**Profile:** GPU-bound, single-threaded

```python
ray_options = {
    "num_cpus": 4,
    "num_gpus": 1,              # Requires GPU
    "max_concurrency": 1,       # Sequential training
}
```

**Characteristics:**
- Requires GPU for model training
- High memory for model weights
- Compute-intensive operations
- Limited by GPU availability

### Trainer (Remote API)

**Profile:** CPU-bound, network-intensive

```python
ray_options = {
    "num_cpus": 1,
    "num_gpus": 0,              # API handles GPU
    "max_concurrency": 10,
}
```

**Characteristics:**
- No GPU needed (remote service handles it)
- Network I/O for API calls
- Minimal compute requirements
- Scales easily

### Trajectory Pool

**Profile:** Memory-bound, high throughput

```python
ray_options = {
    "num_cpus": 1,
    "num_gpus": 0,
    "max_concurrency": 100,     # Many workers putting trajectories
}
```

**Characteristics:**
- No GPU needed
- High memory for trajectory storage
- Frequent small operations
- Concurrency matches worker count

### Data Loader

**Profile:** I/O-bound, moderate CPU

```python
ray_options = {
    "num_cpus": 1,
    "num_gpus": 0,
    "max_concurrency": 10,
}
```

**Characteristics:**
- Disk I/O for data loading
- Minimal memory per item
- Preprocessing may need CPU
- Batch operations

### Weight Sync Controller

**Profile:** Network/I/O-bound

```python
ray_options = {
    "num_cpus": 1,
    "num_gpus": 0,
    "max_concurrency": 10,
}
```

**Characteristics:**
- Checkpoint I/O operations
- Network transfers to inference service
- Coordination logic
- Infrequent operations

### Validator

**Profile:** Memory-bound, periodic

```python
ray_options = {
    "num_cpus": 1,
    "num_gpus": 0,
    "max_concurrency": 10,
}
```

**Characteristics:**
- Memory for trajectory storage
- Metric computation (CPU)
- Infrequent operations
- Low resource usage

## Resource Planning

### Small-Scale Development

Single node with 1 GPU:

```python
# 1 GPU, 16 CPUs available
components = {
    "rollout_workers": 8,      # 8 CPUs
    "trainer": 1,              # 1 GPU + 4 CPUs
    "trajectory_pool": 1,      # 1 CPU (colocated with trainer)
    "data_loader": 1,          # 1 CPU
    "weight_sync": 1,          # 1 CPU (colocated with trainer)
    "validator": 1,            # 1 CPU
}
# Total: 16 CPUs, 1 GPU
```

### Medium-Scale Cluster

Multiple nodes with GPUs:

```python
# 4 nodes, each with 2 GPUs, 32 CPUs
components = {
    "rollout_workers": 64,     # 64 CPUs across nodes
    "trainers": 4,             # 4 GPUs + 16 CPUs
    "trajectory_pools": 4,     # Colocated with trainers
    "data_loaders": 4,         # 4 CPUs
    "weight_sync": 4,          # Colocated with trainers
    "validators": 4,           # 4 CPUs
}
# Total: ~88 CPUs, 4 GPUs
```

### Large-Scale Production

Dedicated GPU nodes for training, CPU nodes for rollout:

```python
# 8 GPU nodes (8 GPUs, 32 CPUs each) + 16 CPU nodes (32 CPUs each)
components = {
    "rollout_workers": 256,    # CPU nodes (256 CPUs)
    "trainers": 8,             # GPU nodes (8 GPUs + 32 CPUs)
    "trajectory_pools": 8,     # Colocated with trainers
    "data_loaders": 8,         # Distributed across GPU nodes
    "weight_sync": 8,          # Colocated with trainers
    "validators": 8,           # Distributed
}
# Total: ~288 CPUs (rollout) + 256 CPUs (GPU nodes), 8 GPUs
```

## Environment Variables

### Runtime Environment

Actors receive minimal environment variables:

```python
def _get_minimal_env_vars() -> dict[str, str]:
    """Get minimal required environment variables."""
    env_vars = {
        "NEXRL_LAUNCH_MODE": "ray",
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
        "NEXRL_USER": os.environ.get("NEXRL_USER", ""),
        "EXPERIMENT_PATH": os.environ.get("EXPERIMENT_PATH", ""),
    }

    # Add experiment tracking credentials
    if "WANDB_HOST" in os.environ:
        env_vars["WANDB_HOST"] = os.environ["WANDB_HOST"]
    if "WANDB_KEY" in os.environ:
        env_vars["WANDB_KEY"] = os.environ["WANDB_KEY"]

    # Add distributed training variables
    if "RANK" in os.environ:
        env_vars["RANK"] = os.environ["RANK"]
    if "WORLD_SIZE" in os.environ:
        env_vars["WORLD_SIZE"] = os.environ["WORLD_SIZE"]
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_vars["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

    return env_vars
```

**Rationale:** HPC environments can have huge environments (thousands of variables), causing Ray serialization timeouts. Minimal environment avoids this issue.

## Concurrency Management

### High Concurrency (Workers)

Rollout workers handle many concurrent tasks:

```python
ray_options = {
    "max_concurrency": 100,
}
```

**Use Cases:**
- Rollout workers (many data items processing)
- Trajectory pool (many workers putting trajectories)
- Stateless operations

### Low Concurrency (Trainers)

Training must be sequential:

```python
ray_options = {
    "max_concurrency": 1,
}
```

**Use Cases:**
- Trainers (sequential optimization steps)
- Single-threaded operations
- Stateful operations requiring synchronization

### Moderate Concurrency (Services)

Service operations can be partially parallel:

```python
ray_options = {
    "max_concurrency": 10,
}
```

**Use Cases:**
- Data loaders (batch operations)
- Colocated actors (shared resources)
- Weight sync controllers (coordinated operations)

## Resource Monitoring

### Check Ray Cluster Status

```bash
ray status
```

Output:
```
Resources
---------------------------------------------------------------
Usage:
 0.0/16.0 CPU
 0.0/1.0 GPU
 0B/50.00GiB memory
 0B/10.00GiB object_store_memory

Demands:
 (no resource demands)
```

### Monitor Actor Placement

```python
import ray

# List all actors
actors = ray.list_actors()
for actor in actors:
    print(f"Actor: {actor['name']}")
    print(f"  Resources: {actor['required_resources']}")
    print(f"  Node: {actor['node_id']}")
```

### Check Resource Availability

```python
import ray

# Get cluster resources
resources = ray.cluster_resources()
print("Available resources:", resources)

# Get available resources (not in use)
available = ray.available_resources()
print("Free resources:", available)
```

## Optimization Strategies

### 1. Colocation for Resource Efficiency

Combine components to reduce actor overhead:

```python
# Before: 3 actors
register_role(NexRLRole.TRAINER, count=1, colocation_group=None)
register_role(NexRLRole.TRAJECTORY_POOL, count=1, colocation_group=None)
register_role(NexRLRole.WEIGHT_SYNC_CONTROLLER, count=1, colocation_group=None)
# Resources: 3 CPUs

# After: 1 colocated actor
colocation_group = "training_group"
register_role(NexRLRole.TRAINER, count=1, colocation_group=colocation_group)
register_role(NexRLRole.TRAJECTORY_POOL, count=1, colocation_group=colocation_group)
register_role(NexRLRole.WEIGHT_SYNC_CONTROLLER, count=1, colocation_group=colocation_group)
# Resources: 1 CPU
```

### 2. Scale Rollout Workers

Adjust worker count based on throughput needs:

```python
# Compute-bound: Few workers
num_workers = 4

# I/O-bound (LLM API): Many workers
num_workers = 64

register_role(
    NexRLRole.ROLLOUT_WORKER,
    count=num_workers,
    colocation_group=None  # Standalone for parallelism
)
```

### 3. GPU Allocation for Self-Hosted Training

Dedicate GPU to training actor:

```python
# Modify _create_standalone_actors for GPU trainers
ray_options = {
    "num_cpus": 4,             # Multi-core for data processing
    "num_gpus": 1,             # Full GPU
    "max_concurrency": 1,      # Sequential training
}
```

### 4. Node Affinity

Place related actors on same node:

```python
# Not currently implemented in NexRL, but possible:
ray_options = {
    "scheduling_strategy": ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=specific_node_id,
        soft=False,  # Hard constraint
    )
}
```

## Best Practices

### 1. Profile Before Scaling

Understand resource usage before scaling:

```python
# Start small
num_workers = 4

# Monitor metrics
# - CPU utilization
# - Memory usage
# - Throughput (trajectories/sec)

# Scale based on bottlenecks
if cpu_util < 50%:
    num_workers = 16  # Scale up
```

### 2. Match Concurrency to Workload

Set concurrency based on operation characteristics:

```python
# I/O-bound: High concurrency
ray_options = {"max_concurrency": 100}

# CPU-bound: Moderate concurrency
ray_options = {"max_concurrency": 10}

# GPU-bound: Low concurrency
ray_options = {"max_concurrency": 1}
```

### 3. Monitor Resource Contention

Check for resource starvation:

```bash
# Check if actors are pending
ray status

# Look for:
# - "pending_actors" > 0
# - High resource utilization
# - Long actor creation times
```

### 4. Use Minimal Environment

Avoid environment variable bloat:

```python
# Good: Minimal environment
env_vars = _get_minimal_env_vars()

# Bad: Full environment (causes timeouts in HPC)
env_vars = os.environ.copy()  # Don't do this!
```

## Troubleshooting

### Actors Pending Creation

**Symptom:** Actors stuck in pending state

**Causes:**
- Insufficient cluster resources
- Resource specification too high
- Node failures

**Solutions:**
- Check cluster capacity: `ray status`
- Reduce resource requirements
- Scale cluster or reduce actor count

### Out of Memory Errors

**Symptom:** Actors crash with OOM errors

**Causes:**
- Trajectory pool accumulation
- Large model weights
- Memory leaks

**Solutions:**
- Reduce trajectory pool size
- Use gradient checkpointing
- Monitor memory usage
- Increase node memory

### GPU Not Utilized

**Symptom:** GPU shows 0% utilization during training

**Causes:**
- Trainer not requesting GPU
- Wrong CUDA_VISIBLE_DEVICES
- Model on CPU

**Solutions:**
- Add `num_gpus=1` to ray_options
- Verify CUDA_VISIBLE_DEVICES
- Check model device placement

### Concurrency Bottlenecks

**Symptom:** Low throughput despite many workers

**Causes:**
- Concurrency limit too low
- Shared resource contention
- Network bottlenecks

**Solutions:**
- Increase max_concurrency
- Profile actor utilization
- Check network bandwidth
- Consider colocation changes

## Future Enhancements

### Placement Groups

Support Ray placement groups for GPU gang scheduling:

```python
# Not yet implemented
pg = ray.util.placement_group([
    {"CPU": 4, "GPU": 1},  # Trainer
    {"CPU": 1},            # Trajectory pool
    {"CPU": 1},            # Weight sync
], strategy="STRICT_PACK")
```

### Dynamic Resource Allocation

Auto-scale workers based on load:

```python
# Not yet implemented
if trajectory_pool.is_full():
    scale_down_workers()
elif dataloader.has_pending_items():
    scale_up_workers()
```

### Resource Pools

Support multiple resource pools for different components:

```python
# Not yet implemented
resource_pools = {
    "cpu_pool": {"num_cpus": 64},      # Rollout workers
    "gpu_pool": {"num_gpus": 4},       # Trainers
    "storage_pool": {"num_cpus": 8},   # Data loaders
}
```

## Related Documentation

- [Ray Integration](./ray-integration.md) - Ray execution modes
- [Colocation](./colocation.md) - Actor colocation patterns
- [Controller](../02-core-architecture/controller.md) - Component initialization
