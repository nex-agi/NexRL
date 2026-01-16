# Trajectory Pool

The Trajectory Pool manages collection, grouping, and batching of trajectories from multiple rollout workers. It provides flexible strategies for organizing trajectories before training.

## Overview

**Purpose**: Collect trajectories from rollout workers, group them appropriately, and provide batches to the trainer

**Key Features**:
- Multi-store architecture (one store per model tag)
- Flexible grouping strategies (simple, grouped, hierarchical)
- Configurable batch readiness criteria
- Weight synchronization coordination
- Thread-safe operation

## TrajectoryPool

**Location**: `nexrl/trajectory_pool.py`

**Main interface** for trajectory management.

### Constructor

```python
def __init__(self, config: DictConfig)
```

**Configuration**:

```yaml
trajectory_pool:
  type: "default"
  batch_size: 32  # Default batch size

  # Grouping configuration (optional)
  group_size: 4  # Trajectories per group
  key_list: ["query"]  # Keys for grouping

  # Batch readiness criteria
  check_batch_ready_function: "batch_size"  # or "loaded_batch_finished"
```

### Core Methods

#### put_trajectory(trajectory: Trajectory) -> str

Adds a trajectory to the appropriate store.

```python
result = trajectory_pool.put_trajectory(trajectory)
```

**Parameters**:
- `trajectory`: Trajectory data to store

**Returns**:
- `"success"`: Trajectory stored successfully
- `"fail"`: Failed to store trajectory
- `"re-rollout"`: Weight sync in progress, should retry

**Process**:
1. Extract `ModelTag` from trajectory (defaults to "default")
2. Create or retrieve appropriate `TrajectoryPoolInstance`
3. Add trajectory to instance (may block during weight sync)

#### get_batch(batch_size: int | None = None, model_tag: ModelTag | None = None) -> Batch | None

Retrieves a batch of trajectories.

```python
batch = trajectory_pool.get_batch(batch_size=32, model_tag="policy")
```

**Parameters**:
- `batch_size`: Number of trajectories to retrieve (defaults to config.batch_size)
- `model_tag`: Specific model to get batch from (None = any model)

**Returns**: `Batch` object or `None` if insufficient samples

**Behavior**:
- If `model_tag=None`, tries any available store
- If specified model has no store, returns None
- Only returns batch when enough samples available

#### get_batch_any(batch_size: int | None = None) -> Batch | None

Gets batch from any store with sufficient samples.

```python
batch = trajectory_pool.get_batch_any(batch_size=32)
```

**Parameters**:
- `batch_size`: Number of trajectories to retrieve

**Returns**: Batch from any store, or None

#### is_empty(model_tag: ModelTag | None = None) -> bool

Checks if trajectory pool is empty.

```python
if trajectory_pool.is_empty():
    print("No trajectories available")
```

**Parameters**:
- `model_tag`: Specific model to check (None = all models)

**Returns**: `True` if specified store (or all stores) is empty

#### get_model_tags() -> list[ModelTag]

Gets all active model tags.

```python
tags = trajectory_pool.get_model_tags()
# ["policy", "reference"]
```

**Returns**: List of model tags with active stores

## TrajectoryPoolInstance

**Purpose**: Individual pool instance managing trajectories for a single model

**Location**: `nexrl/trajectory_pool.py`

### Setup

#### set_module_references(dataloader, weight_sync_controller, activity_tracker)

Sets references to other modules.

```python
instance.set_module_references(
    dataloader=dataloader,
    weight_sync_controller=weight_sync_controller,
    activity_tracker=activity_tracker
)
```

### Trajectory Management

#### put_trajectory(trajectory: Trajectory) -> str

Adds trajectory to this instance's store.

**Returns**:
- `"success"`: Added successfully
- `"fail"`: Failed to add
- `"re-rollout"`: Weight sync in progress, retry needed

#### notify_weight_sync_starting()

Blocks new trajectory additions during weight synchronization.

```python
instance.notify_weight_sync_starting()
# New trajectories will return "re-rollout"
```

#### unlock_for_weight_sync()

Unblocks trajectory additions after weight sync completes.

```python
instance.unlock_for_weight_sync()
# New trajectories can be added again
```

## Trajectory Store Types

The pool automatically creates appropriate stores based on configuration:

### SimpleTrajectoryStore

**Use Case**: No grouping required

**Behavior**: Directly adds trajectories to finished samples

**Configuration**:
```yaml
trajectory_pool:
  # No key_list specified = SimpleTrajectoryStore
  batch_size: 32
```

**When to Use**:
- Tasks with independent samples
- No need for trajectory grouping
- Simplest and fastest option

### GroupedTrajectoryStore

**Use Case**: Single-level grouping (e.g., by user ID or query)

**Behavior**: Groups trajectories by specified key, releases when group reaches target size

**Configuration**:
```yaml
trajectory_pool:
  key_list: ["query"]  # Single key = GroupedTrajectoryStore
  group_size: 4  # 4 trajectories per group
  batch_size: 32
```

**Example**: GRPO grouping
```
Query A: [traj1, traj2, traj3, traj4] → Released as group
Query B: [traj1, traj2, traj3, traj4] → Released as group
```

**When to Use**:
- GRPO algorithm (group by query/prompt)
- Multi-sample per query tasks
- Need to compare samples from same context

### HierarchicalTrajectoryStore

**Use Case**: Multi-level grouping (e.g., by user ID then session ID)

**Behavior**: Creates nested hierarchy, releases leaf groups when complete

**Configuration**:
```yaml
trajectory_pool:
  key_list: ["user_id", "session_id"]  # Multiple keys = HierarchicalTrajectoryStore
  group_size: 4  # Size of leaf groups
  batch_size: 32
```

**Example**: User sessions
```
User A:
  Session 1: [traj1, traj2, traj3, traj4] → Released
  Session 2: [traj1, traj2, traj3, traj4] → Released
User B:
  Session 1: [traj1, traj2, traj3, traj4] → Released
```

**When to Use**:
- Hierarchical data structure
- Need to group at multiple levels
- Complex grouping requirements

## Batch Readiness Criteria

### "batch_size" Mode

Batch ready when exact batch_size samples accumulated.

```yaml
trajectory_pool:
  check_batch_ready_function: "batch_size"
  batch_size: 32
```

**Behavior**: Waits until exactly 32 trajectories available

**Use When**: You want consistent batch sizes

### "loaded_batch_finished" Mode

Batch ready when dataloader finishes and all workers idle.

```yaml
trajectory_pool:
  check_batch_ready_function: "loaded_batch_finished"
  batch_size: 32
```

**Behavior**: May return batches smaller than batch_size at end of data

**Use When**: You want to process all data without leftover samples

## Weight Synchronization Coordination

The trajectory pool coordinates with weight synchronization:

### Blocking Flow

```
1. Trainer completes training step
2. Trainer notifies WeightSync
3. WeightSync calls notify_weight_sync_starting()
4. TrajectoryPool blocks new trajectories
5. Workers get "re-rollout" status
6. WeightSync performs sync
7. WeightSync calls unlock_for_weight_sync()
8. TrajectoryPool accepts trajectories again
```

### Worker Perspective

```python
def step(self, task):
    # Generate trajectory
    trajectory = create_trajectory(task)

    # Try to submit
    result = self._put_trajectory(trajectory)

    if result == "re-rollout":
        # Weight sync in progress
        # Return task to dataloader
        self._dataloader.add_item_front(task)
        return "re-rollout"

    return result
```

## Usage Patterns

### Basic Usage in Trainer

```python
class MyTrainer(BaseTrainer):
    def train_step(self):
        # Get batch
        batch = execute(self.trajectory_pool.get_batch, batch_size=32)

        if batch is None:
            # Not enough samples yet
            return None

        # Train on batch
        metrics = self._train_on_batch(batch)

        return metrics
```

### GRPO Grouping

```yaml
# Group 4 samples per query for GRPO
trajectory_pool:
  key_list: ["run_id"]  # GRPO uses run_id
  group_size: 4
  batch_size: 32  # Total batch = 32 trajectories (8 groups of 4)
```

```python
# Rollout worker sets run_id
trajectory = {
    "tokens": ...,
    "run_id": query_hash,  # Same for all samples from one query
    # ...
}
```

### Multiple Models

```python
# Separate stores for policy and reference models
trajectory_policy = {
    "tokens": ...,
    "model_tag": "policy",
}

trajectory_reference = {
    "tokens": ...,
    "model_tag": "reference",
}

# Get batches per model
policy_batch = pool.get_batch(model_tag="policy")
reference_batch = pool.get_batch(model_tag="reference")
```

## Configuration Examples

### Simple (No Grouping)

```yaml
trajectory_pool:
  type: "default"
  batch_size: 32
  check_batch_ready_function: "batch_size"
```

### GRPO (Single-Level Grouping)

```yaml
trajectory_pool:
  type: "default"
  batch_size: 32
  group_size: 4  # 4 samples per query
  key_list: ["run_id"]  # Group by run_id
  check_batch_ready_function: "batch_size"
```

### Hierarchical (Multi-Level Grouping)

```yaml
trajectory_pool:
  type: "default"
  batch_size: 32
  group_size: 4  # Leaf group size
  key_list: ["user_id", "session_id"]  # Two-level hierarchy
  check_batch_ready_function: "batch_size"
```

### Variable Batch Size

```yaml
trajectory_pool:
  type: "default"
  batch_size: 32  # Target size
  check_batch_ready_function: "loaded_batch_finished"  # Allow smaller batches
```

## Best Practices

### Grouping

1. **Use grouping for GRPO** - Essential for proper advantage computation
2. **Set appropriate group_size** - Balance statistical significance vs. throughput
3. **Use run_id consistently** - Same run_id for samples from same query
4. **Test grouping logic** - Verify trajectories grouped correctly

### Batch Sizes

1. **Balance memory and throughput** - Larger batches = better GPU utilization but more memory
2. **Consider group_size** - batch_size should be divisible by group_size for GRPO
3. **Use "loaded_batch_finished"** - To avoid wasting data at end of epoch
4. **Monitor batch timing** - Adjust if trainer waiting too long for batches

### Performance

1. **Minimize lock contention** - Trajectory stores use fine-grained locking
2. **Pre-group in workers** - Set run_id/group keys before submission
3. **Monitor pool size** - Don't let pools grow unbounded
4. **Use appropriate store type** - SimpleTrajectoryStore is fastest when grouping not needed

### Debugging

1. **Log trajectory counts** - Monitor incoming and outgoing trajectories
2. **Check grouping keys** - Verify trajectories have required keys
3. **Monitor batch readiness** - Check if batches forming as expected
4. **Inspect trajectories** - Look at first few trajectories to verify structure

## Troubleshooting

### Issue: "Trainer not receiving batches"

**Symptoms**: Trainer idle, trajectories accumulating

**Solutions**:
1. Check `batch_size` configuration matches expected throughput
2. Verify `group_size` allows complete groups
3. Check `key_list` keys exist in trajectories
4. Monitor trajectory count in pool

### Issue: "Incomplete groups"

**Symptoms**: Some groups never complete, trajectories stuck

**Solutions**:
1. Verify all trajectories have grouping keys
2. Check for typos in group key names
3. Ensure group_size is achievable
4. Use "loaded_batch_finished" to flush incomplete groups

### Issue: "Weight sync blocking too long"

**Symptoms**: Many "re-rollout" events, slow training

**Solutions**:
1. Reduce weight sync frequency
2. Use async weight sync mode
3. Increase staleness threshold
4. Optimize sync operation speed

## Next Steps

- Learn about [Trainers](../06-trainers/overview.md) that consume batches
- Understand [Weight Synchronization](../08-features/weight-synchronization.md)
- Review [GRPO Algorithm](../07-algorithms/grpo.md) for grouping details
- Explore [Configuration Reference](../12-configuration-reference/complete-config.md)
