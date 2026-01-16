# Validation

## Overview

`Validator` collects validation trajectories and computes metrics for model evaluation. Unlike training which requires batching, validation simply collects all trajectories and aggregates metrics.

**Location**: `nexrl/validator.py`

## Key Concepts

### Validation Cycle

```
Weight Sync → Unlock Validate DataLoader → Rollout Workers Generate → Collect Trajectories → Compute Metrics
```

Validation runs at regular intervals controlled by `validate_freq` in `WeightSyncController`:

```yaml
weight_sync_controller:
  validate_freq: 5  # Run validation every 5 training steps
```

### Completion Criteria

Validation is complete when:
1. Validate dataloader is finished (all data consumed)
2. All rollout workers are quiescent (no active work)

## Core Methods

### Constructor

```python
def __init__(self, config: DictConfig)
```

Initializes the validator with configuration.

**Parameters**:
- `config`: Configuration dictionary

### set_module_references

```python
def set_module_references(self, validate_dataloader: BaseDataLoader)
```

Sets reference to the validation dataloader.

**Parameters**:
- `validate_dataloader`: DataLoader providing validation data

### put_trajectory

```python
def put_trajectory(self, trajectory: Trajectory) -> str
```

Store validation trajectory from rollout workers.

**Parameters**:
- `trajectory`: Validation trajectory to store

**Returns**: `"success"` (matches TrajectoryPool interface)

### is_complete

```python
def is_complete(self) -> bool
```

Check if all validation trajectories have been collected.

**Returns**: `True` if validation complete

**Logic**:
```python
return (
    execute(self._validate_dataloader.is_finished)
    and self._activity_tracker.is_rollout_worker_quiescent()
)
```

### compute_and_log_metrics

```python
def compute_and_log_metrics(self) -> dict[str, float]
```

Compute validation metrics and log via activity tracker.

**Returns**: Dictionary of computed metrics with `"val/"` prefix

**Process**:
1. Collect score values from all trajectories
2. Group by score key and data source
3. Compute mean for each key
4. Add `"val/"` prefix to metric names
5. Log to experiment tracking (e.g., wandb)
6. Clear collected trajectories

**Metric Keys**:
- `val/{score_key}`: Mean of score across all trajectories
- `val/{data_source}_{score_key}`: Mean per data source
- `val/num_samples`: Number of validation samples

## Trajectory Score Structure

Validation trajectories should include a `score` dictionary:

```python
trajectory = {
    "prompt": "...",
    "response": "...",
    "data_source": "dataset_name",
    "score": {
        "reward": 0.85,
        "accuracy": 1.0,
        "custom_metric": 0.9
    }
}
```

## Validation Workflow

### 1. Controller Triggers Validation

When `rollout_model_version % validate_freq == 0`:

```python
# In WeightSyncController.train_worker_notify_weight_update
if self._validate_freq > 0 and rollout_model_version % self._validate_freq == 0:
    self._waiting_for_validation = True
    # Keep locks in place
```

### 2. Controller Runs Validation

```python
# In NexRLController._run_validate
def _run_validate(self):
    # Unlock validate dataloader
    execute(self._validate_dataloader.unlock_for_weight_sync)

    # Wait for completion
    while not execute(self._validator.is_complete):
        time.sleep(1)

    # Compute metrics
    metrics = execute(self._validator.compute_and_log_metrics)

    # Signal completion
    execute(self._weight_sync_controller.end_validate, model_tag)
```

### 3. Rollout Workers Process Validation Data

Rollout workers generate trajectories using the validate dataloader:

```python
# In RolloutWorker
if not self._is_validation:
    # Training rollout
    target_pool = self._trajectory_pool
else:
    # Validation rollout
    target_pool = self._validator

execute(target_pool.put_trajectory, trajectory)
```

### 4. Validator Computes Metrics

After all trajectories collected:

```python
metrics = execute(self._validator.compute_and_log_metrics)
# Example output:
# {
#   "val/reward": 0.85,
#   "val/accuracy": 0.92,
#   "val/dataset1_reward": 0.83,
#   "val/dataset2_reward": 0.87,
#   "val/num_samples": 100
# }
```

## Configuration

### Validator Config

```yaml
validator:
  type: "validator"
```

The validator itself has minimal configuration. Validation behavior is controlled by:

### Validation DataLoader

```yaml
validate_dataloader:
  type: "torch"
  data_config:
    dataset_paths:
      - /path/to/validation_data.jsonl
    split_type: "none"  # Use full dataset
  shuffle: false
  batch_size: 1
```

### Validation Frequency

```yaml
weight_sync_controller:
  validate_freq: 5  # Every 5 steps (0 = disabled)
```

## Validation vs Training Dataloaders

NexRL uses separate dataloaders for training and validation:

```python
# In NexRLController
self._dataloader = BaseDataLoader(config.dataloader)  # Training data
self._validate_dataloader = BaseDataLoader(config.validate_dataloader)  # Validation data
```

**Key Differences**:
- Training dataloader: Cycles indefinitely, shuffles data
- Validation dataloader: Single pass, no shuffle, unlocked only during validation

## Integration Example

### Recipe Configuration

```yaml
# Training data
dataloader:
  type: "torch"
  data_config:
    dataset_paths:
      - /data/train.jsonl
    split_type: "train"
  shuffle: true
  batch_size: 1

# Validation data
validate_dataloader:
  type: "torch"
  data_config:
    dataset_paths:
      - /data/validation.jsonl
    split_type: "none"
  shuffle: false
  batch_size: 1

# Validation frequency
weight_sync_controller:
  validate_freq: 10

# Validator
validator:
  type: "validator"
```

### Custom Evaluator with Validation Scores

```python
from nexrl.rollout_worker import Evaluator, EvaluationRunResult

class MyEvaluator(Evaluator):
    def evaluate(self, data, evaluation_target):
        # Compute multiple metrics
        reward = self._compute_reward(data, evaluation_target)
        accuracy = self._compute_accuracy(data, evaluation_target)

        return EvaluationRunResult(
            reward=reward,
            ground_truth=data.get("answer", ""),
            metrics={
                "accuracy": accuracy,
                "length": len(evaluation_target.final_answer)
            }
        )
```

These metrics will appear in validation as:
- `val/reward`: Mean reward
- `val/accuracy`: Mean accuracy
- `val/length`: Mean answer length

## Related Documentation

- [Core Architecture](../02-core-architecture/overview.md) - System overview
- [Weight Synchronization](./weight-synchronization.md) - Weight sync coordination
- [Activity Tracking](../02-core-architecture/activity-tracking.md) - Logging and monitoring
- [Data Loader](../03-data-loader/data-loader.md) - Data loading system
- [Evaluators](../05-rollout-workers/evaluators.md) - Evaluation patterns
