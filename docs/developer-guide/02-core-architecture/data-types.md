# Core Data Types

This document describes the fundamental data types used throughout the NexRL framework.

## Overview

NexRL defines several core data types in `nexrl/nexrl_types.py` that form the foundation of the framework's data flow and component interaction.

## Type Aliases

### ModelTag

```python
ModelTag = str  # Type alias for model identification
```

**Purpose**: Identifies different models within the system

**Usage**:
```python
model_tag = "policy"  # Main training model
model_tag = "reference"  # Reference model for KL penalty
```

**Common Values**:
- `"policy"`: Main trainable model
- `"reference"`: Reference model (frozen)
- `"default"`: Default model tag when not specified

## Data Classes

### Trajectory

```python
@dataclass
class Trajectory:
    tokens: list[int]                    # Token IDs for full sequence
    loss_mask: list[int]                 # 0/1 mask for loss computation
    reward: float                        # Reward value for this trajectory
    is_val: bool                         # Validation flag
    extra_fields: dict[str, Any]         # Additional fields
```

**Purpose**: Represents a single rollout episode with all training data

**Required Fields**:
- `tokens`: Complete token sequence (prompt + response)
- `loss_mask`: Binary mask indicating which tokens to train on
- `reward`: Scalar reward from evaluator
- `is_val`: Whether this is a validation trajectory

**Extra Fields** (common):
- `model_tag`: Model used for generation
- `temperature`: Sampling temperature
- `uid`: User/group identifier
- `run_id`: Run identifier (for grouping)
- `query`: Original input query
- `final_answer`: Agent's final output
- `ground_truth`: Reference answer
- `metrics`: Additional evaluation metrics

**Dictionary-Like Interface**:

```python
# Access fields
tokens = trajectory["tokens"]
reward = trajectory.get("reward", 0.0)

# Set fields
trajectory["custom_field"] = value

# Check existence
if "run_id" in trajectory:
    run_id = trajectory["run_id"]

# Iterate
for key, value in trajectory.items():
    print(f"{key}: {value}")
```

**Example**:

```python
from nexrl.nexrl_types import Trajectory

trajectory = Trajectory(
    tokens=[1, 2, 3, 4, 5],
    loss_mask=[0, 0, 1, 1, 1],  # Only train on last 3 tokens
    reward=1.0,
    is_val=False,
    extra_fields={
        "model_tag": "policy",
        "temperature": 0.7,
        "query": "What is 2+2?",
        "final_answer": "4",
    }
)

# Access via dict-like interface
print(trajectory["tokens"])  # [1, 2, 3, 4, 5]
print(trajectory["query"])   # "What is 2+2?"
```

### Batch

```python
@dataclass
class Batch:
    values: dict[str, Any]      # Tensor or data arrays
    metadata: dict[str, Any]    # Batch metadata
```

**Purpose**: Represents a batch of trajectories for training

**Structure**:
- `values`: Dictionary mapping field names to batched data
  - Tensors: 2D/3D tensors for token sequences
  - Lists: Non-tensor data (strings, dicts, etc.)
- `metadata`: Batch-level information
  - `batch_size`: Number of trajectories in batch
  - `model_tag`: Model tag for this batch
  - `temperature`: Sampling temperature (if consistent)

**Key Methods**:

#### `__len__() -> int`

Returns the batch size:

```python
batch_size = len(batch)
```

#### `copy() -> Batch`

Creates a deep copy:

```python
batch_copy = batch.copy()
```

#### `to_dict() -> dict[str, Any]`

Merges values and metadata (metadata overwrites values):

```python
combined = batch.to_dict()
# Access both values and metadata in one dict
```

#### `from_trajectories(trajectories, model_tag=None) -> Batch`

Creates a batch from a list of trajectories:

```python
trajectories = [traj1, traj2, traj3]
batch = Batch.from_trajectories(trajectories, model_tag="policy")
```

**Behavior**:
- Stacks 1D tensors into 2D tensors
- Collects non-tensors into lists
- Validates all trajectories have same keys
- Extracts metadata from first trajectory

#### `to_trajectories() -> list[Trajectory]`

Converts batch back to list of trajectories:

```python
trajectories = batch.to_trajectories()
```

#### `pad_to_world_size(world_size: int) -> Batch`

Pads batch to be divisible by world_size (for distributed training):

```python
batch = batch.pad_to_world_size(world_size=4)
# Batch size becomes multiple of 4
```

#### `remove_redundant_left_padding(data, pad_token_id, ...)`

Static method to remove common left padding:

```python
batch = Batch.remove_redundant_left_padding(
    data=batch,
    pad_token_id=0,
    fields=["input_ids", "attention_mask"],
    anchor_field="input_ids",
    max_strip_threshold=512
)
```

**Parameters**:
- `pad_token_id`: ID of padding token
- `fields`: Which fields to strip (None = all 2D tensors)
- `anchor_field`: Field to determine padding length
- `max_strip_threshold`: Maximum tokens to strip (-1 = no limit)

#### `remove_redundant_right_padding(data, pad_token_id, ...)`

Static method to remove common right padding:

```python
batch = Batch.remove_redundant_right_padding(
    data=batch,
    pad_token_id=0,
    fields=["input_ids", "attention_mask"],
    anchor_field="input_ids"
)
```

#### `to_nextrainer_batch() -> dict[str, Any]`

Converts batch to NexTrainer format:

```python
nextrainer_batch = batch.to_nextrainer_batch()
# Returns: {
#     "batch": {tensor_fields...},
#     "non_tensor_batch": {non_tensor_fields...},
#     "meta_info": {...}
# }
```

**Example Usage**:

```python
from nexrl.nexrl_types import Batch
import torch

# Create batch from trajectories
trajectories = [
    Trajectory(
        tokens=[1, 2, 3],
        loss_mask=[0, 1, 1],
        reward=1.0,
        is_val=False,
        extra_fields={"query": "test1"}
    ),
    Trajectory(
        tokens=[4, 5, 6],
        loss_mask=[0, 1, 1],
        reward=0.5,
        is_val=False,
        extra_fields={"query": "test2"}
    ),
]

batch = Batch.from_trajectories(trajectories)

print(f"Batch size: {len(batch)}")
print(f"Tokens shape: {batch.values['tokens'].shape}")  # torch.Size([2, 3])
print(f"Rewards: {batch.values['reward']}")  # [1.0, 0.5]

# Remove padding
batch = Batch.remove_redundant_left_padding(
    batch,
    pad_token_id=0,
    anchor_field="tokens"
)

# Convert back to trajectories
new_trajectories = batch.to_trajectories()
```

### NexRLRole

```python
class NexRLRole(Enum):
    ROLLOUT_WORKER = "rollout_worker"
    TRAINER = "trainer"
    TRAJECTORY_POOL = "trajectory_pool"
    WEIGHT_SYNC_CONTROLLER = "weight_sync_controller"
    DATA_LOADER = "data_loader"
    VALIDATE_DATALOADER = "validate_dataloader"
    VALIDATOR = "validator"
```

**Purpose**: Defines different component roles for:
- Resource pool mapping
- Module registration
- Actor co-location

**Usage**:

```python
from nexrl.nexrl_types import NexRLRole

# Register module with resource manager
resource_manager.register_role(
    role=NexRLRole.ROLLOUT_WORKER,
    cls=MyRolloutWorker,
    config=config,
    count=8
)

# Get actor wrappers
workers = resource_manager.get_actor_wrapper(NexRLRole.ROLLOUT_WORKER)
```

## Evaluation System Types

### BaseEvaluationTarget

```python
@dataclass
class BaseEvaluationTarget:
    final_answer: str
```

**Purpose**: Base class for evaluation targets containing agent's final answer

**Usage**:

```python
from nexrl.rollout_worker import BaseEvaluationTarget

target = BaseEvaluationTarget(final_answer="The answer is 42")
```

### NexAUEvaluationTarget

```python
@dataclass
class NexAUEvaluationTarget(BaseEvaluationTarget):
    final_answer: str
    observation: list[dict[str, Any]]  # Complete execution trajectory
```

**Purpose**: Extended evaluation target for NexAU agents with full execution trace

**Fields**:
- `final_answer`: Agent's final output
- `observation`: List of execution steps with traces, tool calls, etc.

**Usage**:

```python
from nexrl.rollout_worker import NexAUEvaluationTarget

target = NexAUEvaluationTarget(
    final_answer="The answer is 42",
    observation=[
        {"trace": {...}, "step": 1},
        {"trace": {...}, "step": 2},
    ]
)
```

### EvaluationRunResult

```python
@dataclass
class EvaluationRunResult:
    reward: float = 0.0                         # Primary RL reward signal
    ground_truth: str = ""                      # Reference answer
    metrics: dict[str, float] = field(default_factory=dict)  # Additional metrics
    extra_info: dict[str, Any] = field(default_factory=dict)  # Extra information
```

**Purpose**: Contains evaluation results from an evaluator

**Fields**:
- `reward`: Primary training signal (should be in [0, 1] range)
- `ground_truth`: Expected/reference answer
- `metrics`: Additional scalar metrics for logging (must be floats)
- `extra_info`: Any additional information (can be any type)

**Usage**:

```python
from nexrl.rollout_worker import EvaluationRunResult

result = EvaluationRunResult(
    reward=1.0,
    ground_truth="42",
    metrics={
        "exact_match": 1.0,
        "answer_length": 2.0,
    },
    extra_info={
        "parsed_answer": 42,
        "reasoning_steps": ["step1", "step2"],
    }
)
```

## Type Conversions

### Trajectory ↔ Batch

```python
# Trajectories to Batch
trajectories = [traj1, traj2, traj3]
batch = Batch.from_trajectories(trajectories)

# Batch to Trajectories
trajectories = batch.to_trajectories()
```

### Batch ↔ Dictionary

```python
# Batch to dict
combined_dict = batch.to_dict()

# Dict to Batch (manual)
batch = Batch(
    values={"tokens": [...], "rewards": [...]},
    metadata={"batch_size": 32}
)
```

### Batch → NexTrainer Format

```python
# For self-hosted training
nextrainer_batch = batch.to_nextrainer_batch()

# Result structure:
# {
#     "batch": {tensor_fields},
#     "non_tensor_batch": {non_tensor_fields},
#     "meta_info": {metadata}
# }
```

## Common Patterns

### Creating a Trajectory in Rollout Worker

```python
def step(self, task: dict[str, Any]) -> str | None:
    # Run agent/LLM
    response = self.agent.run(task["query"])

    # Evaluate
    eval_result = self.evaluator.evaluate(task, response)

    # Create trajectory
    trajectory = Trajectory(
        tokens=generated_tokens,
        loss_mask=compute_loss_mask(generated_tokens),
        reward=eval_result.reward,
        is_val=False,
        extra_fields={
            "model_tag": self._model_tag,
            "query": task["query"],
            "final_answer": response,
            "ground_truth": eval_result.ground_truth,
            **eval_result.metrics,
            **eval_result.extra_info,
        }
    )

    return self._put_trajectory(trajectory)
```

### Processing a Batch in Trainer

```python
def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
    metrics = {}

    # Remove padding
    batch = Batch.remove_redundant_left_padding(
        batch, pad_token_id=self._pad_token_id
    )

    # Extract rewards
    rewards = batch.values["reward"]

    # Compute advantages
    advantages = compute_advantages(rewards)
    batch.values["advantages"] = advantages

    # Log metrics
    metrics["train/mean_reward"] = rewards.mean().item()
    metrics["train/mean_advantage"] = advantages.mean().item()

    return batch, metrics
```

### Grouping Trajectories

```python
# Group trajectories by run_id
grouped = {}
for trajectory in trajectories:
    run_id = trajectory.get("run_id", "default")
    if run_id not in grouped:
        grouped[run_id] = []
    grouped[run_id].append(trajectory)

# Process each group
for run_id, group in grouped.items():
    # Compute group statistics
    rewards = [t["reward"] for t in group]
    mean_reward = sum(rewards) / len(rewards)
```

## Best Practices

### For Trajectories

1. **Always include model_tag** for multi-model scenarios
2. **Use consistent extra_fields** across your task
3. **Include ground_truth** when available
4. **Store metrics as floats** for aggregation
5. **Use run_id** for GRPO grouping

### For Batches

1. **Validate batch_size** in metadata matches actual size
2. **Use padding removal** to optimize memory and computation
3. **Check tensor shapes** before operations
4. **Convert back to trajectories** sparingly (expensive)
5. **Use to_dict()** carefully (loses structure)

### For Type Safety

1. **Use type hints** with these types in your code
2. **Validate fields** before accessing from extra_fields
3. **Handle missing fields** gracefully with `.get()`
4. **Document custom extra_fields** in your implementation
5. **Test conversions** (especially Trajectory ↔ Batch)

## Next Steps

- Learn how [Controller](./controller.md) manages these types
- Understand [Rollout Workers](../05-rollout-workers/overview.md) generate trajectories
- Explore [Trainers](../06-trainers/overview.md) process batches
- See [Trajectory Pool](../04-trajectory-pool/trajectory-pool.md) batching strategies
