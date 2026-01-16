# Checkpointing

## Overview

NexRL provides checkpoint management for model persistence and training resumption. Checkpoints can be saved periodically or for weight synchronization, and loaded at startup for resuming training.

**Key Components**:
- `NexRLController`: Handles checkpoint loading at initialization
- `Trainer`: Triggers checkpoint saves during training
- `TrainServiceClient`: Interface for checkpoint operations
- `DCPCheckpointManager`: Manages FSDP checkpoint storage (in training service backend)

## Checkpoint Types

### 1. Weight-Only Checkpoints

Saved after each training step for weight synchronization.

**Purpose**: Fast weight updates to inference service

**Configuration**:
```yaml
trainer:
  sync_weight_path: "/path/to/sync_weights"
```

**Saved Contents**:
- Model weights only
- No optimizer state
- No scheduler state

### 2. Full Checkpoints

Saved periodically for training resumption.

**Purpose**: Complete training state for resume

**Configuration**:
```yaml
trainer:
  checkpoint_path: "/path/to/checkpoints"
  save_freq: 100  # Save every 100 steps
```

**Saved Contents**:
- Model weights (sharded with FSDP)
- Optimizer state (sharded with FSDP)
- LR scheduler state
- Tokenizer and model config
- Global step number

**Directory Structure**:
```
checkpoint_path/
├── global_step_100/
│   ├── model/
│   ├── optimizer/
│   └── ...
├── global_step_200/
└── global_step_300/
```

## Resume Modes

NexRL supports three resume modes via `resume.mode` config:

### 1. disable

Train from scratch, ignore existing checkpoints.

```yaml
resume:
  mode: "disable"
```

### 2. auto

Automatically find and load the latest checkpoint.

```yaml
resume:
  mode: "auto"
```

**Process**:
1. Search `checkpoint_path` for `global_step_*` directories
2. Find highest step number
3. Load that checkpoint
4. If no checkpoint found, train from scratch

### 3. from_path

Load from specific checkpoint path.

```yaml
resume:
  mode: "from_path"
  resume_path: "/path/to/checkpoint/global_step_500"
```

**Use Case**: Resume from specific checkpoint (not necessarily latest)

## Checkpoint Workflow

### Saving Checkpoints

#### Weight-Only (Every Step)

```python
# In SelfHostedTrainer.train
self._train_service_client.save_checkpoint(
    local_path=self._config.sync_weight_path,
    global_step=0,  # Not tracked for sync weights
    saved_fully_shared_ckpt=False,
    save_weight_only=True,
    remove_previous_ckpt=False
)
```

#### Full Checkpoint (Periodic)

```python
# In SelfHostedTrainer.train
if self._train_step % save_freq == 0:
    local_global_step_folder = os.path.join(
        self._config.checkpoint_path,
        f"global_step_{self._train_step}"
    )
    self._train_service_client.save_checkpoint(
        local_path=local_global_step_folder,
        global_step=self._train_step,
        saved_fully_shared_ckpt=True,
        save_weight_only=False,
        remove_previous_ckpt=False
    )
```

### Loading Checkpoints

#### Initial Load (Controller)

```python
# In NexRLController._load_initial_checkpoint
if self._config.resume.mode != "disable":
    self._load_resume_checkpoint()
```

#### Resume Load

```python
# In NexRLController._load_resume_checkpoint
def _load_resume_checkpoint(self):
    # Find checkpoint based on resume mode
    if self._config.resume.mode == "auto":
        global_step_folder = self._find_latest_checkpoint(checkpoint_folder)
    elif self._config.resume.mode == "from_path":
        global_step_folder = self._config.resume.resume_path

    # Load checkpoint
    with self._train_service_client.actor_context():
        result = self._train_service_client.load_checkpoint(
            path=global_step_folder,
            del_local_after_load=False,
            load_weight_only=False
        )

    # Update trainer's global step
    execute(self.trainer.set_train_step, global_step)
```

## TrainServiceClient Interface

### save_checkpoint

```python
def save_checkpoint(
    self,
    local_path: str,
    hdfs_path: str | None = None,
    global_step: int = 0,
    saved_fully_shared_ckpt: bool = True,
    save_weight_only: bool = False,
    remove_previous_ckpt: bool = True
) -> dict[str, Any]
```

**Parameters**:
- `local_path`: Local path to save checkpoint
- `hdfs_path`: Optional HDFS path (not commonly used)
- `global_step`: Training step number
- `saved_fully_shared_ckpt`: Whether to save complete state
- `save_weight_only`: Save only weights (for sync)
- `remove_previous_ckpt`: Remove previous checkpoint (cleanup)

### load_checkpoint

```python
def load_checkpoint(
    self,
    path: str,
    del_local_after_load: bool = True,
    load_weight_only: bool = False
) -> dict[str, Any]
```

**Parameters**:
- `path`: Path to checkpoint directory
- `del_local_after_load`: Delete local copy after loading
- `load_weight_only`: Load only weights (ignore optimizer/scheduler)

## DCPCheckpointManager

Backend implementation for FSDP checkpoint management (in training service).

**Location**: `nexrl/train_service_backend/utils/checkpoint_manager.py`

### Key Features

- **Sharded Storage**: Saves FSDP model and optimizer shards per rank
- **DCP Format**: Uses PyTorch Distributed Checkpoint (DCP)
- **Weight Provider**: Optional network-based weight sync (internal feature)

### save_checkpoint

```python
def save_checkpoint(
    self,
    local_path: str,
    global_step: int,
    remove_previous_ckpt=False,
    save_weight_only=False,
    saved_fully_shared_ckpt=True
)
```

Saves model, optimizer, scheduler state to `local_path`.

### load_checkpoint

```python
def load_checkpoint(
    self,
    path=None,
    del_local_after_load=True,
    load_weight_only=False
)
```

Loads checkpoint from `path` into model and optimizer.

## Configuration Reference

### Complete Example

```yaml
trainer:
  # Weight sync path (used every step)
  sync_weight_path: "/workspace/sync_weights"

  # Full checkpoint path
  checkpoint_path: "/workspace/checkpoints"

  # Save frequency (steps)
  save_freq: 100

resume:
  # Resume mode: "disable" | "auto" | "from_path"
  mode: "auto"

  # Path for from_path mode
  resume_path: ""
```

### Resume from Scratch

```yaml
resume:
  mode: "disable"
```

### Auto Resume Latest

```yaml
resume:
  mode: "auto"

trainer:
  checkpoint_path: "/workspace/checkpoints"
```

### Resume from Specific Checkpoint

```yaml
resume:
  mode: "from_path"
  resume_path: "/workspace/checkpoints/global_step_500"
```

## Best Practices

### Save Frequency

Balance checkpoint overhead with recovery granularity:
- **High frequency** (every 10-50 steps): Minimize lost work, higher overhead
- **Low frequency** (every 100-500 steps): Lower overhead, more potential work loss

```yaml
trainer:
  save_freq: 100  # Reasonable default
```

### Storage Management

Full checkpoints consume significant storage. Consider:
- Periodic cleanup of old checkpoints
- Separate checkpoint_path for experiments
- Use shared storage for multi-node setups

### Path Configuration

Use absolute paths for reliability:

```yaml
trainer:
  sync_weight_path: "/workspace/exp001/sync_weights"
  checkpoint_path: "/workspace/exp001/checkpoints"
```

Relative paths resolve from working directory (can be fragile).

### Weight-Only vs Full

- **Weight-only**: Fast, used for every training step (weight sync)
- **Full checkpoint**: Slower, used periodically (resume capability)

Both are necessary for production deployments.

## Troubleshooting

### Checkpoint Not Found

```
WARNING: Checkpoint folder does not exist: /path/to/checkpoints
```

**Solution**: Ensure `checkpoint_path` is correct and accessible. In `auto` mode, missing checkpoints are not an error (trains from scratch).

### Resume Path Invalid

```
ERROR: Resume checkpoint path does not exist: /path/to/checkpoint
```

**Solution**: Verify `resume_path` when using `from_path` mode. Path must exist.

### Checkpoint Load Failed

```
ERROR: Failed to load checkpoint: ...
```

**Possible Causes**:
- Checkpoint corrupted
- Model architecture mismatch
- FSDP configuration changed

**Solution**: Verify checkpoint integrity, model config matches training setup.

## Related Documentation

- [Core Architecture](../02-core-architecture/overview.md) - System overview
- [Self-Hosted Trainers](../06-trainers/self-hosted-trainers.md) - Trainer implementation
- [Training Service](../07-services/training-service.md) - Training backend
- [Weight Synchronization](./weight-synchronization.md) - Weight sync coordination
