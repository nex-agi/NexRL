# Trainers Overview

This section covers the trainer architecture in NexRL, which handles the training loop and algorithm integration.

## Trainer Architecture

NexRL supports two training backend modes, each with its own trainer hierarchy:

### 1. Self-Hosted Training (NexTrainer Backend)

```
                BaseTrainer
                    ↓
            SelfHostedTrainer
      (abstract _prepare_batch method)
                    ↓
        ┌──────────────────────┐
        ↓                      ↓
SelfHostedGrpoTrainer    Custom Trainer
 (implements GRPO)      (custom algorithm)
```

**Pipeline:**
```
Trajectories → Process → _prepare_batch (algorithm) → Train Service → NexTrainer
```

**Key Features:**
- Batch-based training with tokenization and padding
- Abstract `_prepare_batch()` method for algorithm-specific logic
- Integrated checkpointing and metrics tracking
- Handles tensor conversions and GPU data transfer

### 2. Remote API Training (Tinker/Weaver Backend)

```
                 BaseTrainer
                     ↓
             RemoteApiTrainer
    (abstract _prepare_trajectories method)
                     ↓
    ┌────────────────────────────────┐
    ↓                                ↓
RemoteApiGrpoTrainer          RemoteApiCrossEntropyTrainer
  (GRPO algorithm)              (supervised learning)
```

**Pipeline:**
```
Trajectories → _prepare_trajectories (algorithm) → Service Datums → Tinker/Weaver API
```

**Key Features:**
- Trajectory-based training (no Batch conversion)
- Abstract `_prepare_trajectories()` for algorithm-specific logic
- Service holder pattern for API communication
- Remote backend handles tokenization and batching

## BaseTrainer

Abstract base class for all trainers. Located in `nexrl/trainer/base_trainer.py`.

### Constructor

```python
def __init__(self, config: DictConfig)
```

**Parameters:**
- `config`: Configuration dictionary with training settings

### Core Methods

#### train()

```python
@abstractmethod
def train(self, trajectories: list[Trajectory]) -> dict
```

Main training method that derived classes must implement.

**Parameters:**
- `trajectories`: List of trajectories to train on

**Returns:** Dictionary of training metrics

**Responsibilities:**
1. Trajectory processing
2. Algorithm-specific preparation
3. Training step execution
4. Metrics collection

#### initialize_workers()

```python
def initialize_workers() -> None
```

Initialize backend training workers. Override in derived classes if needed.

#### run() / stop()

```python
def run() -> None
def stop() -> None
```

Start/stop the training loop in a background thread.

### Module References

```python
def set_module_references(
    self,
    trajectory_pool: TrajectoryPool,
    weight_sync_controller: WeightSyncController,
) -> None
```

Sets references to other NexRL modules. Called by the controller during initialization.

## Training Loop

The base trainer implements a standard training loop:

```python
def _main_loop(self):
    while not self._stop_event.is_set() and self._train_step < self._total_train_steps:
        # 1. Get batch from trajectory pool
        trajectories = self._trajectory_pool.get_batch_blocking()

        # 2. Train on trajectories (algorithm-specific)
        metrics = self.train(trajectories)

        # 3. Update step counter
        self._train_step += 1

        # 4. Log metrics
        self._activity_tracker.log_train_step(metrics, self._train_step)
```

## Choosing a Trainer Type

### Use Self-Hosted Trainers When:
- Full control over training algorithm needed
- Custom batch processing required
- Using NexTrainer backend
- Need direct access to model parameters

### Use Remote API Trainers When:
- Using Tinker or Weaver service backends
- Training on external infrastructure
- Want service abstraction for easier deployment
- Using pre-built training services

## Configuration

Trainers are configured in the recipe YAML:

```yaml
trainer:
  # Training duration
  total_train_steps: 100

  # Trainer class selection (auto-selected based on backend)
  # - SelfHostedGrpoTrainer (for self-hosted + GRPO)
  # - RemoteApiGrpoTrainer (for remote + GRPO)
  # - RemoteApiCrossEntropyTrainer (for remote + supervised)

  # Backend configuration
  train_service:
    backend: "self-hosted"  # or "tinker", "weaver"
    url: "http://localhost:5000"
    world_size: 8  # for self-hosted
```

## Next Steps

- [Self-Hosted Trainers](./self-hosted-trainers.md) - SelfHostedTrainer and SelfHostedGrpoTrainer
- [Remote API Trainers](./remote-api-trainers.md) - RemoteApiTrainer variants
- [Custom Trainers](./custom-trainers.md) - Creating custom algorithm trainers
- [GRPO Algorithm](../07-algorithms/grpo.md) - GRPO implementation details

## Related Documentation

- [Training Services](../08-services/training-service.md) - Backend service integration
- [Configuration Reference](../12-configuration-reference/trainer-config.md) - Complete trainer configuration
- [Core Architecture](../02-core-architecture/overview.md) - System architecture overview
