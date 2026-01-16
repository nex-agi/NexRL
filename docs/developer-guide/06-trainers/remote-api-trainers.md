# Remote API Trainers

Remote API trainers are used with Tinker/Weaver backends for training on external infrastructure. These trainers work directly with trajectories and delegate tokenization/batching to the remote service.

## Architecture

```python
BaseTrainer
    ↓
RemoteApiTrainer (abstract)
    ↓
    ┌────────────────────────────┐
    ↓                            ↓
RemoteApiGrpoTrainer    RemoteApiCrossEntropyTrainer
```

## RemoteApiTrainer

Base class for remote API training backends. Located in `nexrl/trainer/remote_api_trainer.py`.

### Purpose

Provides core training loop for remote API backends:
1. Algorithm-specific trajectory preparation
2. Conversion to service datum format
3. Remote API communication
4. Metrics collection and logging

### Key Differences from Self-Hosted

| Aspect | Self-Hosted | Remote API |
|--------|-------------|------------|
| **Data Format** | Batch (tensors) | Trajectories (dicts) |
| **Tokenization** | Client-side | Service-side |
| **Batching** | Client-side | Service-side |
| **Algorithm Hook** | `_prepare_batch()` | `_prepare_trajectories()` |
| **Padding** | Explicit left/right | Handled by service |

### Constructor

```python
def __init__(self, config: DictConfig)
```

**Key Initialization:**
- Extracts training hyperparameters (lr, beta1, beta2, etc.)
- Sets loss function (importance_sampling, cross_entropy, etc.)
- Service holder is set later via `set_service_holder()`

### Main Training Flow

```python
def train(self, trajectories: list[Trajectory]) -> dict:
    metrics = {}

    # 1. Prepare trajectories (algorithm-specific)
    trajectories = self._prepare_trajectories(trajectories, metrics)

    # 2. Log rollout metrics
    log_rollout_metrics(trajectories, metrics, prefix="rollout/")

    # 3. Convert trajectories to service datums
    datums = convert_trajectories_to_datums(trajectories)

    # 4. Call remote API
    train_config = {...}  # lr, loss_fn, etc.
    response = self._service_holder.train(datums, config=train_config)

    # 5. Update metrics and counters
    metrics.update(response.get("metrics", {}))
    self._train_step += 1

    return metrics
```

### Key Methods

#### _prepare_trajectories() (Abstract)

```python
@abstractmethod
def _prepare_trajectories(
    self,
    trajectories: list[Trajectory],
    metrics: dict[str, Any]
) -> list[Trajectory]
```

Algorithm-specific trajectory preparation. Subclasses must implement this.

**Parameters:**
- `trajectories`: List of trajectories from rollout
- `metrics`: Dictionary to populate with metrics

**Returns:** Processed trajectories with algorithm-specific fields

**Purpose:** Add algorithm-specific fields (advantages, labels, etc.) to trajectories before sending to remote service.

#### set_service_holder()

```python
def set_service_holder(self, service_holder: TinkerServiceHolder | WeaverServiceHolder)
```

Sets the remote service holder for API communication.

**Called by:** Controller during initialization

## RemoteApiGrpoTrainer

Implements GRPO algorithm for remote API backends. Located in `nexrl/trainer/remote_api_grpo_trainer.py`.

### _prepare_trajectories() Implementation

```python
def _prepare_trajectories(
    self, trajectories: list[Trajectory], metrics: dict[str, Any]
) -> list[Trajectory]:
    from ..algorithm.core_algos import compute_grpo_advantage_for_trajectories
    from ..utils.logging_utils import log_grpo_metrics

    # Compute GRPO advantages (groups by run_id)
    trajectories = compute_grpo_advantage_for_trajectories(
        trajectories, logger=logger, use_run_ids=True
    )

    # Log GRPO statistics
    log_grpo_metrics(trajectories, metrics)

    return trajectories
```

**Process:**
1. Groups trajectories by `run_id` (same prompt)
2. Computes group-relative advantages within each group
3. Normalizes by group standard deviation: `(reward - mean) / std`
4. Adds `advantage` field to each trajectory
5. Logs GRPO metrics (mean reward, std, etc.)

### Configuration Example

```yaml
trainer:
  total_train_steps: 100

  train_service:
    backend: "tinker"  # or "weaver"
    url: "http://tinker-service:8000"

    config:
      # Training hyperparameters
      loss_fn: "importance_sampling"
      learning_rate: 2e-6
      beta1: 0.9
      beta2: 0.95
      eps: 1e-8

algorithm:
  # Trainer is auto-selected as RemoteApiGrpoTrainer
  # GRPO uses run_id grouping automatically
```

## RemoteApiCrossEntropyTrainer

Implements supervised learning for remote API backends. Located in `nexrl/trainer/remote_api_cross_entropy_trainer.py`.

### Purpose

Used for supervised fine-tuning tasks where:
- Trajectories contain reference outputs
- No advantage computation needed
- Uses cross-entropy loss directly

### _prepare_trajectories() Implementation

```python
def _prepare_trajectories(
    self, trajectories: list[Trajectory], metrics: dict[str, Any]
) -> list[Trajectory]:
    # No preparation needed for cross entropy
    # loss_mask already indicates which tokens to train on
    return trajectories
```

**Rationale:** For supervised learning, the `loss_mask` field already indicates which tokens should contribute to the loss. No advantage computation or reward processing is needed.

### Configuration Example

```yaml
trainer:
  total_train_steps: 100

  train_service:
    backend: "weaver"
    url: "http://weaver-service:8000"

    config:
      loss_fn: "cross_entropy"
      learning_rate: 1e-5
      beta1: 0.9
      beta2: 0.999

algorithm:
  # Trainer is auto-selected as RemoteApiCrossEntropyTrainer
```

## Service Datum Format

Remote API trainers convert trajectories to a service-specific datum format:

```python
from nexrl.utils.finetune_service_utils import convert_trajectories_to_datums

datums = convert_trajectories_to_datums(trajectories)
# Each datum contains:
# - prompt: str
# - response: str
# - advantage: float (for GRPO)
# - metadata: dict
```

The service backend handles:
- Tokenization
- Batching
- Padding
- Device placement
- Training step execution

## Key Features

### Trajectory-Based Processing

Remote API trainers work directly with trajectories (no Batch conversion):

```python
# Trajectory format
{
    "prompt": "What is 2+2?",
    "response": "The answer is 4.",
    "reward": 1.0,
    "run_id": "uid_123_run_0",
    "advantage": 0.5,  # Added by _prepare_trajectories
    # ... other fields
}
```

### Service Holder Pattern

Remote trainers use service holders for API abstraction:

```python
# Controller sets the service holder
if backend == "tinker":
    service_holder = TinkerServiceHolder(url)
elif backend == "weaver":
    service_holder = WeaverServiceHolder(url)

trainer.set_service_holder(service_holder)
```

This allows the same trainer code to work with different backend services.

### Automatic Backend Selection

The controller automatically selects the appropriate trainer based on configuration:

```python
# In controller
if backend == "self-hosted":
    trainer = SelfHostedGrpoTrainer(config)
elif backend in ["tinker", "weaver"]:
    if algorithm == "grpo":
        trainer = RemoteApiGrpoTrainer(config)
    elif algorithm == "cross_entropy":
        trainer = RemoteApiCrossEntropyTrainer(config)
```

## Usage Example

Remote API trainers are automatically instantiated by the controller. Typical usage only requires configuration:

```yaml
# For GRPO with Tinker
trainer:
  total_train_steps: 100
  train_service:
    backend: "tinker"
    url: "http://tinker-service:8000"
    config:
      loss_fn: "importance_sampling"
      learning_rate: 2e-6

# For supervised learning with Weaver
trainer:
  total_train_steps: 100
  train_service:
    backend: "weaver"
    url: "http://weaver-service:8000"
    config:
      loss_fn: "cross_entropy"
      learning_rate: 1e-5
```

For custom algorithms, extend `RemoteApiTrainer`:

```python
from nexrl.trainer import RemoteApiTrainer
from nexrl.nexrl_types import Trajectory

class MyCustomRemoteTrainer(RemoteApiTrainer):
    def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict
    ) -> list[Trajectory]:
        # Your custom algorithm logic
        for traj in trajectories:
            # Add custom fields
            traj.custom_field = compute_custom_value(traj)

        return trajectories
```

See [Custom Trainers](./custom-trainers.md) for more details.

## Comparison: GRPO vs Cross Entropy

| Feature | RemoteApiGrpoTrainer | RemoteApiCrossEntropyTrainer |
|---------|----------------------|------------------------------|
| **Algorithm** | Group Relative Policy Optimization | Supervised Learning |
| **Loss Function** | importance_sampling | cross_entropy |
| **Advantage Computation** | Yes (group-relative) | No |
| **Trajectory Grouping** | By run_id | No grouping |
| **Use Case** | RL with reward signal | Supervised fine-tuning |
| **Requires reward** | Yes | No |

## Related Documentation

- [Overview](./overview.md) - Trainer architecture overview
- [Self-Hosted Trainers](./self-hosted-trainers.md) - Self-hosted training details
- [Training Services](../08-services/training-service.md) - Backend service integration
- [GRPO Algorithm](../07-algorithms/grpo.md) - GRPO implementation details
- [Custom Trainers](./custom-trainers.md) - Creating custom trainers
