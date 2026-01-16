# Weight Synchronization

## Overview

`WeightSyncController` manages model weight synchronization between training and inference services in NexRL. It coordinates when weights should be synchronized, blocks services during updates, and manages validation triggers.

**Location**: `nexrl/weight_sync/weight_sync_controller.py`

## Key Concepts

### Synchronization Lifecycle

```
Training Complete → Check Sync Mode → Sync Weights → Update Version → Unlock Services
                                           ↓
                                    Validation (if needed)
```

### Model Version Tracking

Each model maintains two version numbers:
- **train_model_version**: Latest trained model version
- **rollout_model_version**: Version currently used for rollout/inference

### Service States

`RolloutServiceState` tracks the synchronization state:
- **running**: Service operational, no sync needed
- **need_sync**: Sync required, services being locked
- **syncing**: Sync in progress

## Synchronization Modes

NexRL supports three synchronization modes via `sync_mode` config:

### 1. sync (Synchronous)

All rollout workers block until weight sync completes after every training step.

**When to use**: Maximum training stability, immediate weight updates

**Trade-off**: Lower throughput due to frequent blocking

```yaml
weight_sync_controller:
  sync_mode: "sync"
```

### 2. fully-async (Fully Asynchronous)

No blocking - workers continue with stale weights until sync completes opportunistically.

**When to use**: Maximum throughput, can tolerate stale weights

**Trade-off**: Workers may use outdated weights for multiple steps

```yaml
weight_sync_controller:
  sync_mode: "fully-async"
```

### 3. batch-async (Batch Asynchronous)

Workers block only when staleness exceeds threshold.

**When to use**: Balance between throughput and weight freshness

**Trade-off**: Configurable staleness threshold

```yaml
weight_sync_controller:
  sync_mode: "batch-async"
  staleness_threshold: 2  # Block if behind by 2+ versions
```

## Weight Sync Methods

NexRL supports multiple weight transfer methods via `sync_method` config:

### network

Transfer weights via HTTP to vLLM or SGLang backends.

```yaml
weight_sync_controller:
  sync_method: "network"
  sync_weight_path: "/path/to/weights"
```

**Backends**: vLLM, SGLang

### disk

Update weights from disk path. For SGLang, queries all workers via sglang-router and updates in parallel.

```yaml
weight_sync_controller:
  sync_method: "disk"
  sync_weight_path: "/path/to/weights"
```

**Backends**: SGLang (with sglang-router)

### tinker

Use Tinker service backend for weight sync.

```yaml
weight_sync_controller:
  sync_method: "tinker"
```

### weaver

Use Weaver service backend for weight sync.

```yaml
weight_sync_controller:
  sync_method: "weaver"
```

## Core Methods

### Constructor

```python
def __init__(self, config: DictConfig)
```

Initializes weight sync controller with configuration.

**Parameters**:
- `config`: Configuration including sync_mode, sync_method, validate_freq

### check_rollout_service_status

```python
def check_rollout_service_status(self, model_tag: ModelTag) -> Literal["continue", "block"]
```

Called by `InferenceServiceClient` before each request to check if service should proceed or block.

**Returns**:
- `"continue"`: Service can proceed
- `"block"`: Service should block and retry

### trajectory_pool_notify_batch_ready

```python
def trajectory_pool_notify_batch_ready(self, model_tag: ModelTag) -> None
```

Called when TrajectoryPool has a batch ready. Checks if sync is needed and locks services if required.

**Logic**:
1. Check sync mode and version staleness
2. If sync needed: set state to `"need_sync"`, lock trajectory pool
3. If not needed: unlock services

### train_worker_notify_weight_update

```python
def train_worker_notify_weight_update(self, worker_name: str, model_tag: ModelTag) -> None
```

Called when training completes. Performs synchronous weight update if needed.

**Process**:
1. Increment `train_model_version`
2. If state is `"need_sync"`, perform sync
3. Update `rollout_model_version`
4. Check validation frequency
5. Unlock services (unless validation needed)

### sync_weight_to_rollout_service

```python
def sync_weight_to_rollout_service(self, model_tag: ModelTag) -> bool
```

Performs actual weight synchronization based on `sync_method`.

**Returns**: `True` if successful, `False` otherwise

## Validation Coordination

Weight sync controller coordinates validation runs triggered at regular intervals.

### Configuration

```yaml
weight_sync_controller:
  validate_freq: 5  # Run validation every 5 training steps
```

### Process

When `rollout_model_version % validate_freq == 0`:
1. Set `_waiting_for_validation = True`
2. Keep services locked
3. Controller runs validation
4. Controller calls `end_validate()` to unlock

### end_validate

```python
def end_validate(self, model_tag: ModelTag) -> None
```

Called by controller after validation completes. Releases locks on dataloader and trajectory pool.

## RolloutServiceState

Data structure tracking per-model sync state:

```python
@dataclass
class RolloutServiceState:
    # State tracking
    state: Literal["need_sync", "syncing", "running"] = "running"
    rollout_model_version: int = 0
    train_model_version: int = 0

    # Service info
    model_name: str = ""
    weight_type: str = ""
    weight_path: str = ""
    backend: str = ""
    base_url: str = ""
```

## Configuration Reference

```yaml
weight_sync_controller:
  type: "weight_sync_controller"

  # Synchronization mode
  sync_mode: "sync"  # "sync" | "fully-async" | "batch-async"

  # Staleness threshold (for batch-async mode)
  staleness_threshold: 2

  # Weight sync method
  sync_method: "disk"  # "network" | "disk" | "tinker" | "weaver" | "mock"

  # Weight path (for network/disk methods)
  sync_weight_path: "/path/to/weights"

  # Validation frequency (0 = disabled)
  validate_freq: 5

  # Inference service config (model info)
  inference_service:
    model_tag: "default"
    model: "model-name"
    weight_type: "full"
    backend: "sglang"
    base_url: "http://localhost:8000"
```

## Integration Example

The controller initializes and wires the weight sync controller:

```python
# In NexRLController.__init__
self.weight_sync_controller = WeightSyncController(config.weight_sync_controller)

# Set module references
self.weight_sync_controller.set_module_references(
    dataloader=self.dataloader,
    trajectory_pool=self.trajectory_pool
)

# For Tinker backend
if config.inference_service.backend == "tinker":
    self.weight_sync_controller.set_tinker_service_holder(self.tinker_service_holder)

# For Weaver backend
if config.inference_service.backend == "weaver":
    self.weight_sync_controller.set_weaver_service_holder(self.weaver_service_holder)
```

## Related Documentation

- [Core Architecture](../02-core-architecture/overview.md) - System architecture
- [Trajectory Pool](../04-trajectory-pool/trajectory-pool.md) - Trajectory collection
- [Training Service](../07-services/training-service.md) - Training backend
- [Inference Service](../07-services/inference-service.md) - Inference backend
- [Validation](./validation.md) - Validation coordination
