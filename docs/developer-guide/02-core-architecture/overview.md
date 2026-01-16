# Core Architecture Overview

NexRL is a large-scale distributed reinforcement learning training framework designed for modern RL applications. This document provides an overview of the system architecture and core components.

## Key Features

- **Multiple Launch Mode Support**: Seamlessly runs in both local and Ray distributed modes
- **Modular Design**: Clean separation of concerns with well-defined interfaces and extensible components
- **Training-as-a-Service & Rollout-as-a-Service**: Unified API architecture that seamlessly supports different training and inference frameworks through service abstraction
- **Resource Management**: Intelligent placement and co-location of services for optimal performance
- **Activity Tracking**: Comprehensive monitoring and health checking system for production deployments
- **Error Handling**: Centralized error reporting and recovery mechanisms

## Architecture Diagram

![NexRL Architecture](../imgs/nexrl_architecture.png)

## Core Components

### 1. NexRLController

**Purpose**: Main orchestrator that initializes and coordinates all components

**Responsibilities**:
- Initialize all framework modules based on launch mode
- Manage module lifecycle (start, stop, health checks)
- Coordinate validation cycles and checkpointing
- Monitor system health and handle errors
- Manage weight synchronization timing

**Key Methods**:
- `__init__(config)`: Initialize all modules
- `run()`: Start training loop and monitoring
- `stop()`: Gracefully shutdown all components
- `_check_finish()`: Determine if training should terminate

See [Controller Details](./controller.md) for in-depth documentation.

### 2. DataLoader

**Purpose**: Provides input data for rollout workers (training and validation)

**Responsibilities**:
- Load and manage training/validation datasets
- Shuffle and batch data appropriately
- Track data consumption and completion
- Support data replay for re-rollout scenarios

**Implementations**:
- `MockDataLoader`: For testing and development
- `TorchDataLoader`: PyTorch-based data loading from files
- `BaseDataLoader`: Abstract interface for custom loaders

See [Data Loader Guide](../03-data-loader/data-loader.md) for detailed documentation.

### 3. RolloutWorkers

**Purpose**: Execute environment interactions and generate trajectories

**Responsibilities**:
- Retrieve tasks from dataloader
- Execute agent/LLM interactions
- Evaluate results and compute rewards
- Generate training trajectories
- Submit trajectories to trajectory pool
- Support validation mode switching

**Implementations**:
- `SimpleRolloutWorker`: Basic LLM completion
- `AgentRolloutWorker`: Agent-based with tools
- `BaseNexAURolloutWorker`: NexAU agent integration (default for most tasks)
- Custom workers: Task-specific implementations

See [Rollout Workers Guide](../05-rollout-workers/overview.md) for detailed documentation.

### 4. TrajectoryPool

**Purpose**: Collects and batches trajectories from rollout workers

**Responsibilities**:
- Receive trajectories from multiple workers
- Group trajectories (by query, user, etc.)
- Create batches when ready
- Coordinate with weight synchronization
- Manage multiple model stores

**Features**:
- Flexible grouping strategies (simple, grouped, hierarchical)
- Configurable batch readiness criteria
- Weight sync coordination

See [Trajectory Pool Guide](../04-trajectory-pool/trajectory-pool.md) for detailed documentation.

### 5. Trainer

**Purpose**: Processes trajectories and executes training (with integrated algorithm logic)

**Responsibilities**:
- Fetch batches from trajectory pool
- Prepare batches (algorithm-specific)
- Execute training steps via training service
- Manage checkpoints
- Compute and log metrics
- Notify weight sync on updates

**Implementations**:
- `SelfHostedGrpoTrainer`: GRPO for NexTrainer backend
- `RemoteApiGrpoTrainer`: GRPO for Tinker/Weaver
- `RemoteApiCrossEntropyTrainer`: Supervised learning for Tinker/Weaver
- Custom trainers: New RL algorithms

See [Trainers Guide](../06-trainers/overview.md) for detailed documentation.

### 6. WeightSyncController

**Purpose**: Manages model weights and synchronization coordination

**Responsibilities**:
- Coordinate weight updates between training and inference
- Manage synchronization modes (sync, fully-async, batch-async)
- Track model versions and staleness
- Trigger validation cycles
- Block workers during sync when needed

**Synchronization Modes**:
- **sync**: All workers wait for newest version
- **fully-async**: No blocking, opportunistic updates
- **batch-async**: Block when staleness exceeds threshold

See [Weight Synchronization Guide](../08-features/weight-synchronization.md) for detailed documentation.

### 7. Validator

**Purpose**: Collects validation trajectories and computes metrics

**Responsibilities**:
- Receive validation trajectories from workers
- Track validation completion
- Compute metrics (mean reward, accuracy, etc.)
- Log validation results
- Notify controller when complete

**Features**:
- Simple collection without complex batching
- Automatic metric aggregation
- Integration with weight sync cycles

See [Validation Guide](../08-features/validation.md) for detailed documentation.

### 8. ActivityTracker

**Purpose**: Monitors system health and activity, coordinates experiment logging

**Responsibilities**:
- Track in-flight work across all modules
- Monitor module liveness
- Centralized error reporting
- Coordinate experiment logging (W&B, etc.)
- Provide quiescence detection

**Key Features**:
- Context manager API for easy tracking
- Automatic error reporting
- Rollout worker-specific monitoring
- Training step tracking

See [Activity Tracking Guide](./activity-tracking.md) for detailed documentation.

### 9. RayResourceManager

**Purpose**: Handles distributed resource allocation and actor co-location

**Responsibilities**:
- Register and create Ray actors
- Manage actor co-location groups
- Provide unified access to actors
- Handle method name prefixing for co-located actors

**Features**:
- Flexible co-location strategies
- Transparent method access
- Resource pool management

See [Distributed Execution Guide](../11-distributed-execution/ray-integration.md) for detailed documentation.

## Data Flow

### Training Pipeline

```
┌─────────────┐
│ DataLoader  │
└──────┬──────┘
       │ get_next_item()
       ▼
┌─────────────────┐
│ RolloutWorker   │
│  - format_query │
│  - run_agent    │
│  - evaluate     │
└──────┬──────────┘
       │ put_trajectory()
       ▼
┌──────────────────┐
│ TrajectoryPool   │
│  - group         │
│  - batch         │
└──────┬───────────┘
       │ get_batch()
       ▼
┌──────────────────┐
│ Trainer          │
│  - prepare_batch │
│  - train         │
└──────┬───────────┘
       │ notify_weight_update()
       ▼
┌──────────────────────┐
│ WeightSyncController │
│  - sync_weights      │
│  - trigger_validate  │
└──────────────────────┘
```

### Validation Pipeline

```
┌──────────────────┐
│ ValidateDataLoader│
└──────┬────────────┘
       │ (validation mode)
       ▼
┌─────────────────┐
│ RolloutWorker   │
│  (validate mode)│
└──────┬──────────┘
       │ put_trajectory()
       ▼
┌──────────────┐
│ Validator    │
│  - collect   │
│  - compute   │
└──────┬───────┘
       │ metrics
       ▼
┌──────────────────────┐
│ WeightSyncController │
│  (unlock)            │
└──────────────────────┘
```

## Module Interaction Patterns

### Initialization Pattern

1. **Controller** creates all modules
2. **Controller** sets up module references
   - RolloutWorkers ← DataLoader, TrajectoryPool, WeightSync
   - TrajectoryPool ← DataLoader, WeightSync
   - Trainer ← TrajectoryPool, WeightSync
3. **Controller** sets up ActivityTracker for all
4. **Controller** starts all modules

### Activity Tracking Pattern

```python
# In any module
with self._activity_tracker.track("ModuleName", "work_description"):
    # Perform work
    result = do_work()
    # Automatic error reporting and activity tracking
```

### Weight Synchronization Pattern

1. **Trainer** completes training step
2. **Trainer** notifies WeightSync
3. **WeightSync** blocks new trajectories
4. **WeightSync** syncs weights to inference service
5. **WeightSync** triggers validation (if configured)
6. **WeightSync** unblocks trajectories

### Error Handling Pattern

1. **Module** encounters error
2. **Module** reports via ActivityTracker
3. **ActivityTracker** logs error with context
4. **Controller** checks health periodically
5. **Controller** decides action based on policy

## Configuration Structure

NexRL uses Hydra for configuration management. Key sections:

```yaml
# Launch and project info
launch_mode: "local" or "ray"
project_name: "NexRL-Project"
experiment_name: "experiment-v1"

# Core components
data: {...}                    # DataLoader configuration
rollout_worker: {...}          # RolloutWorker configuration
trajectory_pool: {...}         # TrajectoryPool configuration
trainer: {...}                 # Trainer configuration
algorithm: {...}               # Algorithm configuration (self-hosted only)
weight: {...}                  # WeightSyncController configuration

# Services
service:
  train_service: {...}         # Training service configuration
  inference_service: {...}     # Inference service configuration

# Features
validate: {...}                # Validation configuration
resume: {...}                  # Checkpoint resume configuration
logger: {...}                  # Experiment logging configuration
runtime_monitor: {...}         # Health and exception monitoring
```

See [Configuration Reference](../12-configuration-reference/complete-config.md) for complete documentation.

## Launch Modes

### Local Mode

- All components run in the same process
- Suitable for development and debugging
- No Ray dependency
- Limited scalability

```python
config.launch_mode = "local"
controller = NexRLController(config)
controller.run()
```

### Ray Mode

- Components run as distributed Ray actors
- Enables co-location and resource management
- Scales to large clusters
- Production-ready

```python
config.launch_mode = "ray"
ray.init()
controller = ray.remote(NexRLController).remote(config)
ray.get(controller.run.remote())
```

## Extension Points

NexRL is designed for extensibility. Key extension points:

### 1. Custom Rollout Workers

Extend `BaseRolloutWorker` or `BaseNexAURolloutWorker`:

```python
class MyWorker(BaseNexAURolloutWorker):
    def format_task_query(self, data_item):
        # Custom query formatting
        pass
```

### 2. Custom Trainers

Extend `SelfHostedTrainer` or `RemoteApiTrainer`:

```python
class MyTrainer(SelfHostedTrainer):
    def _prepare_batch(self, batch):
        # Custom algorithm logic
        pass
```

### 3. Custom Data Loaders

Extend `BaseDataLoader`:

```python
class MyDataLoader(BaseDataLoader):
    def get_next_item(self):
        # Custom data loading
        pass
```

### 4. Custom Evaluators

Extend `Evaluator`:

```python
class MyEvaluator(Evaluator):
    def evaluate(self, data, evaluation_target):
        # Custom evaluation logic
        pass
```

## Best Practices

### Module Development

1. **Always inherit from NexRLModule** for Ray compatibility
2. **Implement proper cleanup** in stop() methods
3. **Use activity tracking** for long-running operations
4. **Report exceptions** through activity tracker
5. **Implement health_check()** for liveness monitoring

### Resource Management

1. **Define clear resource pool mappings** based on workload
2. **Use co-location** for related services to reduce communication overhead
3. **Monitor resource usage** through activity tracker
4. **Plan GPU allocation** based on model requirements

### Error Handling

1. **Use structured error reporting** through ErrorReporter
2. **Implement proper retry logic** for transient failures
3. **Monitor system health** through activity tracker
4. **Define clear error policies** for different scenarios

### Configuration Management

1. **Use Hydra** for configuration management
2. **Define environment-specific overrides**
3. **Validate configurations** before deployment
4. **Document configuration options** clearly

## Next Steps

- Explore [Controller Details](./controller.md) for orchestration logic
- Learn about [Data Types](./data-types.md) used throughout the system
- Understand [Activity Tracking](./activity-tracking.md) for monitoring
- Review individual component guides for detailed implementation
