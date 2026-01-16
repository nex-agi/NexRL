# NexRL Developer Guide

Complete developer documentation for the NexRL distributed reinforcement learning framework.

## 📚 Documentation Structure

### ✅ 01. Getting Started
Quickstart guides and configuration setup.

- [Quick Start](./01-getting-started/quick-start.md) - Installation, zero-setup, minimal setup
- [Deployment Modes](./01-getting-started/deployment-modes.md) - Self-hosted vs Training-service
- [Configuration Setup](./01-getting-started/configuration-setup.md) - Environment variables, ConfigMaps

### ✅ 02. Core Architecture
System architecture and foundational components.

- [Overview](./02-core-architecture/overview.md) - Architecture diagram, component overview
- [Controller](./02-core-architecture/controller.md) - NexRLController, module registry, lifecycle
- [Data Types](./02-core-architecture/data-types.md) - Core types (Trajectory, Batch, etc.)
- [Activity Tracking](./02-core-architecture/activity-tracking.md) - Monitoring and logging

### ✅ 03. Data Management
Data loading and management.

- [Data Loader](./03-data-loader/data-loader.md) - BaseDataLoader, TorchDataLoader, custom loaders

### ✅ 04. Trajectory Pool
Trajectory collection and batching.

- [Trajectory Pool](./04-trajectory-pool/trajectory-pool.md) - TrajectoryPool, store types, grouping strategies

### ✅ 05. Rollout Workers
Rollout worker architecture and implementation.

- [Overview](./05-rollout-workers/overview.md) - Architecture, workflow, worker hierarchy
- [Base Rollout Worker](./05-rollout-workers/base-rollout-worker.md) - Core interface, lifecycle, methods
- [NexAU Rollout Worker](./05-rollout-workers/nexau-rollout-worker.md) - Agent integration, trace processing
- [Custom Workers](./05-rollout-workers/custom-workers.md) - Creating task-specific workers
- [Evaluators](./05-rollout-workers/evaluators.md) - Evaluation patterns and best practices

### ✅ 06. Trainers
Trainer architecture and algorithm integration.

- [Overview](./06-trainers/overview.md) - Trainer architecture hierarchy, BaseTrainer
- [Self-Hosted Trainers](./06-trainers/self-hosted-trainers.md) - SelfHostedTrainer, SelfHostedGrpoTrainer
- [Remote API Trainers](./06-trainers/remote-api-trainers.md) - RemoteApiTrainer, GRPO, CrossEntropy
- [Custom Trainers](./06-trainers/custom-trainers.md) - Creating custom algorithm trainers

### ✅ 07. Services
Service integration and backends.

- [Inference Service](./07-services/inference-service.md) - LLM service integration, OpenAI client
- [Training Service](./07-services/training-service.md) - TrainServiceClient, NexTrainer backend
- [Service Holders](./07-services/service-holders.md) - Tinker/Weaver service holders

### ✅ 08. Features
Cross-cutting features.

- [Weight Synchronization](./08-features/weight-synchronization.md) - WeightSyncController, sync modes, coordination
- [Validation](./08-features/validation.md) - Validator, validation cycles, metrics computation
- [Checkpointing](./08-features/checkpointing.md) - Checkpoint management, resume modes, loading
- [Error Handling](./08-features/error-handling.md) - ErrorReporter, health checking, error policies

### ✅ 09. Recipes
Recipe development and structure.

- [Recipe Structure](./09-recipes/recipe-structure.md) - Directory layout, file organization, best practices
- [Agent Configuration](./09-recipes/agent-configuration.md) - NexAU agent config, system prompts, tools
- [Recipe Configuration](./09-recipes/recipe-configuration.md) - Main YAML config, Hydra composition, inheritance
- [Environment Setup](./09-recipes/environment-setup.md) - Environment scripts, dependencies, workspace

### ✅ 10. Distributed Execution
Ray integration and resource management.

- [Ray Integration](./10-distributed-execution/ray-integration.md) - Ray resource management, actors, execution modes
- [Colocation](./10-distributed-execution/colocation.md) - Actor colocation patterns, RayActorWrapper
- [Resource Allocation](./10-distributed-execution/resource-allocation.md) - GPU/CPU allocation, optimization strategies

### ✅ 11. Configuration Reference
Complete configuration documentation.

- [Complete Config](./11-configuration-reference/complete-config.md) - Full configuration example with Hydra composition
- [Data Config](./11-configuration-reference/data-config.md) - Data loader configuration options
- [Rollout Config](./11-configuration-reference/rollout-config.md) - Rollout worker configuration
- [Trainer Config](./11-configuration-reference/trainer-config.md) - Trainer configuration by type
- [Service Config](./11-configuration-reference/service-config.md) - Inference and training service configuration

### 🚧 12. Best Practices
Development guidelines and patterns.

- Module Development - Writing NexRL modules
- Recipe Development - Creating recipes


## 🚀 Quick Navigation

### For New Users
1. Start with [Quick Start](./01-getting-started/quick-start.md)
2. Understand [Deployment Modes](./01-getting-started/deployment-modes.md)
3. Review [Architecture Overview](./02-core-architecture/overview.md)

### For Task Developers
1. Review [Recipes](./09-recipes/recipe-structure.md)
2. Learn about [Rollout Workers](./05-rollout-workers/overview.md)
3. Implement [Evaluators](./05-rollout-workers/evaluators.md)
4. Study [Examples](./13-examples/simple-task.md)

### For Algorithm Developers
1. Understand [Trainer Architecture](./06-trainers/overview.md)
2. Review [GRPO Implementation](./06-trainers/self-hosted-trainers.md) and [Remote API GRPO](./06-trainers/remote-api-trainers.md)
3. Learn about [Custom Trainers](./06-trainers/custom-trainers.md)
4. Explore [Training Services](./07-services/training-service.md)

### For System Administrators
1. Set up [Configuration](./01-getting-started/configuration-setup.md)
2. Understand [Ray Integration](./10-distributed-execution/ray-integration.md)
3. Learn about [Resource Allocation](./10-distributed-execution/resource-allocation.md)
## 📖 Key Concepts

### Architecture
- **Modular Design**: Clean separation of concerns
- **Service Abstraction**: Unified API for different backends
- **Activity Tracking**: Comprehensive monitoring system
- **Resource Management**: Intelligent co-location and allocation

### Data Flow
```
DataLoader → RolloutWorker → TrajectoryPool → Trainer
                ↓
            Evaluator
```

### Weight Synchronization
```
Trainer → WeightSync → InferenceService
             ↓
         Validator
```

### Lifecycle
```
Controller → Initialize → Run → Monitor → Stop
                ↓
         [ Training Loop ]
```

## 🔧 Development Workflow

### Setting Up a New Task

1. **Create recipe structure**:
   ```bash
   mkdir -p recipe/my_task/agent_workspace
   ```

2. **Configure task**:
   - Create `my_task.yaml` (recipe config)
   - Create `agent_config.yaml` (agent config)
   - Create `evaluator.py` (task evaluator)

3. **Run training**:
   ```bash
   nexrl -m self-hosted -c recipe/my_task/my_task.yaml --run-nexrl
   ```

### Implementing a Custom Worker

1. **Create worker file** in recipe:
   ```python
   from nexrl.rollout_worker import BaseNexAURolloutWorker

   class MyWorker(BaseNexAURolloutWorker):
       def format_task_query(self, data_item):
           # Custom formatting
           pass
   ```

2. **Configure in recipe**:
   ```yaml
   rollout_worker:
     custom_rollout_worker_module_path: "recipe/my_task/my_worker.py"
     custom_rollout_worker_class_name: "MyWorker"
   ```

### Implementing a Custom Trainer

1. **Create trainer class**:
   ```python
   from nexrl.trainer import SelfHostedTrainer

   class MyTrainer(SelfHostedTrainer):
       def _prepare_batch(self, batch):
           # Custom algorithm
           pass
   ```

2. **Register in controller**:
   ```python
   MODULE_REGISTRY[NexRLRole.TRAINER]["my_trainer"] = MyTrainer
   ```

## 📝 Contributing

When adding new features:

1. **Update relevant docs** in appropriate section
2. **Add examples** to demonstrate usage
3. **Update configuration reference** if new config added
4. **Add to best practices** if introducing new patterns

## 🔗 Related Documentation

- [Main README](../../README.md) - Project overview
- [User Guide](../user-guide.md) - End-to-end tutorial
- [API Reference](../../api/) - Auto-generated API docs

## ❓ Getting Help

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions
- **Examples**: See `recipe/` directory for complete examples
- **Tests**: See `tests/` directory for usage examples

## Legend

- ✅ Complete documentation
- 🚧 Work in progress / Placeholder
- 📝 Needs review/update

---

**Last Updated**: 2026-01-16
