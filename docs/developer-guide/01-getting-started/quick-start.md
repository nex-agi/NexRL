# Quick Start Guide

This guide helps you get started with NexRL as quickly as possible.

## Prerequisites

Before starting, ensure you have:
- Python 3.12+
- kubectl configured with access to a Kubernetes cluster
- [Volcano Scheduler](https://github.com/volcano-sh/volcano) installed in the cluster
- High-performance shared storage (e.g., NFS, GPFS)

## Installation

Install NexRL from source:

```bash
git clone https://github.com/nex-agi/NexRL.git
cd NexRL
pip install -e .
```

## Zero-Setup Quick Start

Run immediately with built-in defaults (uses public images and /tmp storage):

```bash
# Self-hosted mode (runs all infrastructure on your cluster)
nexrl -m self-hosted -c recipe/my_task/my_task.yaml --run-nexrl

# Training-service mode (uses external training services like Tinker/Weaver)
nexrl -m training-service -c recipe/my_task/my_task.yaml --run-nexrl
```

The zero-setup mode uses sensible defaults:
- **Docker Images**: Public images from the registry
- **Storage**: `/tmp` directory (not persistent)
- **Configuration**: Built-in default values

**Note**: This is suitable for quick testing but not for production use.

## Minimal Setup for a New Task

Follow these steps to create a minimal working task:

### 1. Create Recipe Directory Structure

```bash
mkdir -p recipe/my_task/agent_workspace
```

### 2. Create Configuration File

Create `recipe/my_task/my_task.yaml`:

```yaml
# Import common configuration
defaults:
  - self_hosted_nexau_common
  - _self_

project_name: "NexRL-MyTask"
experiment_name: "my-task-training"

# Task-specific data
data:
  data_files:
    - "/path/to/train.parquet"

# Rollout worker configuration
rollout_worker:
  type: "nexau"
  nexau_agent_config_path: "recipe/my_task/agent_workspace/agent_config.yaml"
  evaluator_module_path: "recipe/my_task/agent_workspace/evaluator.py:MyEvaluator"
  task_name: "my_task"
```

### 3. Create Agent Configuration

Create `recipe/my_task/agent_workspace/agent_config.yaml`:

```yaml
type: agent
name: my_task_agent
max_context_tokens: 100000
system_prompt: "You are a helpful assistant..."
system_prompt_type: string

llm_config:
  temperature: 0.7
  max_tokens: 8192
  api_type: openai_chat_completion

tracers:
  - import: nexau.archs.tracer.adapters.in_memory:InMemoryTracer
```

### 4. Create Evaluator

Create `recipe/my_task/agent_workspace/evaluator.py`:

```python
from nexrl.rollout_worker import Evaluator, BaseEvaluationTarget, EvaluationRunResult

class MyEvaluator(Evaluator):
    def evaluate(self, data, evaluation_target):
        # Your evaluation logic
        reward = 1.0 if evaluation_target.final_answer == data["answer"] else 0.0
        return EvaluationRunResult(reward=reward)
```

### 5. Run Training

Using the CLI (recommended):

```bash
nexrl -m self-hosted -c recipe/my_task/my_task.yaml --run-nexrl
```

Or using Python directly:

```bash
python -m nexrl.main recipe=my_task/my_task
```

## Verify Installation

Check that the training job is running:

```bash
# List all NexRL resources
kubectl get all -n nexrl

# Monitor driver pod logs
kubectl logs -f -l app=JOB_NAME-driver -n nexrl

# Check rollout worker logs
kubectl logs -l app=JOB_NAME-rollout -n nexrl
```

## Next Steps

Now that you have a basic setup running:

1. **Customize your task**: Modify the evaluator to match your task requirements
2. **Configure deployment**: Set up proper storage, images, and ConfigMaps (see [Configuration Setup](./configuration-setup.md))
3. **Choose deployment mode**: Understand the differences between self-hosted and training-service modes (see [Deployment Modes](./deployment-modes.md))
4. **Implement custom workers**: Create task-specific rollout workers if needed (see [Custom Workers](../05-rollout-workers/custom-workers.md))
5. **Monitor training**: Use W&B or other logging backends to track progress

## Common Quick Start Issues

### Issue: "No module named 'nexrl'"

**Solution**: Ensure you installed NexRL with `pip install -e .` from the repository root.

### Issue: "kubectl: command not found"

**Solution**: Install kubectl and configure it to access your Kubernetes cluster.

### Issue: "Volcano scheduler not found"

**Solution**: Install Volcano scheduler in your cluster:
```bash
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml
```

### Issue: "Permission denied" when accessing storage

**Solution**: Ensure your storage path is accessible from the cluster nodes, or use the default `/tmp` for testing.

## Testing Your Setup

Create a simple test to verify everything works:

```bash
# Use the built-in mock configuration
nexrl -m self-hosted -c recipe/mock/mock.yaml --run-nexrl

# This should run a simple training loop with mock data
# Check logs to verify it's working
kubectl logs -f -l app=mock-driver -n nexrl
```

If the mock task runs successfully, your NexRL installation is working correctly.

## What's Next?

- Learn about [Deployment Modes](./deployment-modes.md) to choose the right setup for your needs
- Configure [Environment Setup](./configuration-setup.md) for production deployments
- Explore the [Core Architecture](../02-core-architecture/overview.md) to understand how NexRL works
- Review [Recipe Structure](../10-recipes/recipe-structure.md) to organize your tasks properly
