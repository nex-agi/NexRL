# Deployment Modes

NexRL supports two deployment modes via the `cli/` launcher, each optimized for different use cases.

## Overview

| Feature | Self-Hosted Mode | Training-Service Mode |
|---------|------------------|----------------------|
| **Training Infrastructure** | Runs on your cluster | Uses external service (Tinker/Weaver) |
| **Inference Service** | Runs on your cluster | Runs on your cluster |
| **GPU Requirements** | Needs GPUs for training | Only needs GPUs for inference |
| **Control** | Full control | Service-managed |
| **Setup Complexity** | Higher | Lower |
| **Resource Footprint** | Larger | Smaller |

## Self-Hosted Mode

Runs all training infrastructure (training backend, inference service) on your Kubernetes cluster. Provides full control over the entire stack.

### Command

```bash
nexrl --mode self-hosted --train-config recipe/my_task.yaml --run-nexrl
```

### Components Launched

1. **Training API Server and GPU Workers**
   - API server for coordinating training
   - Multiple GPU workers for distributed training
   - Checkpoint management
   - Weight synchronization

2. **Inference Service**
   - SGLang or vLLM for LLM serving
   - Model loading and caching
   - Request batching and optimization

3. **NexRL Driver Pod**
   - Main controller
   - Rollout workers
   - Trajectory pool
   - Weight synchronization coordinator

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
│                                                              │
│  ┌──────────────┐         ┌─────────────────┐              │
│  │   Driver     │         │  Training       │              │
│  │   Pod        │◄────────┤  Service        │              │
│  │              │         │  (API + Workers)│              │
│  │  - Controller│         └─────────────────┘              │
│  │  - Rollout   │                                           │
│  │  - Workers   │         ┌─────────────────┐              │
│  │              │◄────────┤  Inference      │              │
│  └──────────────┘         │  Service        │              │
│                           │  (SGLang/vLLM)  │              │
│                           └─────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Use Cases

Use self-hosted mode when:
- You want full control over training infrastructure
- You have sufficient GPU resources available
- You need to customize training backend configuration
- You want to run everything in your own environment
- You need specific checkpoint formats or storage
- You're developing new training algorithms

### Configuration Example

```yaml
# recipe/my_task/my_task.yaml
defaults:
  - self_hosted_nexau_common
  - _self_

project_name: "NexRL-MyTask"
experiment_name: "my-task-v1"

# Training service configuration
service:
  train_service:
    backend: "nextrainer"
    url: "http://train-service:8000"
    model_tag: "policy"
    world_size: 4  # Number of GPUs

  inference_service:
    backend: "vllm"
    url: "http://inference-service:8001"
    model_tag: "policy"

# Trainer configuration
trainer:
  type: "self_hosted_grpo"
  total_train_steps: 1000
  checkpoint_path: "/path/to/checkpoints"
  sync_weight_path: "/path/to/sync_weights"
  save_freq: 100

# Algorithm configuration (required for self-hosted)
algorithm:
  type: "grpo"
  batch_size: 32
  do_old_log_prob_compute: true
  use_kl_in_reward: false
```

### Resource Requirements

**Minimum requirements for small-scale training:**
- Driver Pod: 8 CPU cores, 32GB RAM
- Training Service: 4 GPUs (A100 or equivalent), 16 CPU cores, 128GB RAM
- Inference Service: 2 GPUs, 8 CPU cores, 64GB RAM
- Shared Storage: High-performance NFS/GPFS

## Training-Service Mode

Uses external training services (Tinker/Weaver). Only launches the NexRL driver pod and inference service locally.

### Command

```bash
nexrl --mode training-service --train-config recipe/my_task.yaml --run-nexrl
```

### Components Launched

1. **NexRL Driver Pod**
   - Main controller
   - Rollout workers
   - Trajectory pool
   - Remote API trainer

2. **Inference Service**
   - SGLang or vLLM for LLM serving
   - Model loading and caching

**External (not launched):**
- Training service (Tinker or Weaver API)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
│                                                              │
│  ┌──────────────┐                                           │
│  │   Driver     │         ┌─────────────────┐              │
│  │   Pod        │◄────────┤  Inference      │              │
│  │              │         │  Service        │              │
│  │  - Controller│         │  (SGLang/vLLM)  │              │
│  │  - Rollout   │         └─────────────────┘              │
│  │  - Workers   │                                           │
│  │              │                                           │
│  └──────┬───────┘                                           │
│         │                                                   │
└─────────┼───────────────────────────────────────────────────┘
          │
          │ HTTPS/gRPC
          ▼
┌─────────────────────┐
│  External Training  │
│  Service            │
│  (Tinker/Weaver)    │
└─────────────────────┘
```

### Use Cases

Use training-service mode when:
- You want to use managed training services
- You want lighter resource footprint on your cluster
- You have access to Tinker or Weaver training APIs
- You don't need to modify the training backend
- You want faster setup with less infrastructure management
- You're focused on task development, not training infrastructure

### Configuration Example

**For Tinker:**

```yaml
# recipe/my_task/tinker.yaml
defaults:
  - tinker_nexau_common
  - _self_

project_name: "NexRL-MyTask"
experiment_name: "my-task-tinker"

# Training service configuration
service:
  train_service:
    backend: "tinker"
    config:
      loss_fn: "importance_sampling"
      learning_rate: 2e-6
      beta1: 0.9
      beta2: 0.95
      eps: 1e-8

  tinker_service:
    lora_rank: 32
    api_key: "your-tinker-api-key"

  inference_service:
    backend: "vllm"
    url: "http://inference-service:8001"

# Trainer configuration
trainer:
  type: "remote_api_grpo"
  total_train_steps: 1000
  max_prompt_length: 15000
  max_response_length: 13000
```

**For Weaver:**

```yaml
# recipe/my_task/weaver.yaml
defaults:
  - weaver_nexau_common
  - _self_

project_name: "NexRL-MyTask"
experiment_name: "my-task-weaver"

# Training service configuration
service:
  train_service:
    backend: "weaver"
    config:
      loss_fn: "importance_sampling"
      learning_rate: 2e-6

  weaver_service:
    api_key: "your-weaver-api-key"

  inference_service:
    backend: "vllm"
    url: "http://inference-service:8001"

# Trainer configuration
trainer:
  type: "remote_api_grpo"
  total_train_steps: 1000
```

### Resource Requirements

**Minimum requirements:**
- Driver Pod: 8 CPU cores, 32GB RAM
- Inference Service: 2 GPUs, 8 CPU cores, 64GB RAM
- Shared Storage: Standard NFS/GPFS (lighter requirements)

**Note**: Training GPUs are provided by the external service.

## CLI Options Reference

```bash
nexrl [OPTIONS]

Required:
  -m, --mode              Deployment mode: 'self-hosted' or 'training-service'
  -c, --train-config      Path to training config YAML

Optional:
  -r, --run-nexrl         Auto-start training immediately
  -t, --tag               Custom tag for job names
  --inference-url URL     [self-hosted] Use existing inference service (skip launching)
```

## Examples

### Self-Hosted with Auto-Start

```bash
nexrl -m self-hosted -c recipe/math/config.yaml -r
```

### Training-Service with Custom Tag

```bash
nexrl -m training-service -c recipe/math/tinker.yaml -r -t exp-v2
```

### Self-Hosted with External Inference Service

If you already have an inference service running, skip launching a new one:

```bash
nexrl -m self-hosted -c recipe/math/config.yaml -r --inference-url my-service:8000
```

This is useful for:
- Sharing inference services across multiple experiments
- Using pre-warmed inference services
- Debugging inference separately from training

## Choosing the Right Mode

### Choose Self-Hosted If:

✅ You need full control over the training process
✅ You want to implement custom training algorithms
✅ You have dedicated GPU resources
✅ You need specific checkpoint formats or storage locations
✅ You want to optimize training infrastructure for your workload
✅ You're developing new features for the training backend

### Choose Training-Service If:

✅ You want to focus on task development
✅ You have access to managed training services
✅ You want simpler infrastructure management
✅ You have limited GPU resources locally
✅ You prefer using battle-tested training services
✅ You want faster iteration on experiments

## Switching Between Modes

You can switch between modes by changing your configuration and using different base configs:

```yaml
# Self-hosted
defaults:
  - self_hosted_nexau_common
  - _self_

trainer:
  type: "self_hosted_grpo"

algorithm:
  type: "grpo"
  # algorithm config...
```

```yaml
# Training-service (Tinker)
defaults:
  - tinker_nexau_common
  - _self_

trainer:
  type: "remote_api_grpo"

# No algorithm section needed
```

## Next Steps

- Learn about [Configuration Setup](./configuration-setup.md) for production deployments
- Understand [Trainer Architecture](../06-trainers/overview.md) to see implementation differences
- Explore [Service Integration](../08-services/training-service.md) for backend details
- Review [Best Practices](../13-best-practices/module-development.md) for development
