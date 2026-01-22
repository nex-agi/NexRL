[中文](docs/README-CN.md) | English | [Blog](https://dawning-road.github.io/blog/nexrl)

# NexRL

NexRL is a production-ready, distributed LLM post-training framework defined by its ultra-loosely-coupled philosophy. Its service-oriented architecture provides maximum flexibility and extensibility while maintaining clean abstractions and ease of use.

## News 🚀

- **[2026.01.15] NexRL v1.0.0 is here!** Train [NexAU](https://github.com/nex-agi/nexau) agents with **zero code modification**—just configs and evaluators. New training-service mode supports [Weaver](https://weaver.nex-agi.com/) and [Tinker](https://thinkingmachines.ai/tinker/) APIs for effortless cloud training.
- **[2025.11.18]** NexRL goes open-source! Pre-release version now available.

## Key Features

- **Training-as-a-Service & Rollout-as-a-Service**: Unified API architecture that seamlessly supports different training and inference frameworks through service abstraction. Switch between training backends (FSDP, Megatron, etc.) and inference engines (SGLang, vLLM, TGI, etc.) without modifying your code.
- **Decoupled Modular Architecture**: Clean separation of concerns with well-defined interfaces and extensible components. Each module operates independently, enabling easy customization and maintenance.
- **Zero-Code Agent-Training Support**: Agents can seamlessly integrate with RL training without any RL-specific code modifications.
- **Intelligent Resource Management**: Configurable placement and co-location of services for optimal performance in distributed environments
- **Comprehensive Monitoring**: Built-in activity tracking and health checking system for production deployments
- **Robust Error Handling**: Centralized error reporting and recovery mechanisms for production reliability


## Architecture

NexRL follows a modular architecture where components communicate through explicit interfaces and APIs:

![NexRL Architecture](./docs/imgs/nexrl_architecture.png)

**Core Components:**
1. **DataLoader**: Provides training data (supports custom datasets)
2. **RolloutWorker**: Executes environment interactions (your agent goes here!)
3. **TrajectoryPool**: Manages trajectory collection and batching
4. **Trainer**: Applies algorithm logic (e.g., GRPO) and coordinates training through service APIs
5. **WeightSyncController**: Manages model weight synchronization between training and inference

**Services:**
1. **Inference Service**: Adopts the standard OpenAI API as the unified interaction interface with inference engines. This API-centric design ensures that the upper-layer modules can interact with various inference engines (such as SGLang, vLLM, etc.) in a consistent manner, eliminating the need for code modifications when switching between different inference engines.
2. **Train Service**: Utilizes standardized forward() and forward_backward() APIs to communicate with different training backends (including FSDP, Megatron, etc.). To achieve compatibility with diverse backends, we implement lightweight adapters tailored for each backend. These adapters translate the standardized API calls into backend-specific operations, enabling seamless switching of training backends without altering the core training logic.
3. **Agent Service**: Provides a streamlined integration path for agents to participate in RL training. Agents can directly push generated trajectories into the TrajectoryPool through this service, eliminating the need for developers to rewrite or modify agent code to adapt to RL training requirements.


## Getting Started

### Prerequisites

- Python 3.12+
- CUDA 12.8+ (for GPU support)
- Ray 2.48+ (for distributed mode)
- kubectl installed and configured
- Access to a Kubernetes cluster
- [Volcano Scheduler](https://github.com/volcano-sh/volcano) installed in the cluster
- High-performance network file system, e.g., [GPFS](https://en.wikipedia.org/wiki/GPFS)

Check [pyproject](pyproject.toml) for the full dependency list.


### Quick Start

Install NexRL:

```bash
git clone git@github.com:nex-agi/NexRL.git
cd NexRL
pip install -e .
```

**Zero-Setup (Quickest!)**

Run immediately with built-in defaults:

```bash
nexrl -m self-hosted \
  -c recipe/math/self_hosted.yaml \
  --run-nexrl
```

```bash
nexrl -m training-service \
  -c recipe/math/tinker.yaml \
  --run-nexrl
```

Uses public images (`nexagi/nexrl:v1.0.0`, `lmsysorg/sglang:v0.5.4.post2`) and `/tmp` storage - perfect for testing!

**Development Setup**

Use environment variables for quick configuration:

```bash
# Option 1: Use the provided setup script
source cli/setup_env.sh

# Option 2: Set variables manually
export NEXRL_STORAGE_PATH="/your/persistent/storage"
export NEXRL_WORKER_IMAGE="your-registry/nexrl:tag"
export WANDB_KEY="your-wandb-key"

# Then run
nexrl -m self-hosted -c recipe/your_recipe.yaml --run-nexrl
```

**Production Setup**

Configure cluster with custom images and persistent storage:

```bash
# Edit and apply ConfigMaps (one-time setup)
kubectl apply -f cli/setup/01-namespace.yaml
kubectl apply -f cli/setup/02-admin-config.yaml  # Edit first!
kubectl apply -f cli/setup/03-user-config.yaml   # Edit first!

# Run with production config
nexrl -m self-hosted \
  -c recipe/single_turn_math_qwen_2a5_7b/single_turn_math_qwen2a5_7b.yaml \
  --run-nexrl --tag prod-v1
```

**CLI Options:**
- `-m, --mode`: `self-hosted` or `training-service` (required)
- `-c, --train-config`: Path to training YAML (required)
- `-r, --run-nexrl`: Auto-start training
- `-t, --tag`: Custom job tag
- `--serving-only`: [self-hosted] Only launch inference
- `--no-serving`: [self-hosted] Skip inference

**Configuration Priority:**
1. **Kubernetes ConfigMaps** (production) → `kubectl apply -f cli/setup/`
2. **Environment Variables** (development) → `source cli/setup_env.sh` or `export NEXRL_*`
3. **Built-in Defaults** (testing) → public images, `/tmp` storage

**Key Variables:**
- `NEXRL_STORAGE_PATH`: Storage path (default: `/tmp/nexrl`)
- `NEXRL_WORKER_IMAGE`: Worker image (default: `nexagi/nexrl:v1.0.0`)
- `NEXRL_CONTROLLER_IMAGE`: Controller image (default: `nexagi/nexrl:v1.0.0`)
- `NEXRL_INFERENCE_IMAGE`: Inference image (default: `lmsysorg/sglang:v0.5.4.post2`)
- `WANDB_KEY`: WandB API key (optional)

**See also:** [`cli/README.md`](cli/README.md) for comprehensive documentation.

## Documentation

- **[User Guide](docs/user-guide.md)**: Complete guide for developing and integrating RL algorithms. Train NexAU agents with zero code modification—just provide configuration files and task-specific evaluators.
- **[Developer Guide](docs/developer-guide)**: Comprehensive documentation on architecture, APIs, and advanced usage
- **[Configuration Examples](recipe/)**: Ready-to-use training recipes for various models and tasks
- **[Test Suite](tests/README.md)**: Testing guide and examples

## More on the Way

This release represents a foundational version of NexRL, designed to demonstrate our loosely-coupled and service-oriented architecture. We are actively working on preparing the code for open source and will release more of our work soon, including:

- More model & agent support
- Additional trainging and inference backend ntegrations
- High-performance weight synchronization
- Post-training algorithm exploration
- More usability tools
- ...

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowlegement

NexRL aims for ultimate scalability and usability, fully embracing the open-source ecosystem to minimize code adaptation costs and improve experimental efficiency. NexRL is built upon several excellent open-source frameworks, including [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [FSDP](https://github.com/pytorch/pytorch), [Megatron](https://github.com/NVIDIA/Megatron-LM), and [VeRL](https://github.com/volcengine/verl) (the adapter for the FSDP backend adopts the implementation from VeRL). Additionally, the zero-agent code development design of the Agent Service is inspired by [Agent Lightning](https://github.com/microsoft/agent-lightning).
