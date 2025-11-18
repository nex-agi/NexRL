[中文](docs/README-CN.md) ｜ English

# NexRL

NexRL is a production-ready, distributed LLM post-training framework defined by its ultra-loosely-coupled philosophy. Its service-oriented architecture provides maximum flexibility and extensibility while maintaining clean abstractions and ease of use.

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
4. **AlgorithmProcessor**: Computes advantages and prepares training batches
5. **TrainWorker**: Coordinates model training through service APIs

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

Install the NexRL repository to enable the CLI:
```
git clone git@github.com:nex-agi/NexRL.git
cd NexRL
pip install -e .
```

Next, ask your cluster maintainer to perform the one-time `admin-setup`:
```
nexrl admin-setup
```

This step saves cluster-shared configurations in Kubernetes and launches cluster-level services such as `train-router` and `rollout-router`.

Once the setup is complete, you can prepare an `rl_train.yaml` configuration file in a folder and launch your job:
```
nexrl launch --job-path /path/to/your/configuration/folder/
```

For detailed configuration examples, please refer to `examples/single_turn_math`.

## Documentation

- **[Developer Guide](docs/developer-guide.md)**: Comprehensive documentation on architecture, APIs, and advanced usage
- **[Configuration Reference](nexrl/config/rl_train.yaml)**: Full configuration options with detailed comments
- **[Test Suite](tests/README.md)**: Testing guide and examples

## More on the Way

This release represents a foundational version of NexRL, designed to demonstrate our loosely-coupled and service-oriented architecture. We are actively working on preparing the code for open source and will release more of our work soon, including:

- More model & agent support
- Additional trainging and inference backend ntegrations
- High-performance weight synchronization
- Advanced agent training support
- Post-training algorithm exploration
- More usability tools
- ...


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowlegement

NexRL aims for ultimate scalability and usability, fully embracing the open-source ecosystem to minimize code adaptation costs and improve experimental efficiency. NexRL is built upon several excellent open-source frameworks, including [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [FSDP](https://github.com/pytorch/pytorch), [Megatron](https://github.com/NVIDIA/Megatron-LM), and [VeRL](https://github.com/volcengine/verl) (the adapter for the FSDP backend adopts the implementation from VeRL). Additionally, the zero-agent code development design of the Agent Service is inspired by [Agent Lightning](https://github.com/microsoft/agent-lightning).
