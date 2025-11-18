# NexRL

NexRL 是一款极致松耦合的大语言模型后训练框架。它将后训练过程中的组件进行了充分解耦，并对训练和推理引擎进行了服务化封装，组件和服务间通过统一接口交互，以充分拥抱开源生态，在保持清晰抽象和易用性的同时，提供了极高的灵活性和可扩展性。

## 核心特性

- **训练即服务 & 推理即服务**：统一的 API 架构通过服务抽象，无缝支持不同的训练和推理框架。无需修改代码，即可切换训练后端（FSDP、Megatron 等）和推理引擎（SGLang、vLLM等）。

- **解耦式模块化架构**：通过定义明确的接口和可扩展组件，实现清晰的职责分离。每个模块独立运行，便于定制化开发和维护。

- **零代码智能体训练支持**：智能体无需进行任何强化学习相关的代码修改，即可无缝集成到强化学习训练流程中。

- **智能资源管理**：支持可配置的服务部署和协同部署策略，在分布式环境中实现最优性能。

- **全面监控能力**：内置活动跟踪和健康检查系统，适配生产环境部署需求。

- **稳健的错误处理**：集中式错误上报和恢复机制，保障生产环境的可靠性。

## 架构设计

NexRL 采用模块化架构，各组件通过明确的接口和 API 进行通信：

![NexRL Architecture](./imgs/nexrl_architecture.png)

**核心组件：**

1. **NexRLController**：协调管理整个训练流水线

2. **DataLoader**：提供训练数据（支持自定义数据集）

3. **RolloutWorker**：执行环境交互（你的智能体在此集成！）

4. **TrajectoryPool**：管理轨迹收集和批处理

5. **AlgorithmProcessor**：计算优势函数并准备训练批次数据

6. **TrainWorker**：通过服务 API 协调模型训练过程

**服务模块：**

1. **推理服务（Inference Service）**：采用标准 OpenAI API 作为与推理引擎交互的统一接口。这种以 API 为中心的设计，确保上层模块能以一致的方式与各类推理引擎（如 SGLang、vLLM、TGI 等）交互，切换不同推理引擎时无需修改代码。

2. **训练服务（Train Service）**：通过标准化的 forward() 和 forward_backward() API，与不同训练后端（包括 FSDP、Megatron 等）进行通信。为实现多后端兼容，我们为每个后端实现了轻量级适配器，将标准化 API 调用转换为后端特定操作，无需改动核心训练逻辑即可无缝切换训练后端。

3. **智能体服务（Agent Service）**：为智能体参与强化学习训练提供简洁的集成路径。智能体可通过该服务直接将生成的轨迹推入 TrajectoryPool，开发者无需重写或修改智能体代码以适配强化学习训练需求。

## 快速开始

### 前置依赖

- Python 3.12+
- CUDA 12.8+
- Ray 2.48+
- 已安装并配置 kubectl
- 可访问 Kubernetes 集群
- 集群中已安装 [Volcano Scheduler](https://github.com/volcano-sh/volcano)
- 高性能网络文件系统，例如 [GPFS](https://en.wikipedia.org/wiki/GPFS)

完整依赖列表请查看 [pyproject](../pyproject.toml)。

### 快速上手

安装 NexRL 仓库以启用 CLI：
```bash
git clone git@github.com:nex-agi/NexRL.git
cd NexRL
pip install -e .
```

接下来，请集群管理员执行一次性 `admin-setup`：
```bash
nexrl admin-setup
```

此步骤会在 Kubernetes 中保存集群共享配置，并启动集群级服务，如 `train-router` 和 `rollout-router`。

设置完成后，您可以在文件夹中准备 `rl_train.yaml` 配置文件并启动任务：
```bash
nexrl launch --job-path /path/to/your/configuration/folder/
```

详细配置示例请参考 `examples/single_turn_math`。

## 文档资源

- **[开发者指南](developer-guide.md)**：包含架构细节、API 说明和高级使用场景的完整文档

- **[配置参考](../nexrl/config/rl_train.yaml)**：完整配置选项及详细注释说明

- **[测试套件](../tests/README.md)**：测试指南和示例代码

## 开源计划

当前发布版本为 NexRL 的基础版本，旨在展示其松耦合与面向服务的核心架构。我们正积极筹备代码开源工作，即将发布更多内容，包括：
- 更多模型与智能体支持
- 更多训练与推理后端集成
- 高性能权重同步机制
- 端到端智能体训练支持
- 后训练算法探索
- ...

## 许可证

本项目基于 Apache License 2.0 许可证开源 - 详见 [LICENSE](LICENSE) 文件了解具体条款。

## 致谢

NexRL 致力于实现极致的可扩展性和易用性，充分拥抱开源生态以最小化代码适配成本、提升实验效率。本项目基于多个优秀的开源框架构建，包括 [vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)、[FSDP](https://github.com/pytorch/pytorch)、[Megatron](https://github.com/NVIDIA/Megatron-LM) 以及 [VeRL](https://github.com/volcengine/verl)（FSDP 后端的适配器采用了 VeRL 的实现方案）。此外，智能体服务的零代码开发设计灵感源自 [Agent Lightning](https://github.com/microsoft/agent-lightning)。
