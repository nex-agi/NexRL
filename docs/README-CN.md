[中文](docs/README-CN.md) ｜ English

# NexRL

NexRL 是一款生产级、分布式的大语言模型后训练框架，以其极致松耦合的设计理念为核心。其面向服务的架构在保持清晰抽象和易用性的同时，提供了最大的灵活性和可扩展性。

## 核心特性

- **训练即服务 & 推理即服务**：统一的 API 架构通过服务抽象无缝支持不同的训练和推理框架。无需修改代码即可在不同的训练后端（FSDP、Megatron 等）和推理引擎（SGLang、vLLM、TGI 等）之间切换。
- **解耦式模块化架构**：通过定义明确的接口和可扩展组件，实现清晰的职责分离。每个模块独立运行，便于定制化开发和维护。
- **零代码智能体训练支持**：智能体无需进行任何强化学习相关的代码修改，即可无缝集成到强化学习训练流程中。
- **智能资源管理**：支持可配置的服务部署和协同部署策略，在分布式环境中实现最优性能。
- **全面监控能力**：内置活动跟踪和健康检查系统，适配生产环境部署需求。
- **稳健的错误处理**：集中式错误上报和恢复机制，保障生产环境的可靠性。

## 架构设计

NexRL 采用模块化架构，各组件通过明确的接口和 API 进行通信：

![NexRL Architecture](./imgs/nexrl_architecture.png)

**核心组件：**
1. **DataLoader**：提供训练数据（支持自定义数据集）
2. **RolloutWorker**：执行环境交互（你的智能体在此集成！）
3. **TrajectoryPool**：管理轨迹收集和批处理
4. **Trainer**：应用算法逻辑（例如 GRPO）并通过服务 API 协调训练
5. **WeightSyncController**：管理训练和推理之间的模型权重同步

**服务模块：**
1. **推理服务（Inference Service）**：采用标准 OpenAI API 作为与推理引擎交互的统一接口。这种以 API 为中心的设计确保上层模块能以一致的方式与各类推理引擎（如 SGLang、vLLM 等）交互，切换不同推理引擎时无需修改代码。
2. **训练服务（Train Service）**：通过标准化的 forward() 和 forward_backward() API 与不同训练后端（包括 FSDP、Megatron 等）进行通信。为实现多后端兼容，我们为每个后端实现了轻量级适配器，将标准化 API 调用转换为后端特定操作，无需改动核心训练逻辑即可无缝切换训练后端。
3. **智能体服务（Agent Service）**：为智能体参与强化学习训练提供简洁的集成路径。智能体可通过该服务直接将生成的轨迹推入 TrajectoryPool，开发者无需重写或修改智能体代码以适配强化学习训练需求。

## 快速开始

### 前置依赖

- Python 3.12+
- CUDA 12.8+（用于 GPU 支持）
- Ray 2.48+（用于分布式模式）
- 已安装并配置 kubectl
- 可访问 Kubernetes 集群
- 集群中已安装 [Volcano Scheduler](https://github.com/volcano-sh/volcano)
- 高性能网络文件系统，例如 [GPFS](https://en.wikipedia.org/wiki/GPFS)

完整依赖列表请查看 [pyproject](../pyproject.toml)。


### 快速上手

安装 NexRL：

```bash
git clone git@github.com:nex-agi/NexRL.git
cd NexRL
pip install -e .
```

**零配置启动（最快！）**

使用内置默认配置立即运行：

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

使用公共镜像（`nexagi/nexrl:latest`、`lmsysorg/sglang:v0.5.4.post2`）和 `/tmp` 存储 - 非常适合测试！

**开发环境配置**

使用环境变量进行快速配置：

```bash
# 方式 1：使用提供的配置脚本
source cli/setup_env.sh

# 方式 2：手动设置变量
export NEXRL_STORAGE_PATH="/your/persistent/storage"
export NEXRL_WORKER_IMAGE="your-registry/nexrl:tag"
export WANDB_KEY="your-wandb-key"

# 然后运行
nexrl -m self-hosted -c recipe/your_recipe.yaml --run-nexrl
```

**生产环境配置**

使用自定义镜像和持久化存储配置集群：

```bash
# 编辑并应用 ConfigMaps（一次性配置）
kubectl apply -f cli/setup/01-namespace.yaml
kubectl apply -f cli/setup/02-admin-config.yaml  # 先编辑！
kubectl apply -f cli/setup/03-user-config.yaml   # 先编辑！

# 使用生产配置运行
nexrl -m self-hosted \
  -c recipe/single_turn_math_qwen_2a5_7b/single_turn_math_qwen2a5_7b.yaml \
  --run-nexrl --tag prod-v1
```

**CLI 选项：**
- `-m, --mode`：`self-hosted` 或 `training-service`（必需）
- `-c, --train-config`：训练 YAML 配置文件路径（必需）
- `-r, --run-nexrl`：自动启动训练
- `-t, --tag`：自定义任务标签
- `--serving-only`：[self-hosted] 仅启动推理服务
- `--no-serving`：[self-hosted] 跳过推理服务

**配置优先级：**
1. **Kubernetes ConfigMaps**（生产环境）→ `kubectl apply -f cli/setup/`
2. **环境变量**（开发环境）→ `source cli/setup_env.sh` 或 `export NEXRL_*`
3. **内置默认值**（测试）→ 公共镜像、`/tmp` 存储

**关键变量：**
- `NEXRL_STORAGE_PATH`：存储路径（默认：`/tmp/nexrl`）
- `NEXRL_WORKER_IMAGE`：Worker 镜像（默认：`nexagi/nexrl:latest`）
- `NEXRL_CONTROLLER_IMAGE`：Controller 镜像（默认：`nexagi/nexrl:latest`）
- `NEXRL_INFERENCE_IMAGE`：推理镜像（默认：`lmsysorg/sglang:v0.5.4.post2`）
- `WANDB_KEY`：WandB API 密钥（可选）

**另请参阅：** [`cli/README.md`](../cli/README.md) 获取完整文档。

## 文档资源

- **[开发者指南](developer-guide.md)**：包含架构细节、API 说明和高级使用场景的完整文档
- **[配置示例](../recipe/)**：适用于各种模型和任务的即用型训练配方
- **[测试套件](../tests/README.md)**：测试指南和示例代码

## 开源计划

当前发布版本为 NexRL 的基础版本，旨在展示其松耦合与面向服务的核心架构。我们正积极筹备代码开源工作，即将发布更多内容，包括：

- 更多模型与智能体支持
- 更多训练与推理后端集成
- 高性能权重同步机制
- [NexAU](https://github.com/nex-agi/nexau) 高级智能体训练支持
- 后训练算法探索
- 更多易用性工具
- ...


## 许可证

本项目基于 Apache License 2.0 许可证开源 - 详见 [LICENSE](../LICENSE) 文件了解具体条款。

## 致谢

NexRL 致力于实现极致的可扩展性和易用性，充分拥抱开源生态以最小化代码适配成本、提升实验效率。本项目基于多个优秀的开源框架构建，包括 [vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)、[FSDP](https://github.com/pytorch/pytorch)、[Megatron](https://github.com/NVIDIA/Megatron-LM) 以及 [VeRL](https://github.com/volcengine/verl)（FSDP 后端的适配器采用了 VeRL 的实现方案）。此外，智能体服务的零代码开发设计灵感源自 [Agent Lightning](https://github.com/microsoft/agent-lightning)。
