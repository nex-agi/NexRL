# Debug Data Dump 使用指南

## 1. 启用 Debug Data Dump

### 1.1 Self-hosted Mode

在 YAML 配置文件中添加 `debug` 节：

```yaml
debug:
  enable_data_dump: true
  dump_dir: "${oc.env:EXPERIMENT_PATH,./debug_dump}/debug_dump"
  dump_format: "pt"  # pt 或 jsonl
  dump_every_n_steps: 1

  # Self-hosted mode 数据类型
  dump_options:
    trajectory: true       # Rollout trajectories
    old_log_probs: true    # GRPO old log probabilities
    forward_data: true     # Forward pass logprobs/entropy
    loss: true             # Loss values
    param: false           # Parameter (需要 use_orig_params)
    gradient: false        # Gradient (需要 use_orig_params)

  # Param/gradient dump 需要指定参数名 pattern
  target_param_pattern: "model\\.layers\\.0\\.self_attn\\.q_proj\\.weight"
```

### 1.2 Remote API Mode

使用 Weaver 远程训练服务时，配置 `remote_api_dump_options`：

```yaml
debug:
  enable_data_dump: true
  dump_dir: "${oc.env:EXPERIMENT_PATH}/debug_dump"
  dump_every_n_steps: 1

  # Remote API mode 数据类型
  remote_api_dump_options:
    prepared_trajectories: true   # GRPO advantage computation results
    datums: true                  # Converted datums sent to Weaver
    training_metrics: true        # Metrics returned from Weaver
```

**Remote API Mode 数据类型说明**：
- `prepared_trajectories`: GRPO 优势值计算结果（包含 advantages, logprobs 等）
- `datums`: 发送给 Weaver 的训练数据（Datum 格式）
- `training_metrics`: Weaver 返回的训练指标（loss, tokens 等）

## 2. 从 Dump 文件加载 Trajectory 训练

修改子配置文件覆盖 `rollout_worker`：

```yaml
# single_turn_math_qwen3_8b.yaml
defaults:
  - single_turn_math_qwen2a5_7b  # 继承基础配置
  - _self_

rollout_worker:
  type: "mock"
  need_llm_inference: false
  trajectory_load_path: "/path/to/debug_dump/trajectory/step_000000.pt"
  trajectory_format: "pt"
```

**注意**：Mock worker 根据 task 的 `(group_id, run_id)` 匹配对应 trajectory。

## 3. 完整示例

参考 `NexRL-recipes/production/single_turn_math_qwen2a5_7b/single_turn_math_qwen3_8b.yaml`：

```yaml
hydra:
  searchpath:
    - file://./

defaults:
  - single_turn_math_qwen2a5_7b
  - _self_

...

# Mock rollout worker 从 dump 文件加载
rollout_worker:
  type: "mock"
  need_llm_inference: false
  trajectory_load_path: "NexRL/logs/baseline-single_turn_math_qwen3_8b/20260126-075017/debug_dump/trajectory/step_000000.pt"
  trajectory_format: "pt"
```

## 4. Dump 目录结构

### 4.1 Self-hosted Mode

```
debug_dump/
├── trajectory/          # Rollout trajectories
│   └── step_000000.pt
├── old_log_probs/       # GRPO old log probabilities
├── forward_data/        # Forward pass data
├── loss/                # Loss values
├── param/               # Parameters (if enabled)
└── gradient/            # Gradients (if enabled)
```

### 4.2 Remote API Mode

```
debug_dump/
├── prepared_trajectories/   # GRPO advantage computation results
│   └── step_000000.pt
├── datums/                  # Datums sent to Weaver
│   └── step_000000.pt
└── training_metrics/        # Metrics from Weaver
    └── step_000000.pt
```
