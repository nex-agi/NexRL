# Debug Data Dump 使用指南

## 1. 启用 Debug Data Dump

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

project_name: "NexRL-Weaver"
experiment_name: "baseline-single_turn_math_qwen3_8b"

resource:
  inference:
    served_model_name: "nexrl-rollout-qwen3-8b-math-tangtian"
    model_path: "/gpfs/models/huggingface.co/Qwen/Qwen3-8B"

# Mock rollout worker 从 dump 文件加载
rollout_worker:
  type: "mock"
  need_llm_inference: false
  trajectory_load_path: "/gpfs/users/tangtian/Weaver/NexRL/logs/baseline-single_turn_math_qwen3_8b/20260126-075017/debug_dump/trajectory/step_000000.pt"
  trajectory_format: "pt"
```

## 4. Dump 目录结构

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
