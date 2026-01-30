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

### 4.3 Weaver-trainer Mode

weaver-trainer 使用环境变量配置 debug dump：

```bash
export WEAVER_DEBUG_DUMP_ENABLED=1
export WEAVER_DEBUG_DUMP_DIR=./weaver_debug_dump
export WEAVER_DEBUG_DUMP_EVERY_N=1
export WEAVER_DEBUG_DUMP_OPTIONS="forward_data,loss,is_loss_debug"
```

目录结构：

```
weaver_debug_dump/
├── forward_data/           # Forward pass logprobs/entropy
│   ├── step_000000_rank0_micro0.pt
│   ├── step_000000_rank0_micro1.pt
│   ├── step_000000_rank1_micro0.pt
│   └── ...
├── loss/                   # Loss values
│   ├── step_000000_rank0_micro0.pt
│   └── ...
├── is_loss_debug/          # Detailed IS loss computation
│   ├── step_000000_rank0_micro0.pt
│   └── ...
├── batch_inputs/           # Model inputs before forward
├── param/                  # Parameters (if enabled)
└── gradient/               # Gradients (if enabled)
```

## 5. 比较 Debug Dumps

使用 `compare_debug_dumps.py` 脚本比较 self-hosted 和 weaver-trainer 的 dump 数据：

### 5.1 基本用法

```bash
python nexrl/utils/compare_debug_dumps.py \
  --self_hosted_dir /path/to/self_hosted/debug_dump \
  --weaver_dir /path/to/weaver-trainer/debug_dump \
  --step 0 \
  --output comparison_report.json
```

### 5.2 完整参数

```bash
python nexrl/utils/compare_debug_dumps.py \
  --self_hosted_dir /path/to/self_hosted/debug_dump \
  --remote_api_dir /path/to/remote_api/debug_dump \
  --weaver_dir /path/to/weaver-trainer/debug_dump \
  --step 0 \
  --num_ranks 8 \
  --num_micros 2 \
  --output comparison_report.json
```

**参数说明**：
- `--self_hosted_dir`: Self-hosted mode 的 debug dump 目录（必需）
- `--remote_api_dir`: Remote API mode 的 debug dump 目录（可选）
- `--weaver_dir`: weaver-trainer 的 debug dump 目录（可选）
- `--step`: 要比较的训练步数（默认：1）
- `--num_ranks`: 要比较的 rank 数量（默认：8）
- `--num_micros`: 要比较的 microbatch 数量（默认：2）
- `--output`: 输出 JSON 报告文件路径（可选）

### 5.3 比较内容

脚本会比较以下数据（跨所有 ranks 和 microbatches）：

1. **Trajectories**: Rollout trajectories（tokens, loss_mask, reward）
2. **Forward Data**:
   - `log_probs`: 模型前向计算的 log probabilities
   - `entropy`: 熵值
   - `response_mask`: 有效位置的 mask
3. **Loss Data**:
   - `loss`: 总损失值
   - `pg_loss`: Policy gradient 损失
   - `entropy_loss`: 熵损失
4. **IS Loss Debug**: 详细的重要性采样损失计算中间值
   - `log_probs`: 新的 log probabilities
   - `old_log_probs`: 旧的 log probabilities（从 rollout）
   - `advantages`: 优势值
   - `prob_ratio`: 概率比率
   - `elementwise_loss`: 逐元素损失
   - `effective_valid`: 有效的 mask
5. **Datums**: 发送给 Weaver 的数据结构分析

### 5.4 输出示例

终端输出：

```
================================================================================
Debug Dump Comparison Report - Step 0
================================================================================

Directories:
  Self-Hosted: /path/to/self_hosted/debug_dump
  Remote API:  /path/to/remote_api/debug_dump
  Weaver:      /path/to/weaver-trainer/debug_dump

Comparing: 8 ranks × 2 microbatches

--- LOSS_DATA_ALL ---
Total comparisons: 16

Loss differences across ranks/micros:
  rank0_micro0 loss: ✗ diff=3.536344e-04
  rank0_micro1 loss: ✗ diff=3.521455e-04
  rank1_micro0 loss: ✓ diff=8.234567e-06
  ...

Diff statistics:
  max: 3.536344e-04
  min: 1.234567e-06
  mean: 1.234567e-04
```

### 5.5 问题诊断

**Shape mismatch**: 序列长度不一致
- 检查 padding 策略是否一致
- 检查 max_length 配置

**Loss 差异过大** (diff > 1e-5):
- 检查 entropy_coeff 是否一致
- 检查数值精度（FP16 vs FP32）
- 检查随机种子是否固定
- 查看 is_loss_debug 中的中间值定位问题

**Missing data**:
- 检查 dump_options 配置是否启用
- 检查环境变量是否正确设置（weaver-trainer）
