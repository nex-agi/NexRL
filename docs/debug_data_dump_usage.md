# Debug Data Dump 使用指南

## 1. 快速开始：使用 Debug Mode (推荐)

### 1.1 简化的 Debug Mode

从 v2.1 开始，NexRL 提供了简化的 `--debug-mode` 选项，自动处理 trajectory dump 和 load。

#### 使用方法

**首次运行（生成 trajectory）：**
```bash
# Internal (scripts/)
python scripts/run.py --mode self-hosted --train-config recipe.yaml --run-nexrl --debug-mode

# Open-source (cli/)
python cli/run.py --mode self-hosted --train-config config.yaml --run-nexrl --debug-mode
```

**第二次运行（自动检测并复用 trajectory）：**
```bash
# 运行相同命令
python scripts/run.py --mode self-hosted --train-config recipe.yaml --run-nexrl --debug-mode
```

系统会：
1. 自动查找最近生成的 trajectory
2. 显示 trajectory 信息（路径、大小、时间）
3. 询问是否复用：`Use this trajectory for mock rollout? [Y/n]:`
4. 如果选择 Y：使用 mock mode，自动将 rollout workers 减少到 1
5. 如果选择 N：正常执行 rollout

**非交互模式（脚本化工作流）：**
```bash
python scripts/run.py --mode self-hosted --train-config recipe.yaml --run-nexrl \
  --debug-mode \
  --debug-baseline-path logs/experiment_name/20260206-120000
```

### 1.2 Debug Mode 特性

- ✅ **自动检测**：智能查找最新的包含 trajectory 的实验（跳过仅复用 trajectory 的运行）
- ✅ **交互确认**：显示详细信息，用户明确确认
- ✅ **性能优化**：Mock mode 自动将 rollout workers 减少到 1（避免 224 个 worker 加载相同文件）
- ✅ **无缝集成**：通过 Hydra overrides 自动配置，无需修改 YAML
- ✅ **兼容两种模式**：同时支持 self-hosted 和 training-service 模式

### 1.3 输出示例

**首次运行（无 trajectory）：**
```
[WARNING] No trajectory found in any previous run.
[WARNING] Normal rollout will execute.
[INFO] Debug mode: Normal rollout (will dump trajectory)
[INFO] Using experiment path: logs/experiment/20260206-120000
```

**第二次运行（检测到 trajectory）：**
```
[INFO] Found trajectory from run: 20260206-120000
Trajectory: logs/experiment/20260206-120000/debug_dump/trajectory/step_000000.pt
Size: 2.5 MB | Modified: 2026-02-06 12:15:32
Use this trajectory for mock rollout? [Y/n]: Y
[INFO] Using trajectory: logs/experiment/20260206-120000/debug_dump/trajectory/step_000000.pt
[INFO] Debug mode: Mock rollout (reusing trajectory)
[INFO] Automatically reducing rollout workers to 1 (no parallel benefit in mock mode)
```

---

## 2. 手动配置方式（高级用户）

如果需要更精细的控制，可以手动配置 debug dump。

### 2.1 启用 Debug Data Dump

#### 2.1.1 Self-hosted Mode

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

#### 2.1.2 Remote API Mode

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

### 2.2 从 Dump 文件加载 Trajectory 训练

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

### 2.3 完整示例

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

---

## 3. Dump 目录结构

### 3.1 Self-hosted Mode

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

### 3.2 Remote API Mode

```
debug_dump/
├── prepared_trajectories/   # GRPO advantage computation results
│   └── step_000000.pt
├── datums/                  # Datums sent to Weaver
│   └── step_000000.pt
└── training_metrics/        # Metrics from Weaver
    └── step_000000.pt
```

### 3.3 Weaver-trainer Mode

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

## 4. 比较 Debug Dumps

使用 `compare_debug_dumps.py` 脚本比较 self-hosted 和 weaver-trainer 的 dump 数据：

### 4.1 基本用法

```bash
python nexrl/utils/compare_debug_dumps.py \
  --self_hosted_dir /path/to/self_hosted/debug_dump \
  --weaver_dir /path/to/weaver-trainer/debug_dump \
  --step 0 \
  --output comparison_report.json
```

### 4.2 完整参数

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

### 4.3 比较内容

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

### 4.4 输出示例

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

### 4.5 问题诊断

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

---

## 5. FAQ

### 5.1 为什么 Mock mode 会自动减少 rollout workers 到 1？

Mock mode 只是重放预先录制的 trajectory，不需要 LLM 推理，因此并行 workers 没有性能优势。如果使用 224 个 workers，每个都会加载整个 trajectory 文件（可能几百 MB），造成极大的内存和 I/O 浪费。

**节省示例：**
- 之前：224 workers × 100MB trajectory = 22.4GB 内存浪费
- 现在：1 worker × 100MB trajectory = 100MB（节省 222x）

### 5.2 如何跳过交互式确认？

使用 `--debug-baseline-path` 参数：

```bash
python scripts/run.py --mode self-hosted --train-config recipe.yaml --run-nexrl \
  --debug-mode \
  --debug-baseline-path logs/experiment/20260206-120000
```

### 5.3 为什么自动检测会跳过某些运行？

Debug mode 使用"filter-then-sort"算法，只查找**包含 trajectory 文件**的运行：

- Run 1 (20260201-100000): 正常 rollout → 生成 trajectory ✓
- Run 2 (20260202-110000): Mock mode 复用 Run 1 → 无 trajectory ✗
- Run 3 (20260203-120000): 自动检测会找到 Run 1（不是 Run 2）

这确保找到的是真正生成数据的运行，而不是中间的复用运行。

### 5.4 Debug mode 支持哪些模式？

- ✅ Self-hosted mode (scripts/ 和 cli/)
- ✅ Training-service mode (scripts/ 和 cli/)
- ✅ 两者都支持交互式和非交互式工作流

### 5.5 如果我想保留多个 workers 进行 mock rollout？

不推荐，但如果确实需要，可以手动配置 YAML（不使用 `--debug-mode`），系统会发出警告：

```
[WARNING] MockRolloutWorker is loading trajectories on 224 workers.
This is wasteful as each worker loads the entire file.
Consider setting rollout_worker.resource.num_workers=1 for trajectory reuse.
```
