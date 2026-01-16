# Trainer Configuration

Configuration options for trainer components.

## Trainer Types

### Self-Hosted GRPO (`type: "self_hosted_grpo"`)

Self-hosted training with GRPO algorithm using NexTrainer backend.

```yaml
trainer:
  type: "self_hosted_grpo"
  total_train_steps: 200
  save_freq: 10

  # Checkpoint paths
  checkpoint_path: "${oc.env:EXPERIMENT_PATH}/ckpt"
  sync_weight_path: "${oc.env:EXPERIMENT_PATH}/sync_weight"
  remove_previous_ckpt: false

  # Length constraints
  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}

  # Algorithm configuration
  algorithm:
    type: "grpo"
    do_old_log_prob_compute: true
    use_kl_in_reward: false
    kl_penalty: kl

    kl_ctrl:
      type: fixed
      kl_coef: 0.001
      kl_reward_coef: ${trainer.algorithm.kl_ctrl.kl_coef}
```

### Remote API GRPO (`type: "remote_api_grpo"`)

GRPO training using Tinker or Weaver backend.

```yaml
trainer:
  type: "remote_api_grpo"
  total_train_steps: 200
  save_freq: 0  # Remote API handles checkpointing

  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}
```

### Remote API Cross Entropy (`type: "remote_api_cross_entropy"`)

Supervised learning using remote API backend.

```yaml
trainer:
  type: "remote_api_cross_entropy"
  total_train_steps: 100

  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}
```

## Common Options

### type

**Type:** `string`
**Options:** `"self_hosted_grpo"`, `"remote_api_grpo"`, `"remote_api_cross_entropy"`
**Default:** Required

Trainer implementation.

- `self_hosted_grpo`: Self-hosted with GRPO algorithm
- `remote_api_grpo`: Remote API with GRPO algorithm
- `remote_api_cross_entropy`: Remote API with supervised learning

### total_train_steps

**Type:** `int`
**Default:** Required

Maximum number of training steps.

**Example:**
```yaml
# Short training
total_train_steps: 50

# Standard training
total_train_steps: 200

# Extended training
total_train_steps: 1000
```

### save_freq

**Type:** `int`
**Default:** `0` (no checkpoints)

Checkpoint save frequency in steps.

**Values:**
- `0`: No checkpoints saved
- `>0`: Save every N steps

**Example:**
```yaml
# No checkpoints
save_freq: 0

# Checkpoint every 10 steps
save_freq: 10

# Checkpoint every step
save_freq: 1
```

### max_prompt_length

**Type:** `int`
**Default:** Inherited from `data.max_prompt_length`

Maximum prompt length for training.

### max_response_length

**Type:** `int`
**Default:** Inherited from `data.max_response_length`

Maximum response length for training.

## Self-Hosted Specific Options

### checkpoint_path

**Type:** `string`
**Default:** Required

Directory for model checkpoints.

**Example:**
```yaml
checkpoint_path: "${oc.env:EXPERIMENT_PATH}/ckpt"
```

**Structure:**
```
ckpt/
├── global_step_10/
│   ├── model_state_dict.pt
│   └── optimizer_state_dict.pt
├── global_step_20/
└── global_step_30/
```

### sync_weight_path

**Type:** `string`
**Default:** Required

Directory for weight synchronization buffer.

**Example:**
```yaml
sync_weight_path: "${oc.env:EXPERIMENT_PATH}/sync_weight"
```

**Purpose:**
- Temporary storage for weight sync
- Used by WeightSyncController
- Synchronized to inference service

### remove_previous_ckpt

**Type:** `bool`
**Default:** `false`

Remove old checkpoints to save disk space.

**When `true`:**
- Only latest checkpoint kept
- Saves disk space
- Cannot resume from earlier steps

**When `false`:**
- All checkpoints retained
- Allows rollback
- Requires more storage

## Algorithm Configuration (Self-Hosted GRPO)

### algorithm.type

**Type:** `string`
**Options:** `"grpo"`
**Default:** `"grpo"`

Algorithm type for self-hosted training.

### algorithm.do_old_log_prob_compute

**Type:** `bool`
**Default:** `true`

Recompute old log probabilities during training.

**When `true`:**
- More accurate advantage estimation
- Accounts for policy drift
- Requires reference model

**When `false`:**
- Use stored log probabilities
- Faster training
- May be less accurate

### algorithm.use_kl_in_reward

**Type:** `bool`
**Default:** `false`

Include KL divergence penalty in rewards.

**When `true`:**
- Penalize policy deviation from reference
- More conservative training
- Requires KL controller configuration

**When `false`:**
- No KL penalty
- Faster policy updates

### algorithm.kl_penalty

**Type:** `string`
**Options:** `"kl"`, `"abs"`, `"mse"`
**Default:** `"kl"`

Type of KL penalty to apply.

### KL Controller Configuration

#### Fixed KL Controller

```yaml
algorithm:
  kl_ctrl:
    type: fixed
    kl_coef: 0.001
    kl_reward_coef: 0.001
```

**Options:**
- `kl_coef`: Fixed KL coefficient
- `kl_reward_coef`: Coefficient for reward KL penalty

#### Adaptive KL Controller

```yaml
algorithm:
  kl_ctrl:
    type: adaptive
    target_kl: 0.1
    horizon: 10000
```

**Options:**
- `target_kl`: Target KL divergence
- `horizon`: Adaptation time horizon

## Training Service Configuration

See [Service Configuration](./service-config.md) for `service.train_service` options.

## Common Patterns

### Quick Experiment

```yaml
trainer:
  type: "self_hosted_grpo"
  total_train_steps: 50
  save_freq: 0  # No checkpoints
  algorithm:
    use_kl_in_reward: false
```

### Production Training

```yaml
trainer:
  type: "self_hosted_grpo"
  total_train_steps: 500
  save_freq: 10  # Regular checkpoints
  checkpoint_path: "${oc.env:EXPERIMENT_PATH}/ckpt"
  sync_weight_path: "${oc.env:EXPERIMENT_PATH}/sync_weight"
  remove_previous_ckpt: false

  algorithm:
    type: "grpo"
    do_old_log_prob_compute: true
    use_kl_in_reward: true
    kl_penalty: kl

    kl_ctrl:
      type: adaptive
      target_kl: 0.1
      horizon: 10000
```

### Remote API Training

```yaml
trainer:
  type: "remote_api_grpo"
  total_train_steps: 200
  save_freq: 0  # Remote service handles checkpoints
```

### Supervised Fine-Tuning

```yaml
trainer:
  type: "remote_api_cross_entropy"
  total_train_steps: 100
```

## Hyperparameter Guidelines

### total_train_steps

**Small Dataset (<10K samples):**
```yaml
total_train_steps: 50-100
```

**Medium Dataset (10K-100K samples):**
```yaml
total_train_steps: 200-500
```

**Large Dataset (>100K samples):**
```yaml
total_train_steps: 500-2000
```

### save_freq

**Development:**
```yaml
save_freq: 0  # No checkpoints
```

**Production:**
```yaml
save_freq: 10  # Every 10 steps
```

**Critical Experiments:**
```yaml
save_freq: 1  # Every step
```

### KL Penalty

**Exploration Phase:**
```yaml
use_kl_in_reward: false  # No constraint
```

**Stabilization Phase:**
```yaml
use_kl_in_reward: true
kl_ctrl:
  type: adaptive
  target_kl: 0.1
```

## Troubleshooting

### Checkpoints Not Saving

**Symptom:** No checkpoint directories created

**Solutions:**
- Check `save_freq > 0`
- Verify `checkpoint_path` directory exists and is writable
- Check disk space availability
- Review trainer logs for errors

### Out of Memory During Training

**Symptom:** OOM errors during training step

**Solutions:**
- Reduce batch size in trajectory pool
- Decrease `ppo_mini_batch_size` (self-hosted)
- Enable gradient checkpointing
- Reduce model size or sequence length

### KL Divergence Exploding

**Symptom:** KL divergence grows rapidly

**Solutions:**
- Enable `use_kl_in_reward`
- Lower learning rate
- Use adaptive KL controller
- Reduce `clip_ratio`

### Training Too Slow

**Symptom:** Low training throughput

**Solutions:**
- Disable `do_old_log_prob_compute` if not needed
- Increase `ppo_micro_batch_size`
- Optimize data loading
- Check GPU utilization

## Related Documentation

- [Trainers Overview](../06-trainers/overview.md) - Trainer architecture
- [Self-Hosted Trainers](../06-trainers/self-hosted-trainers.md) - Self-hosted implementation
- [Remote API Trainers](../06-trainers/remote-api-trainers.md) - Remote API implementation
- [Service Configuration](./service-config.md) - Training service options
- [Complete Config](./complete-config.md) - Full configuration example
