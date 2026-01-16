# Recipe Configuration

This document explains the YAML configuration system used in NexRL recipes, including Hydra composition and inheritance patterns.

## Overview

NexRL uses [Hydra](https://hydra.cc/) for configuration management. Recipes leverage Hydra's composition system to share common configuration and override settings per deployment mode.

## Configuration Files

Each recipe typically has these configuration files:

- `common.yaml` - Shared base configuration
- `self_hosted.yaml` - Self-hosted mode overrides
- `tinker.yaml` - Tinker service overrides
- `weaver.yaml` - Weaver service overrides

## Hydra Composition

### Defaults List

Mode-specific configs inherit from common using the `defaults` list:

```yaml
# self_hosted.yaml
defaults:
  - common         # Load common.yaml first
  - _self_         # Then apply settings from this file

# Mode-specific overrides below
project_name: "NexRL-self-hosted"
```

The order matters:
1. `common` - Load base configuration
2. `_self_` - Apply current file's settings (overrides common)

### Override Behavior

Settings in the mode-specific file override common settings:

```yaml
# common.yaml
data:
  batch_size: 16
  shuffle: true

# self_hosted.yaml
defaults:
  - common
  - _self_

data:
  batch_size: 32  # Overrides common.yaml
  # shuffle: true is inherited from common
```

## Configuration Structure

### Top-Level Sections

```yaml
# Project identification
project_name: "NexRL-MyTask"
experiment_name: "my-task-training"
launch_mode: "local"

# Logging
logger:
  backend: ['console', 'wandb']
  enable_feishu_logging: false

# Environment setup
environment:
  setup_script: "self_hosted.env.sh"
  require_setup_script: false

# Checkpoint resuming
resume:
  mode: disable
  resume_path: ""

# Data loading
data:
  type: "torch"
  # ... data configuration

# Rollout workers
rollout_worker:
  type: "nexau"
  # ... rollout configuration

# Trajectory pool
trajectory_pool:
  type: "default"
  # ... pool configuration

# Trainer
trainer:
  type: "self_hosted_grpo"
  # ... trainer configuration

# Weight synchronization
weight:
  type: "default"
  sync_mode: "sync"
  # ... weight configuration

# Validation
validate:
  validate_before_train: true
  # ... validation configuration

# Runtime monitoring
runtime_monitor:
  exception_handling:
    enabled: true
  health_check:
    enabled: true

# Resource allocation
resource:
  # ... resource configuration
```

## Environment Variable Interpolation

### Hydra OmegaConf Resolver

Use `${oc.env:VAR_NAME}` to reference environment variables:

```yaml
data:
  data_files:
    - "${oc.env:NEXRL_DATA_PATH}/my_task/train.parquet"
  tokenizer_path: "${oc.env:NEXRL_MODEL_PATH}/Qwen/Qwen3-8B"
```

### With Defaults

Provide default values if environment variable is not set:

```yaml
logger:
  feishu_url: "${oc.env:FEISHU_WEBHOOK_URL,}"  # Empty string if not set
```

### Common Environment Variables

- `NEXRL_DATA_PATH` - Base path for training data
- `NEXRL_MODEL_PATH` - Base path for model files
- `EXPERIMENT_PATH` - Output directory for checkpoints
- `INFERENCE_BASE_URL` - Inference service endpoint
- `API_SERVER_URL` - Training service endpoint

## Configuration Value References

### Internal References

Reference other config values using `${path.to.value}`:

```yaml
data:
  max_prompt_length: 100000
  max_response_length: 8192

rollout_worker:
  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}

trainer:
  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}
```

Benefits:
- Single source of truth
- Consistent values across modules
- Easy to update

## Common Configuration Sections

### Data Configuration

```yaml
data:
  type: "torch"                    # DataLoader type
  seed: 42                         # Random seed
  data_files:                      # Training data files
    - "${oc.env:NEXRL_DATA_PATH}/task/train.parquet"
  batch_size: 16                   # Batch size for data loading
  keep_batch_order: true           # Preserve batch order
  rollout_repeat_n: 8              # Rollouts per data item
  prompt_key: "prompt"             # Column name for prompts
  filter_prompts: false            # Filter by length
  max_prompt_length: 100000        # Max prompt tokens
  max_response_length: 8192        # Max response tokens
  tokenizer_path: "${oc.env:NEXRL_MODEL_PATH}/model"
  shuffle: true                    # Shuffle data
  drop_last: true                  # Drop incomplete batches
```

### Rollout Worker Configuration

```yaml
rollout_worker:
  type: "nexau"                    # Worker type
  num_workers: 128                 # Number of rollout workers
  need_llm_inference: true         # Requires LLM service
  temperature: 0.7                 # Sampling temperature

  # Custom worker (optional)
  custom_rollout_worker_module_path: "agent_workspace/my_worker.py"
  custom_rollout_worker_class_name: "MyWorker"

  # NexAU-specific
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
  evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"
  task_name: "my_task"

  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}
```

### Trajectory Pool Configuration

```yaml
trajectory_pool:
  type: "default"                  # Pool type
  batch_size: 128                  # Training batch size
  group_size: 1                    # Trajectories per group
  key_list: []                     # Grouping keys
  check_batch_ready_function: "loaded_batch_finished"
```

### Trainer Configuration

```yaml
trainer:
  type: "self_hosted_grpo"         # Trainer type
  total_train_steps: 200           # Training steps
  save_freq: 0                     # Checkpoint frequency (0=disable)
  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}

  # Self-hosted specific
  checkpoint_path: "${oc.env:EXPERIMENT_PATH}/ckpt"
  sync_weight_path: "${oc.env:EXPERIMENT_PATH}/sync_weight"
  remove_previous_ckpt: false

  # Algorithm configuration
  algorithm:
    type: "grpo"
    do_old_log_prob_compute: true
    use_kl_in_reward: false
    kl_penalty: kl
    kl_ctrl:
      type: fixed
      kl_coef: 0.001
```

### Weight Synchronization Configuration

```yaml
weight:
  type: "default"                  # Weight sync controller type
  sync_mode: "sync"                # Sync mode: sync/async/batch-async
  sync_method: "disk"              # Sync method: disk/api
  sync_weight_path: ${trainer.sync_weight_path}
  staleness_threshold: 0           # Max steps before forced sync
  validate_freq: ${validate.eval.validate_freq}
```

### Validation Configuration

```yaml
validate:
  validate_before_train: true      # Run validation before training

  data:
    type: "torch"
    seed: ${data.seed}
    data_files:
      - "${oc.env:NEXRL_DATA_PATH}/task/test.parquet"
    batch_size: 16
    prompt_key: "prompt"
    filter_prompts: false
    max_prompt_length: ${data.max_prompt_length}
    tokenizer_path: ${data.tokenizer_path}
    shuffle: true
    drop_last: false

  eval:
    type: "default"
    validate_freq: 0               # Steps between validation (0=disable)
```

### Service Configuration (Self-Hosted)

```yaml
service:
  inference_service:
    model_tag: "default"
    api_key: "EMPTY"
    base_url: "${oc.env:INFERENCE_BASE_URL}"
    model: ${resource.inference.served_model_name}
    max_tokens: ${data.max_response_length}
    tokenizer: ${data.tokenizer_path}
    backend: sglang                # sglang, vllm, etc.
    max_retries: 3
    freeze_for_weight_sync: true   # Pause during sync
    weight_type: "sglang_nckpt"    # Weight format

  train_service:
    model_tag: "default"
    backend: http                  # http, tinker, weaver
    world_size: 8
    url: "http://${oc.env:API_SERVER_URL}:8000"

    actor:
      checkpoint_manager: dcp      # Distributed checkpoint
      model:
        path: "${resource.inference.model_path}"
        use_remove_padding: true
      # ... detailed training configuration
```

### Resource Configuration (Self-Hosted)

```yaml
resource:
  train:
    actor-1:
      world_size: 1                # Number of training nodes
      gpus_per_pod: 8              # GPUs per node
      memory_per_gpu: 200          # GB memory per GPU

  inference:
    replicas: 4                    # Number of inference replicas
    gpus_per_replica: 2            # GPUs per replica
    served_model_name: "my-model"
    model_path: "${oc.env:NEXRL_MODEL_PATH}/model"
    extra_args: ""

  agent:
    num_workers: 0                 # Dedicated agent workers (0=collocated)
    agents_per_worker: 32          # Agents per worker
```

## Real Configuration Examples

### Common Configuration

From `recipe/nexau_news/common.yaml`:

```yaml
project_name: "NexRL-NexAU-News"
experiment_name: "nexau-news-training"
launch_mode: "local"
multi_model: "false"

logger:
  backend: ['console', 'wandb']
  enable_feishu_logging: false
  feishu_url: "${oc.env:FEISHU_WEBHOOK_URL,}"

environment:
  setup_script: ""
  require_setup_script: false

resume:
  mode: disable
  resume_path: ""

data:
  type: "torch"
  seed: 42
  data_files:
    - "${oc.env:NEXRL_DATA_PATH}/news/train.parquet"
  batch_size: 16
  keep_batch_order: true
  rollout_repeat_n: 8
  prompt_key: "prompt"
  filter_prompts: false
  max_prompt_length: 100000
  max_response_length: 8192
  tokenizer_path: "${oc.env:NEXRL_MODEL_PATH}/Qwen/Qwen3-8B"
  shuffle: true
  drop_last: true

rollout_worker:
  type: "nexau"
  num_workers: 128
  need_llm_inference: true
  temperature: 0.7
  custom_rollout_worker_module_path: "agent_workspace/news_rollout_worker.py"
  custom_rollout_worker_class_name: "NewsNexAURolloutWorker"
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
  evaluator_module_path: "agent_workspace/evaluator.py:NewsEvaluator"
  task_name: "news"
  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}

trajectory_pool:
  type: "default"
  batch_size: 128
  group_size: 1
  key_list: []
  check_batch_ready_function: "loaded_batch_finished"

trainer:
  total_train_steps: 200
  save_freq: 0
  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}

weight:
  type: "default"
  sync_mode: "sync"
  staleness_threshold: 0
  validate_freq: ${validate.eval.validate_freq}

validate:
  validate_before_train: true
  data:
    type: "torch"
    seed: ${data.seed}
    data_files:
      - "${oc.env:NEXRL_DATA_PATH}/news/test.parquet"
    batch_size: 16
    prompt_key: "prompt"
    filter_prompts: false
    max_prompt_length: ${data.max_prompt_length}
    tokenizer_path: ${data.tokenizer_path}
    shuffle: true
    drop_last: false
  eval:
    type: "default"
    validate_freq: 0

runtime_monitor:
  exception_handling:
    enabled: true
    check_interval: 1.0
    policy: "stop_on_error"
  health_check:
    enabled: true
    check_interval: 10.0
    timeout: 5.0

resource:
  agent:
    num_workers: 0
    agents_per_worker: 32
```

### Self-Hosted Mode Configuration

From `recipe/nexau_news/self_hosted.yaml`:

```yaml
defaults:
  - common
  - _self_

project_name: "NexRL-self-hosted"
experiment_name: "self-hosted-nexau-news"

environment:
  setup_script: "self_hosted.env.sh"
  require_setup_script: true

data:
  tokenizer_path: ${resource.inference.model_path}
  max_prompt_length: ${rollout_worker.max_prompt_length}

trainer:
  type: "self_hosted_grpo"
  checkpoint_path: "${oc.env:EXPERIMENT_PATH}/ckpt"
  sync_weight_path: "${oc.env:EXPERIMENT_PATH}/sync_weight"
  remove_previous_ckpt: false

  algorithm:
    type: "grpo"
    do_old_log_prob_compute: true
    use_kl_in_reward: false
    kl_penalty: kl
    kl_ctrl:
      type: fixed
      kl_coef: 0.001
      kl_reward_coef: ${trainer.algorithm.kl_ctrl.kl_coef}

weight:
  sync_method: "disk"
  sync_weight_path: ${trainer.sync_weight_path}

service:
  inference_service:
    model_tag: "default"
    api_key: "EMPTY"
    base_url: "${oc.env:INFERENCE_BASE_URL}"
    model: ${resource.inference.served_model_name}
    max_tokens: ${data.max_response_length}
    tokenizer: ${data.tokenizer_path}
    backend: sglang
    max_retries: 3
    freeze_for_weight_sync: true
    weight_type: "sglang_nckpt"

  train_service:
    model_tag: "default"
    backend: http
    world_size: 8
    url: "http://${oc.env:API_SERVER_URL}:8000"
    actor:
      checkpoint_manager: dcp
      model:
        path: "${resource.inference.model_path}"
        use_remove_padding: true
      # ... more training configuration

resource:
  train:
    actor-1:
      world_size: 1
      gpus_per_pod: 8
      memory_per_gpu: 200
  inference:
    replicas: 4
    gpus_per_replica: 2
    served_model_name: "nexrl-nexau-news-qwen3-8b"
    model_path: "${oc.env:NEXRL_MODEL_PATH}/Qwen/Qwen3-8B"
    extra_args: ""
  agent:
    num_workers: 0
    agents_per_worker: 32
```

## Best Practices

### Organization

1. **Use common.yaml for shared settings** - Avoid duplication
2. **Override only what changes** - Keep mode-specific configs minimal
3. **Group related settings** - Use nested structure for clarity
4. **Document complex settings** - Add comments for non-obvious values

### Environment Variables

1. **Use environment variables for paths** - Makes configs portable
2. **Provide defaults when appropriate** - Use `${oc.env:VAR,default}`
3. **Document required variables** - List in recipe README
4. **Use consistent variable names** - Follow project conventions

### Value References

1. **Reference values instead of duplicating** - Use `${path.to.value}`
2. **Define once, use everywhere** - Single source of truth
3. **Be careful with circular references** - Avoid infinite loops
4. **Use meaningful paths** - Clear reference structure

### Composition

1. **Follow the defaults pattern** - `common` then `_self_`
2. **Keep inheritance shallow** - One level of inheritance
3. **Test mode-specific configs** - Verify overrides work correctly
4. **Document inheritance structure** - Explain what overrides what

## Troubleshooting

### Configuration Not Loading

**Problem:** Hydra can't find configuration file

**Solution:**
- Verify file is in correct location
- Check `defaults` list syntax
- Ensure file has `.yaml` extension
- Review Hydra error message for path issues

### Environment Variable Not Resolved

**Problem:** Variable shows as `${oc.env:VAR_NAME}` in logs

**Solution:**
- Check variable is exported in environment
- Verify environment script is executed
- Set `require_setup_script: true` if needed
- Check variable name spelling

### Override Not Applied

**Problem:** Mode-specific value not overriding common

**Solution:**
- Verify `_self_` is after `common` in defaults list
- Check indentation matches exactly
- Ensure path to value is correct
- Review merged config with `--cfg job` flag

### Reference Not Resolving

**Problem:** `${path.to.value}` not resolving

**Solution:**
- Verify referenced value exists
- Check path syntax is correct
- Ensure no circular references
- Review OmegaConf documentation

## Advanced Topics

### Conditional Configuration

Use resolvers for conditional values:

```yaml
trainer:
  type: "${oc.env:TRAINER_TYPE,self_hosted_grpo}"  # Default if not set
```

### List Merging

Append to lists from common:

```yaml
# common.yaml
data:
  data_files:
    - "${oc.env:NEXRL_DATA_PATH}/train.parquet"

# self_hosted.yaml (if you want to add more)
data:
  data_files:
    - "${oc.env:NEXRL_DATA_PATH}/train.parquet"
    - "${oc.env:NEXRL_DATA_PATH}/extra.parquet"
```

**Note:** Lists are replaced, not merged by default. Repeat common values if needed.

### Dict Merging

Dicts are merged recursively:

```yaml
# common.yaml
service:
  inference_service:
    model_tag: "default"
    max_tokens: 8192

# self_hosted.yaml
service:
  inference_service:
    base_url: "${oc.env:INFERENCE_BASE_URL}"  # Added to dict
    max_tokens: 4096                           # Overrides common
    # model_tag: "default" is inherited
```

## Related Documentation

- [Recipe Structure](./recipe-structure.md) - Recipe organization
- [Agent Configuration](./agent-configuration.md) - Agent config details
- [Environment Setup](./environment-setup.md) - Environment scripts
- [Configuration Setup](../01-getting-started/configuration-setup.md) - Environment variables
- [Complete Config Reference](../11-configuration-reference/complete-config.md) - Full configuration options

---

**Next**: [Environment Setup](./environment-setup.md) - Learn about environment scripts and dependency management.
