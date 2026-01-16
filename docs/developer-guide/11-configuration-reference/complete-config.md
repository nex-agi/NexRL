# Complete Configuration Reference

Complete annotated configuration example for NexRL.

## Configuration Structure

NexRL uses [Hydra](https://hydra.cc/) for configuration management with support for composition, inheritance, and environment variable interpolation.

### Configuration Composition

Recipes typically use a `common.yaml` base with mode-specific overrides:

```
recipe/my_task/
├── common.yaml           # Shared configuration
├── self_hosted.yaml      # Self-hosted mode overrides
├── tinker.yaml          # Tinker backend overrides
└── weaver.yaml          # Weaver backend overrides
```

**Example Override (`self_hosted.yaml`):**

```yaml
defaults:
  - common              # Inherit from common.yaml
  - _self_             # Apply overrides last

trainer:
  type: "self_hosted_grpo"  # Override trainer type
  checkpoint_path: "${oc.env:EXPERIMENT_PATH}/ckpt"
```

## Complete Configuration Example

Based on `recipe/math/common.yaml` and `recipe/math/self_hosted.yaml`:

```yaml
#==========================================
# PROJECT CONFIGURATION
#==========================================

project_name: "NexRL-MyProject"
experiment_name: "my-experiment"
launch_mode: "local"  # Options: local, ray
multi_model: "false"

#==========================================
# LOGGING CONFIGURATION
#==========================================

logger:
  backend: ['console', 'wandb']  # Options: console, wandb, swanlab
  enable_feishu_logging: false
  feishu_url: "${oc.env:FEISHU_WEBHOOK_URL,}"

environment:
  setup_script: ""  # Optional environment setup script
  require_setup_script: false

#==========================================
# RESUME CONFIGURATION
#==========================================

resume:
  mode: disable  # Options: disable, auto, from_path
  resume_path: ""  # Required when mode is 'from_path'
  resume_dataloader: true  # Skip consumed batches on resume

#==========================================
# DATA LOADER CONFIGURATION
#==========================================

data:
  type: "torch"  # Options: torch, mock
  seed: 42

  # Data files with environment variable interpolation
  data_files:
    - "${oc.env:NEXRL_DATA_PATH}/my_data/train.parquet"

  batch_size: 32
  keep_batch_order: true
  rollout_repeat_n: 8  # Rollouts per prompt
  prompt_key: "prompt"  # Column name for prompts
  filter_prompts: false
  max_prompt_length: 4096
  max_response_length: 8192

  # Tokenizer configuration
  tokenizer_path: "${oc.env:NEXRL_MODEL_PATH}/Qwen/Qwen3-8B"

  shuffle: true
  drop_last: true

#==========================================
# ROLLOUT CONFIGURATION
#==========================================

rollout_worker:
  type: "nexau"  # Options: simple, agent, nexau
  num_workers: 128
  need_llm_inference: true
  temperature: 0.7

  # NexAU-specific (for type: "nexau")
  custom_rollout_worker_module_path: "agent_workspace/my_worker.py"
  custom_rollout_worker_class_name: "MyWorker"
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
  evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"
  task_name: "my_task"

  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}

#==========================================
# TRAJECTORY POOL CONFIGURATION
#==========================================

trajectory_pool:
  type: "default"
  batch_size: 128
  group_size: 1  # Trajectories per group
  key_list: []  # Grouping keys (empty = no grouping)
  check_batch_ready_function: "loaded_batch_finished"
  # Options: batch_size, loaded_batch_finished

#==========================================
# TRAINER CONFIGURATION
#==========================================

trainer:
  type: "self_hosted_grpo"  # See trainer-config.md
  total_train_steps: 200
  save_freq: 10  # Checkpoint save frequency (0 = no checkpoints)

  # Self-hosted specific
  checkpoint_path: "${oc.env:EXPERIMENT_PATH}/ckpt"
  sync_weight_path: "${oc.env:EXPERIMENT_PATH}/sync_weight"
  remove_previous_ckpt: false

  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}

  # Algorithm configuration (self_hosted_grpo only)
  algorithm:
    type: "grpo"
    do_old_log_prob_compute: true
    use_kl_in_reward: false
    kl_penalty: kl

    kl_ctrl:
      type: fixed  # Options: fixed, adaptive
      kl_coef: 0.001
      kl_reward_coef: ${trainer.algorithm.kl_ctrl.kl_coef}

#==========================================
# WEIGHT MANAGEMENT CONFIGURATION
#==========================================

weight:
  type: "default"
  sync_mode: "sync"  # Options: sync, fully-async, batch-async
  staleness_threshold: 0  # For async modes
  validate_freq: ${validate.eval.validate_freq}

  # Self-hosted specific
  sync_method: "disk"  # Options: disk, api
  sync_weight_path: ${trainer.sync_weight_path}

#==========================================
# SERVICE CONFIGURATION
#==========================================

service:
  # Inference service (for rollout)
  inference_service:
    model_tag: "default"
    api_key: "EMPTY"
    base_url: "${oc.env:INFERENCE_BASE_URL}"
    model: "my-model"
    max_tokens: ${data.max_response_length}
    tokenizer: ${data.tokenizer_path}
    backend: sglang  # Options: sglang, vllm, openai
    max_retries: 3
    freeze_for_weight_sync: true
    weight_type: "sglang_nckpt"

  # Training service (self-hosted only)
  train_service:
    model_tag: "default"
    backend: http  # NexTrainer backend
    world_size: 8
    url: "http://${oc.env:API_SERVER_URL}:8000"

    actor:
      checkpoint_manager: dcp  # Distributed checkpoint

      model:
        path: "${oc.env:NEXRL_MODEL_PATH}/Qwen/Qwen3-8B"
        use_remove_padding: true

      # Training hyperparameters
      ppo_mini_batch_size: 28
      ppo_micro_batch_size: 4
      grad_clip: 1.0
      clip_ratio: 0.2
      loss_agg_mode: token-mean
      entropy_coeff: 1e-4
      ppo_epochs: 1

      # Optimizer configuration
      optim:
        lr: 2e-6
        lr_warmup_steps_ratio: 0.0
        warmup_style: constant

      # FSDP configuration
      fsdp_config:
        wrap_policy:
          min_num_params: 0
        param_offload: false
        grad_offload: false

#==========================================
# VALIDATION CONFIGURATION
#==========================================

validate:
  validate_before_train: false

  data:
    type: "torch"
    seed: ${data.seed}
    data_files:
      - "${oc.env:NEXRL_DATA_PATH}/my_data/test.parquet"
    batch_size: 16
    prompt_key: "prompt"
    filter_prompts: false
    max_prompt_length: ${data.max_prompt_length}
    tokenizer_path: ${data.tokenizer_path}
    shuffle: true
    drop_last: false

  eval:
    type: "default"
    validate_freq: 0  # Frequency (0 = disabled)

#==========================================
# RUNTIME MONITORING CONFIGURATION
#==========================================

runtime_monitor:
  exception_handling:
    enabled: true
    check_interval: 1.0  # Seconds
    policy: "stop_on_error"  # Options: stop_on_error, continue, stop_on_critical

  health_check:
    enabled: true
    check_interval: 10.0  # Seconds
    timeout: 5.0  # Ray mode only

#==========================================
# COMPUTE RESOURCE CONFIGURATION
#==========================================

resource:
  # Training resources (self-hosted only)
  train:
    actor-1:
      world_size: 1
      gpus_per_pod: 8
      memory_per_gpu: 200

  # Inference resources (self-hosted only)
  inference:
    replicas: 4
    gpus_per_replica: 2
    served_model_name: "my-model"
    model_path: "${oc.env:NEXRL_MODEL_PATH}/Qwen/Qwen3-8B"
    extra_args: ""

  # Agent resources
  agent:
    num_workers: 0
    agents_per_worker: 32
```

## Environment Variables

Common environment variables used in configurations:

### Required

- `NEXRL_PATH` - NexRL installation path
- `NEXRL_DATA_PATH` - Data directory path
- `NEXRL_MODEL_PATH` - Model directory path
- `EXPERIMENT_PATH` - Experiment output path

### Optional

- `INFERENCE_BASE_URL` - Inference service URL
- `API_SERVER_URL` - Training service URL
- `WANDB_HOST`, `WANDB_KEY` - WandB credentials
- `FEISHU_WEBHOOK_URL` - Feishu logging webhook

## Configuration Modes

### Local Mode

```yaml
launch_mode: "local"

# Suitable for:
# - Development and debugging
# - Single-node execution
# - Small-scale experiments
```

### Ray Mode

```yaml
launch_mode: "ray"

# Suitable for:
# - Multi-node clusters
# - Distributed execution
# - Production deployments
```

## Related Documentation

- [Data Configuration](./data-config.md) - Data loader options
- [Rollout Configuration](./rollout-config.md) - Rollout worker options
- [Trainer Configuration](./trainer-config.md) - Trainer type options
- [Service Configuration](./service-config.md) - Service backend options
- [Configuration Setup](../01-getting-started/configuration-setup.md) - Environment setup
