# Service Configuration

Configuration options for inference and training services.

## Service Types

### Inference Service

LLM service for rollout generation.

### Training Service

Backend service for model training (self-hosted only). Supports multiple services for advanced training methods like On-Policy Distillation.

## Inference Service Configuration

### SGLang Backend

```yaml
service:
  inference_service:
    identifier: "default"
    api_key: "EMPTY"
    base_url: "${oc.env:INFERENCE_BASE_URL}"
    model: "my-model"
    max_tokens: 8192
    tokenizer: "${oc.env:NEXRL_MODEL_PATH}/model"
    backend: sglang
    max_retries: 3
    freeze_for_weight_sync: true
    weight_type: "sglang_nckpt"
```

### vLLM Backend

```yaml
service:
  inference_service:
    identifier: "default"
    api_key: "EMPTY"
    base_url: "${oc.env:INFERENCE_BASE_URL}"
    model: "my-model"
    max_tokens: 8192
    backend: vllm
    max_retries: 3
    freeze_for_weight_sync: true
```

### OpenAI-Compatible API

```yaml
service:
  inference_service:
    identifier: "default"
    api_key: "${oc.env:OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4"
    max_tokens: 4096
    backend: openai
    max_retries: 3
    freeze_for_weight_sync: false
```

## Inference Service Options

### identifier

**Type:** `string`
**Default:** `"default"`

Model identifier for tracking.

**Use Cases:**
- Multi-model training
- Model version tracking
- Weight synchronization coordination

**Note:** Previously named `model_tag` (deprecated but still supported).

### api_key

**Type:** `string`
**Default:** `"EMPTY"`

API key for authentication.

**Examples:**
```yaml
# Self-hosted (no auth)
api_key: "EMPTY"

# OpenAI
api_key: "${oc.env:OPENAI_API_KEY}"

# Custom service
api_key: "${oc.env:MY_API_KEY}"
```

### base_url

**Type:** `string`
**Default:** Required

Base URL for inference service.

**Examples:**
```yaml
# Local service
base_url: "http://localhost:8000"

# Environment variable
base_url: "${oc.env:INFERENCE_BASE_URL}"

# Remote service
base_url: "https://inference.example.com"
```

### model

**Type:** `string`
**Default:** Required

Model name to request from service.

**Examples:**
```yaml
# SGLang served model
model: "nexrl-rollout"

# vLLM model
model: "my-model"

# OpenAI model
model: "gpt-4"
```

### max_tokens

**Type:** `int`
**Default:** Required

Maximum tokens to generate.

**Should Match:**
```yaml
max_tokens: ${data.max_response_length}
```

### tokenizer

**Type:** `string`
**Default:** None (SGLang backend only)

Tokenizer path for SGLang backend.

**Example:**
```yaml
tokenizer: "${oc.env:NEXRL_MODEL_PATH}/model"
```

### backend

**Type:** `string`
**Options:** `"sglang"`, `"vllm"`, `"openai"`
**Default:** `"sglang"`

Inference backend type.

- `sglang`: SGLang inference engine
- `vllm`: vLLM inference engine
- `openai`: OpenAI-compatible API

### max_retries

**Type:** `int`
**Default:** `3`

Number of retry attempts for failed requests.

**Examples:**
```yaml
# No retries
max_retries: 0

# Standard retries
max_retries: 3

# High reliability
max_retries: 10
```

### freeze_for_weight_sync

**Type:** `bool`
**Default:** `true`

Block inference during weight synchronization.

**When `true`:**
- Ensures consistent model weights across rollouts
- Prevents mixed-version trajectories
- Required for sync weight mode

**When `false`:**
- Allows inference during sync
- May use stale weights
- Suitable for external services

### weight_type

**Type:** `string`
**Options:** `"sglang_nckpt"`, `"vllm_ckpt"`
**Default:** None (backend-specific)

Weight format for synchronization.

**SGLang:**
```yaml
weight_type: "sglang_nckpt"
```

**vLLM:**
```yaml
weight_type: "vllm_ckpt"
```

### tool_parser

**Type:** `string`
**Default:** `"qwen25"`

Tool call parser type for extracting structured tool calls from model responses.

**Options:**
- `qwen25` - Qwen 2.5 format (default)
- `qwen3_coder` - Qwen 3 Coder format
- `gpt_oss` - GPT-OSS format
- `deepseekv31` - DeepSeek V3.1 format
- `simple_xml` - Simple XML format

**Example:**
```yaml
inference_service:
  tool_parser: "qwen25"  # Default
```

**See Also:** [Tool Parser Documentation](../08-features/tool-parser.md)

### reasoning_parser

**Type:** `string`
**Default:** `"think_tag"`

Reasoning parser detector type for extracting thinking/reasoning content from model responses.

**Options:**
- `think_tag` - Default simple parser (no streaming)
- `qwen3` - Qwen3 detector (streaming support)
- `deepseek_r1` - DeepSeek R1 detector (forced reasoning)
- `kimi` - Kimi model detector (special markers)
- `step3` - Step3/Step3.5 detector
- `nano_v3` - NanoV3 detector
- `minimax_append_think` - MiniMax special handling

**Examples:**
```yaml
# Default - stable, simple
inference_service:
  reasoning_parser: "think_tag"

# Qwen3 models with streaming
inference_service:
  reasoning_parser: "qwen3"

# DeepSeek R1
inference_service:
  reasoning_parser: "deepseek_r1"

# Kimi model
inference_service:
  reasoning_parser: "kimi"
```

**Comparison:**

| Detector | Streaming | Use For |
|----------|-----------|---------|
| `think_tag` | ❌ | Default/Generic, production stable |
| `qwen3` | ✅ | Qwen3, DeepSeek-V3, GLM-4.5, MiniMax |
| `deepseek_r1` | ✅ | DeepSeek R1 (force reasoning mode) |
| `kimi` | ✅ | Kimi Thinking model |
| `step3` | ✅ | Step3, Step3.5 models |
| `nano_v3` | ✅ | NanoV3 model |

**See Also:** [Reasoning Parser Documentation](../08-features/reasoning-parser.md)

## Training Service Configuration (Self-Hosted)

### Single Service (Standard Training)

```yaml
service:
  train_service:
    main_actor:  # Service name
      identifier: "default"
      role: "actor"
      backend: http
      url: "http://${oc.env:API_SERVER_URL}:8000"

      resource:
        world_size: 8
        gpus_per_pod: 8

      actor:
        checkpoint_manager: dcp

        model:
          path: "${oc.env:NEXRL_MODEL_PATH}/model"
          use_remove_padding: true

        # Training hyperparameters
        ppo_mini_batch_size: 28
        ppo_micro_batch_size: 4
        grad_clip: 1.0
        clip_ratio: 0.2
        entropy_coeff: 1e-4
        ppo_epochs: 1

        # Optimizer
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
```

### Multiple Services (On-Policy Distillation)

```yaml
service:
  train_service:
    student_service:
      identifier: "student"
      role: "actor"
      backend: http
      url: "http://${oc.env:API_SERVER_URL}:8000"
      resource:
        world_size: 8
      actor:
        model:
          path: "/path/to/student"
        # ... student hyperparameters

    teacher_service:
      identifier: "teacher"
      role: "teacher"
      backend: http
      url: "http://${oc.env:API_SERVER_URL}:8000"
      resource:
        world_size: 8
      actor:
        model:
          path: "/path/to/teacher"
        # ... teacher hyperparameters
```

## Training Service Options

### Service Name

**Type:** `string`
**Default:** User-defined (e.g., `main_actor`, `student_service`)

Top-level key for each training service. Choose descriptive names.

### identifier

**Type:** `string`
**Default:** Required

Model identifier for the service.

**Note:** Previously named `model_tag` (deprecated but still supported).

### role

**Type:** `string`
**Options:** `"actor"`, `"teacher"`
**Default:** `"actor"` (auto-inferred for single service)

Service role in training.

- `actor`: Primary training service (required)
- `teacher`: Teacher model for distillation (optional)

### backend

**Type:** `string`
**Options:** `"http"`, `"weaver"`, `"tinker"`, `"mock"`
**Default:** `"http"`

Training service backend.

- `http`: NexTrainer HTTP API
- `weaver`: Weaver training service
- `tinker`: Tinker cloud service
- `mock`: Mock trainer for testing

### url

**Type:** `string`
**Default:** Required (for http/weaver backends)

Training service URL.

**Example:**
```yaml
url: "http://${oc.env:API_SERVER_URL}:8000"
```

### resource.world_size

**Type:** `int`
**Default:** Required

Number of GPUs for training.

**Examples:**
```yaml
resource:
  # Single GPU
  world_size: 1

  # 8 GPUs
  world_size: 8

  # Multi-node (32 GPUs)
  world_size: 32
```

### resource.gpus_per_pod

**Type:** `int`
**Default:** Same as `world_size`

GPUs per pod for distributed training.

## Actor Configuration (NexTrainer)

### checkpoint_manager

**Type:** `string`
**Options:** `"dcp"`, `"torch"`
**Default:** `"dcp"`

Checkpoint format.

- `dcp`: Distributed checkpoint (recommended)
- `torch`: PyTorch checkpoint

### model.path

**Type:** `string`
**Default:** Required

Model weights path.

**Example:**
```yaml
model:
  path: "${oc.env:NEXRL_MODEL_PATH}/Qwen/Qwen3-8B"
```

### model.use_remove_padding

**Type:** `bool`
**Default:** `true`

Remove padding tokens for efficiency.

### Training Hyperparameters

#### ppo_mini_batch_size

**Type:** `int`
**Default:** Required

Number of samples per mini-batch.

**Guidelines:**
```yaml
# Small model
ppo_mini_batch_size: 16

# Medium model
ppo_mini_batch_size: 28

# Large model
ppo_mini_batch_size: 64
```

#### ppo_micro_batch_size

**Type:** `int`
**Default:** Required

Number of samples per micro-batch (gradient accumulation).

**Guidelines:**
```yaml
# High memory
ppo_micro_batch_size: 8

# Medium memory
ppo_micro_batch_size: 4

# Low memory
ppo_micro_batch_size: 2
```

#### grad_clip

**Type:** `float`
**Default:** `1.0`

Gradient clipping threshold.

#### clip_ratio

**Type:** `float`
**Default:** `0.2`

PPO clipping ratio.

**Common Values:**
```yaml
clip_ratio: 0.2  # Standard
clip_ratio: 0.1  # Conservative
clip_ratio: 0.3  # Aggressive
```

#### entropy_coeff

**Type:** `float`
**Default:** `1e-4`

Entropy coefficient for exploration.

#### ppo_epochs

**Type:** `int`
**Default:** `1`

Number of PPO epochs per batch.

## Optimizer Configuration

### lr

**Type:** `float`
**Default:** Required

Learning rate.

**Guidelines:**
```yaml
# Small model
lr: 5e-6

# Medium model
lr: 2e-6

# Large model
lr: 1e-6
```

### lr_warmup_steps_ratio

**Type:** `float`
**Default:** `0.0`

Warmup steps as ratio of total steps.

**Examples:**
```yaml
# No warmup
lr_warmup_steps_ratio: 0.0

# 10% warmup
lr_warmup_steps_ratio: 0.1
```

### warmup_style

**Type:** `string`
**Options:** `"constant"`, `"linear"`, `"cosine"`
**Default:** `"constant"`

Learning rate schedule.

## FSDP Configuration

### wrap_policy.min_num_params

**Type:** `int`
**Default:** `0`

Minimum parameters for FSDP wrapping.

### param_offload

**Type:** `bool`
**Default:** `false`

Offload parameters to CPU.

### grad_offload

**Type:** `bool`
**Default:** `false`

Offload gradients to CPU.

## Common Patterns

### Self-Hosted Development

```yaml
service:
  inference_service:
    identifier: "default"
    base_url: "http://localhost:8000"
    model: "debug-model"
    backend: sglang
    max_retries: 1

  train_service:
    main_actor:
      identifier: "default"
      role: "actor"
      backend: http
      url: "http://localhost:8001"
      resource:
        world_size: 1
      actor:
        ppo_mini_batch_size: 8
        ppo_micro_batch_size: 2
```

### Production Self-Hosted

```yaml
service:
  inference_service:
    identifier: "default"
    base_url: "${oc.env:INFERENCE_BASE_URL}"
    model: "my-model"
    backend: sglang
    max_retries: 3
    freeze_for_weight_sync: true
    weight_type: "sglang_nckpt"

  train_service:
    main_actor:
      identifier: "default"
      role: "actor"
      backend: http
      url: "http://${oc.env:API_SERVER_URL}:8000"
      resource:
        world_size: 8
      actor:
        checkpoint_manager: dcp
        ppo_mini_batch_size: 28
        ppo_micro_batch_size: 4
        optim:
          lr: 2e-6
```

### OpenAI API

```yaml
service:
  inference_service:
    identifier: "default"
    api_key: "${oc.env:OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4"
    max_tokens: 4096
    backend: openai
    max_retries: 3
    freeze_for_weight_sync: false
```

### On-Policy Distillation

```yaml
service:
  train_service:
    student_service:
      identifier: "student"
      role: "actor"
      backend: http
      url: "http://localhost:8000"
      resource:
        world_size: 8
      actor:
        model:
          path: "/path/to/student"
        optim:
          lr: 2e-6

    teacher_service:
      identifier: "teacher"
      role: "teacher"
      backend: http
      url: "http://localhost:8000"
      resource:
        world_size: 8
      actor:
        model:
          path: "/path/to/teacher"
        optim:
          lr: 1e-6
```

## Troubleshooting

### Connection Refused

**Symptom:** Cannot connect to inference/training service

**Solutions:**
- Verify service is running
- Check URL and port
- Verify network connectivity
- Check firewall rules

### Authentication Failed

**Symptom:** API key errors

**Solutions:**
- Verify `api_key` is set correctly
- Check environment variable value
- Verify API key is valid

### Out of Memory (Training)

**Symptom:** OOM during training step

**Solutions:**
- Reduce `ppo_mini_batch_size`
- Reduce `ppo_micro_batch_size`
- Enable FSDP offloading
- Use smaller model or shorter sequences

### Weight Sync Failures

**Symptom:** Weight synchronization errors

**Solutions:**
- Check `weight_type` matches backend
- Verify `sync_weight_path` is accessible
- Check disk space
- Review weight sync controller logs

## Related Documentation

- [Inference Service](../07-services/inference-service.md) - LLM service integration
- [Training Service](../07-services/training-service.md) - Training service client
- [Service Holders](../07-services/service-holders.md) - Tinker/Weaver holders
- [Weight Synchronization](../08-features/weight-synchronization.md) - Weight sync coordination
- [Complete Config](./complete-config.md) - Full configuration example
