# Pig Latin Task

This folder contains recipes for supervised learning on the Pig Latin translation task using different training backends.

## Task Description

Pig Latin is a supervised learning task where the model learns to translate English text to Pig Latin. This is a language game transformation task that serves as a simple test case for sequence-to-sequence learning.

**Important**: This task uses supervised learning with cross-entropy loss, not reinforcement learning. No LLM inference is required during training as ground truth labels are used directly.

## Configuration Files

- **`common.yaml`** - Shared configuration for all training modes
- **`weaver.yaml`** - Weaver cloud training configuration
- **`tinker.yaml`** - Tinker cloud training configuration
- **`self_hosted.yaml`** - Self-hosted training configuration (requires custom implementation)

## Configuration Requirements

This task uses environment variables in the YAML configs. You need to set:

**Data and Model Paths (required for all modes):**
```bash
export NEXRL_DATA_PATH=/path/to/your/data
export NEXRL_MODEL_PATH=/path/to/models/huggingface.co
```

**Mode-specific (set before running):**
- **Weaver:** `export WEAVER_API_KEY=sk-your-weaver-api-key`
- **Tinker:** `export TINKER_API_KEY=tml-your-tinker-api-key`
- **Self-hosted:** `export API_SERVER_URL=10.0.0.1`, `export EXPERIMENT_PATH=/path/to/output`

**Optional:**
- `export FEISHU_WEBHOOK_URL=https://your-webhook-url` (for notifications)

No environment setup script is needed for this task as it's a simple supervised learning task without agent services.

## Data Format

Training data should be in JSONL format with the following fields:
- `input`: The English text
- `output`: The Pig Latin translation

Example:
```json
{"input": "hello world", "output": "ellohay orldway"}
{"input": "pig latin", "output": "igpay atinlay"}
```

Expected files:
```
${NEXRL_DATA_PATH}/pig_latin/
└── train.jsonl
```

## Usage

### Weaver Mode (Recommended)
```bash
python -m nexrl.cli.train --config-path recipe/pig_latin --config-name weaver
```

### Tinker Mode
```bash
python -m nexrl.cli.train --config-path recipe/pig_latin --config-name tinker
```

### Self-hosted Mode
```bash
# Note: Requires custom training service implementation for cross-entropy loss
python -m nexrl.cli.train --config-path recipe/pig_latin --config-name self_hosted
```

## Key Differences from RL Tasks

- **No LLM Inference**: `need_llm_inference: false`
- **Cross-Entropy Loss**: Uses `remote_api_cross_entropy` trainer instead of GRPO
- **Single Rollout**: `rollout_repeat_n: 1` (only need one trajectory per example)
- **Rollout Type**: Uses `pig_latin` rollout worker that constructs trajectories from ground truth

## Key Configuration Parameters

- `batch_size`: Number of examples per batch (default: 4)
- `total_train_steps`: Total training steps (default: 6)
- `num_workers`: Parallel rollout workers (default: 4)
- `learning_rate`: Training learning rate (default: 1e-4)
- `max_prompt_length`: Max input tokens (default: 512)
- `max_response_length`: Max output tokens (default: 512)
