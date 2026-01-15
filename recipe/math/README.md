# Single Turn Math Task

This folder contains recipes for training models on single-turn math problems using different training backends.

## Task Description

Single-turn math is a task where the model receives a math problem and must generate the correct solution in a single response. The reward is calculated using rule-based evaluation comparing the generated answer to the ground truth.

## Configuration Files

- **`common.yaml`** - Shared configuration for all training modes
- **`weaver.yaml`** - Weaver cloud training configuration
- **`tinker.yaml`** - Tinker cloud training configuration
- **`self_hosted.yaml`** - Self-hosted training configuration

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
- **Self-hosted:** `export API_SERVER_URL=10.0.0.1`, `export INFERENCE_BASE_URL=http://localhost:8000`, `export EXPERIMENT_PATH=/path/to/output`

**Optional:**
- `export FEISHU_WEBHOOK_URL=https://your-webhook-url` (for notifications)

No environment setup script is needed for this task as it doesn't require agent-specific services.

## Data Format

Training data should be in Parquet format with the following columns:
- `prompt`: The math problem text
- `ground_truth`: The correct answer

Expected files:
```
${NEXRL_DATA_PATH}/math/
├── level1_train.parquet
├── level2_train.parquet
├── level3_train.parquet
├── level4_train.parquet
├── level5_train.parquet
├── level1_test.parquet
├── level2_test.parquet
├── level3_test.parquet
├── level4_test.parquet
└── level5_test.parquet
```

## Usage

### Weaver Mode
```bash
python -m nexrl.cli.train --config-path recipe/math --config-name weaver
```

### Tinker Mode
```bash
python -m nexrl.cli.train --config-path recipe/math --config-name tinker
```

### Self-hosted Mode
```bash
python -m nexrl.cli.train --config-path recipe/math --config-name self_hosted
```

## Customization

To customize the configuration:

1. Copy the relevant YAML file
2. Modify hyperparameters as needed
3. Run with your custom config:
```bash
python -m nexrl.cli.train --config-path /path/to/your/config --config-name your_config
```

## Key Configuration Parameters

- `batch_size`: Number of prompts per batch (default: 28)
- `rollout_repeat_n`: Responses per prompt (default: 8)
- `total_train_steps`: Total training steps (default: 100)
- `num_workers`: Parallel rollout workers (default: 224)
- `temperature`: Sampling temperature (default: 1.0)
- `learning_rate`: Training learning rate (default: 1e-4)
