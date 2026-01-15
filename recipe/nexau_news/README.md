# NexAU News Task

This folder contains recipes for training NexAU agents on the News recommendation/filtering task using different training backends.

## Task Description

The News task is an agentic classification task where the model acts as a news filtering assistant for investment park managers. The agent must evaluate news articles and decide whether they should be recommended based on importance, relevance, timeliness, and other business criteria.

The agent returns answers in the format `<answer>True</answer>` or `<answer>False</answer>` with reasoning in `<think>...</think>` tags.

Rewards are calculated using accuracy, precision, recall, and F1 score metrics.

## Configuration Files

- **`common.yaml`** - Shared configuration for all training modes
- **`weaver.yaml`** - Weaver cloud training configuration
- **`tinker.yaml`** - Tinker cloud training configuration
- **`self_hosted.yaml`** - Self-hosted training configuration

## Agent Workspace

- **`agent_workspace/agent_config.yaml`** - NexAU agent configuration (Chinese system prompt)
- **`agent_workspace/evaluator.py`** - Reward calculation (accuracy, precision, recall, F1)
- **`agent_workspace/news_rollout_worker.py`** - Custom rollout worker for news-specific query formatting

## Environment Setup

Each training mode has its own environment script with the specific variables it needs:

- **`weaver.env.sh`** - Weaver mode environment (includes WEAVER_API_KEY)
- **`tinker.env.sh`** - Tinker mode environment (includes TINKER_API_KEY)
- **`self_hosted.env.sh`** - Self-hosted mode environment

Edit the appropriate `.env.sh` file for your mode and fill in the required values:

**Required for News:**
- `LLM_BASE_URL` - Inference service URL
- `JUDGE_LLM_BASE_URL` - Judge LLM service URL (for evaluation)

**Mode-specific:**
- Weaver: `WEAVER_API_KEY`
- Tinker: `TINKER_API_KEY`

**Optional:**
- LangFuse credentials for monitoring

## Data Format

Training data should be in Parquet format with the following columns:
- `prompt`: The news article text and metadata
- `标注`: Annotation label ("必须推", "都可以", "不要推") or
- `通过`: Boolean field indicating whether to recommend

Expected files:
```
${NEXRL_DATA_PATH}/news/
├── train.parquet
└── test.parquet
```

## Usage

### Weaver Mode
```bash
# 1. Edit the environment script with your values
nano recipe/nexau_news/weaver.env.sh

# 2. Set up environment variables
source recipe/nexau_news/weaver.env.sh

# 3. Run training
python -m nexrl.cli.train --config-path recipe/nexau_news --config-name weaver
```

### Tinker Mode
```bash
# 1. Edit the environment script with your values
nano recipe/nexau_news/tinker.env.sh

# 2. Set up environment variables
source recipe/nexau_news/tinker.env.sh

# 3. Run training
python -m nexrl.cli.train --config-path recipe/nexau_news --config-name tinker
```

### Self-hosted Mode
```bash
# 1. Edit the environment script with your values
nano recipe/nexau_news/self_hosted.env.sh

# 2. Set up environment variables
source recipe/nexau_news/self_hosted.env.sh

# 3. Run training
python -m nexrl.cli.train --config-path recipe/nexau_news --config-name self_hosted
```

## Agent Configuration

The agent is configured with:
- **Max context tokens**: 100,000 (large context for full news articles)
- **Temperature**: 0.7
- **Tool call mode**: OpenAI format
- **System prompt**: Chinese instructions for news filtering

## Evaluation Metrics

The evaluator calculates multiple metrics:

- **accuracy_strict**: Exact match with ground truth (all cases)
- **accuracy_flex**: Flexible accuracy accounting for "都可以" (both OK) labels
- **precision**: True positives / (True positives + False positives)
- **recall**: True positives / (True positives + False negatives)
- **f1**: Harmonic mean of precision and recall

## Validation

This task uses **validation before training** (`validate_before_train: true`) to establish baseline performance.

## Key Configuration Parameters

- `batch_size`: Number of prompts per batch (default: 16)
- `rollout_repeat_n`: Responses per prompt (default: 8)
- `total_train_steps`: Total training steps (default: 200)
- `num_workers`: Parallel rollout workers (default: 128)
- `temperature`: Agent sampling temperature (default: 0.7)
- `learning_rate`: Training learning rate (default: 1e-4)
- `max_prompt_length`: Max input tokens (default: 100,000)
- `max_response_length`: Max output tokens (default: 8,192)

## Custom Rollout Worker

This task uses a custom rollout worker (`NewsNexAURolloutWorker`) that handles:
- Special data field parsing (Chinese field names)
- News-specific query formatting
- Custom prompt construction

The custom worker is defined in `agent_workspace/news_rollout_worker.py` and is automatically loaded via:
```yaml
custom_rollout_worker_module_path: "agent_workspace/news_rollout_worker.py"
custom_rollout_worker_class_name: "NewsNexAURolloutWorker"
```

## Customizing the Agent

To customize the agent behavior:

1. Edit `agent_workspace/agent_config.yaml` to modify:
   - System prompt (currently in Chinese)
   - Temperature
   - Max context tokens

2. Edit `agent_workspace/evaluator.py` to modify:
   - Reward function
   - Evaluation metrics
   - Answer extraction logic

3. Edit `agent_workspace/news_rollout_worker.py` to modify:
   - Data field parsing
   - Query formatting
   - Prompt construction
