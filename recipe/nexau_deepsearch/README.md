# NexAU DeepSearch Task

This folder contains recipes for training NexAU agents on the DeepSearch task using different training backends.

## Task Description

DeepSearch is an agentic task where the model must conduct web research to answer complex queries. The agent uses the `web_search` tool to gather information from the internet and must provide a final answer in the format `<answer>Your answer</answer>`.

Rewards are calculated using F1 score between the agent's answer and the ground truth.

## Configuration Files

- **`common.yaml`** - Shared configuration for all training modes
- **`weaver.yaml`** - Weaver cloud training configuration
- **`tinker.yaml`** - Tinker cloud training configuration
- **`self_hosted.yaml`** - Self-hosted training configuration

## Agent Workspace

- **`agent_workspace/agent_config.yaml`** - NexAU agent configuration
- **`agent_workspace/evaluator.py`** - Reward calculation (F1 score)
- **`agent_workspace/web_tool.py`** - Web search tool implementation
- **`agent_workspace/tools/WebSearch.yaml`** - Tool schema definition

## Environment Setup

Each training mode has its own environment script with the specific variables it needs:

- **`weaver.env.sh`** - Weaver mode environment (includes WEAVER_API_KEY)
- **`tinker.env.sh`** - Tinker mode environment (includes TINKER_API_KEY)
- **`self_hosted.env.sh`** - Self-hosted mode environment

Edit the appropriate `.env.sh` file for your mode and fill in the required values:

**Required for DeepSearch:**
- `SERPER_API_KEY` - Web search API key (get from serper.dev)
- `LLM_BASE_URL` - Inference service URL
- `JUDGE_LLM_BASE_URL` - Judge LLM service URL (for evaluation)

**Mode-specific:**
- Weaver: `WEAVER_API_KEY`
- Tinker: `TINKER_API_KEY`

**Optional:**
- LangFuse credentials for monitoring
- HTTP_PROXY/HTTPS_PROXY for web access

## Data Format

Training data should be in Parquet format with the following columns:
- `prompt`: The research query
- `ground_truth`: The expected answer (can contain multiple answers separated by `<|answer_split|>`)

Expected files:
```
${NEXRL_DATA_PATH}/deepsearch/
├── train_data.parquet
└── valid_data.parquet
```

## Usage

### Weaver Mode
```bash
# 1. Edit the environment script with your values
nano recipe/nexau_deepsearch/weaver.env.sh

# 2. Set up environment variables
source recipe/nexau_deepsearch/weaver.env.sh

# 3. Run training
python -m nexrl.cli.train --config-path recipe/nexau_deepsearch --config-name weaver
```

### Tinker Mode
```bash
# 1. Edit the environment script with your values
nano recipe/nexau_deepsearch/tinker.env.sh

# 2. Set up environment variables
source recipe/nexau_deepsearch/tinker.env.sh

# 3. Run training
python -m nexrl.cli.train --config-path recipe/nexau_deepsearch --config-name tinker
```

### Self-hosted Mode
```bash
# 1. Edit the environment script with your values
nano recipe/nexau_deepsearch/self_hosted.env.sh

# 2. Set up environment variables
source recipe/nexau_deepsearch/self_hosted.env.sh

# 3. Run training
python -m nexrl.cli.train --config-path recipe/nexau_deepsearch --config-name self_hosted
```

## API Requirements

### Serper API
Sign up at [serper.dev](https://serper.dev/) to get your API key for web search functionality.

### LangFuse (Optional)
Sign up at [langfuse.com](https://langfuse.com/) for agent tracing and monitoring.

## Agent Configuration

The agent is configured with:
- **Max context tokens**: 20,000
- **Max iterations**: 10
- **Temperature**: 0.7
- **Tool**: web_search (Serper API)
- **Answer format**: `<answer>Your answer</answer>`

## Key Configuration Parameters

- `batch_size`: Number of prompts per batch (default: 16)
- `rollout_repeat_n`: Responses per prompt (default: 8)
- `total_train_steps`: Total training steps (default: 200)
- `num_workers`: Parallel rollout workers (default: 128)
- `temperature`: Agent sampling temperature (default: 0.7)
- `learning_rate`: Training learning rate (default: 1e-4)
- `max_prompt_length`: Max input tokens (default: 20,000)
- `max_response_length`: Max output tokens (default: 8,192)

## Customizing the Agent

To customize the agent behavior:

1. Edit `agent_workspace/agent_config.yaml` to modify:
   - System prompt
   - Temperature
   - Max iterations
   - Tool configurations

2. Edit `agent_workspace/evaluator.py` to modify:
   - Reward function
   - Evaluation metrics

3. Edit `agent_workspace/web_tool.py` to add new tools or modify search behavior
