# Recipe Development

Best practices for creating task-specific recipes.

## Recipe Structure

Standard recipe organization:

```
recipe/my_task/
├── common.yaml                 # Shared configuration
├── self_hosted.yaml           # Self-hosted overrides
├── tinker.yaml                # Tinker backend overrides
├── weaver.yaml                # Weaver backend overrides
├── self_hosted.env.sh         # Environment setup (self-hosted)
├── tinker.env.sh              # Environment setup (Tinker)
├── weaver.env.sh              # Environment setup (Weaver)
├── README.md                  # Recipe documentation
└── agent_workspace/           # Agent-specific files (if using NexAU)
    ├── agent_config.yaml      # NexAU agent configuration
    ├── evaluator.py          # Task evaluator
    ├── custom_worker.py      # Optional: custom worker
    └── tools/                # Optional: custom tools
        └── MyTool.yaml
```

## Configuration Organization

### common.yaml

Shared configuration across all backends:

```yaml
project_name: "NexRL-MyTask"
experiment_name: "my-task-training"
launch_mode: "local"

data:
  type: "torch"
  seed: 42
  data_files:
    - "${oc.env:NEXRL_DATA_PATH}/my_task/train.parquet"
  batch_size: 32
  rollout_repeat_n: 8
  max_prompt_length: 4096
  max_response_length: 8192
  tokenizer_path: "${oc.env:NEXRL_MODEL_PATH}/model"

rollout_worker:
  type: "nexau"
  num_workers: 128
  temperature: 0.7

trainer:
  total_train_steps: 200

weight:
  type: "default"
  sync_mode: "sync"

validate:
  validate_before_train: false
```

### self_hosted.yaml

Self-hosted specific configuration:

```yaml
defaults:
  - common
  - _self_

trainer:
  type: "self_hosted_grpo"
  checkpoint_path: "${oc.env:EXPERIMENT_PATH}/ckpt"
  sync_weight_path: "${oc.env:EXPERIMENT_PATH}/sync_weight"
  algorithm:
    type: "grpo"
    do_old_log_prob_compute: true

service:
  inference_service:
    base_url: "${oc.env:INFERENCE_BASE_URL}"
    backend: sglang

  train_service:
    backend: http
    url: "http://${oc.env:API_SERVER_URL}:8000"
```

## Environment Setup Scripts

### self_hosted.env.sh

```bash
#!/bin/bash

# Paths
export NEXRL_PATH="/path/to/nexrl"
export NEXRL_DATA_PATH="/path/to/data"
export NEXRL_MODEL_PATH="/path/to/models"
export EXPERIMENT_PATH="/path/to/experiments/my_task"

# Services
export INFERENCE_BASE_URL="http://localhost:8000"
export API_SERVER_URL="localhost"

# Logging
export WANDB_KEY="your_key"
export WANDB_HOST="https://api.wandb.ai"

# Create directories
mkdir -p "${EXPERIMENT_PATH}/ckpt"
mkdir -p "${EXPERIMENT_PATH}/sync_weight"
```

## Evaluator Implementation

### Task-Specific Evaluator

```python
from nexrl.rollout_worker import (
    Evaluator,
    BaseEvaluationTarget,
    EvaluationRunResult
)
from typing import Any

class MyTaskEvaluator(Evaluator):
    """Evaluator for my specific task."""

    def evaluate(
        self,
        data: dict[str, Any],
        evaluation_target: BaseEvaluationTarget
    ) -> EvaluationRunResult:
        """
        Evaluate agent output.

        Args:
            data: Original data item with ground truth
            evaluation_target: Agent output with final_answer

        Returns:
            EvaluationRunResult with reward and metrics
        """
        # Extract agent answer
        agent_answer = evaluation_target.final_answer.strip()
        ground_truth = data.get("answer", "").strip()

        # Compute reward (0.0 to 1.0)
        reward = self._compute_reward(agent_answer, ground_truth)

        # Compute additional metrics (must be scalar floats)
        metrics = {
            "exact_match": float(agent_answer == ground_truth),
            "answer_length": float(len(agent_answer)),
        }

        # Store extra information
        extra_info = {
            "agent_answer": agent_answer,
            "ground_truth": ground_truth,
        }

        return EvaluationRunResult(
            reward=reward,
            ground_truth=ground_truth,
            metrics=metrics,
            extra_info=extra_info
        )

    def _compute_reward(self, prediction: str, ground_truth: str) -> float:
        """Compute reward for prediction."""
        if prediction == ground_truth:
            return 1.0
        else:
            return 0.0
```

### Fuzzy Matching Evaluator

```python
from difflib import SequenceMatcher

class FuzzyMatchEvaluator(Evaluator):
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def evaluate(self, data, evaluation_target):
        agent_answer = evaluation_target.final_answer.lower().strip()
        ground_truth = data["answer"].lower().strip()

        # Fuzzy match
        similarity = SequenceMatcher(None, agent_answer, ground_truth).ratio()
        reward = 1.0 if similarity >= self.threshold else 0.0

        metrics = {
            "similarity": float(similarity),
            "exact_match": float(agent_answer == ground_truth),
        }

        return EvaluationRunResult(
            reward=reward,
            ground_truth=ground_truth,
            metrics=metrics
        )
```

## Custom Worker Implementation

### Task-Specific Query Formatting

```python
from nexrl.rollout_worker import BaseNexAURolloutWorker
from typing import Any

class MyTaskWorker(BaseNexAURolloutWorker):
    """Custom worker for my task."""

    def format_task_query(self, data_item: dict[str, Any]) -> str:
        """Format data item into task query."""
        # Extract relevant fields
        context = data_item.get("context", "")
        question = data_item.get("question", "")

        # Format prompt
        query = f"""Context: {context}

Question: {question}

Please provide your answer:"""

        return query
```

### Custom Loss Masking

```python
class MyTaskWorker(BaseNexAURolloutWorker):
    def get_train_loss_mask(self, trajectory_infos: list[dict]) -> list[bool]:
        """Custom loss mask for training."""
        mask = []
        for traj_info in trajectory_infos:
            # Only train on final answer, not intermediate steps
            is_final_answer = traj_info.get("is_final", False)
            mask.append(is_final_answer)
        return mask
```

## Data Preparation

### Data Format

Use Parquet format for efficiency:

```python
import pandas as pd

# Prepare training data
train_data = [
    {
        "prompt": "What is 2+2?",
        "answer": "4",
        "difficulty": "easy"
    },
    # ... more examples
]

df = pd.DataFrame(train_data)
df.to_parquet("train.parquet")
```

### Data Validation

Validate data before training:

```python
def validate_data(parquet_file: str):
    df = pd.read_parquet(parquet_file)

    # Check required columns
    assert "prompt" in df.columns, "Missing 'prompt' column"
    assert "answer" in df.columns, "Missing 'answer' column"

    # Check data quality
    assert len(df) > 0, "Empty dataset"
    assert df["prompt"].notna().all(), "Null prompts found"

    print(f"Dataset valid: {len(df)} examples")
```

## Testing

### Local Testing

Test recipe locally before distributed run:

```yaml
# test_config.yaml
defaults:
  - common
  - _self_

launch_mode: "local"

data:
  data_files:
    - "data/small_test.parquet"  # Small test set
  batch_size: 4

rollout_worker:
  num_workers: 2
  type: "mock"  # Mock worker for fast testing

trainer:
  type: "self_hosted_grpo"
  total_train_steps: 5  # Few steps
  save_freq: 0
```

### Evaluator Testing

```python
def test_evaluator():
    evaluator = MyTaskEvaluator()

    # Test correct answer
    data = {"answer": "4"}
    target = BaseEvaluationTarget(final_answer="4")
    result = evaluator.evaluate(data, target)
    assert result.reward == 1.0

    # Test incorrect answer
    target = BaseEvaluationTarget(final_answer="5")
    result = evaluator.evaluate(data, target)
    assert result.reward == 0.0
```

## Documentation

### README.md

Document recipe usage:

```markdown
# My Task Recipe

Task-specific training recipe for [describe task].

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare data:
   ```bash
   python prepare_data.py
   ```

3. Set environment:
   ```bash
   source self_hosted.env.sh
   ```

## Running

### Self-Hosted

```bash
nexrl -m self-hosted -c recipe/my_task/self_hosted.yaml --run-nexrl
```

### Tinker

```bash
nexrl -m tinker -c recipe/my_task/tinker.yaml --run-nexrl
```

## Results

Expected performance: [describe expected results]
```

## Common Patterns

### Multi-File Data

```yaml
data:
  data_files:
    - "${NEXRL_DATA_PATH}/my_task/shard_001.parquet"
    - "${NEXRL_DATA_PATH}/my_task/shard_002.parquet"
    - "${NEXRL_DATA_PATH}/my_task/shard_003.parquet"
```

### Task-Specific Tools

```python
# agent_workspace/tools/custom_tool.py
class MyCustomTool:
    def __init__(self, config):
        self.config = config

    def execute(self, query: str) -> str:
        """Execute tool logic."""
        result = self._process(query)
        return result
```

### Staged Training

```yaml
# Stage 1: Easy examples
data:
  data_files:
    - "${NEXRL_DATA_PATH}/my_task/easy.parquet"

trainer:
  total_train_steps: 100

# Stage 2: Hard examples
# (create separate config for hard examples)
```

## Troubleshooting

### Low Rewards

**Check:**
- Evaluator logic correct
- Ground truth format matches
- Task is learnable
- Prompt format clear

### High Memory Usage

**Solutions:**
- Reduce `batch_size`
- Reduce `rollout_repeat_n`
- Reduce `max_prompt_length` / `max_response_length`
- Use data filtering

### Slow Training

**Check:**
- `num_workers` matches cluster capacity
- LLM service responsive
- Data loading efficient
- Network not bottlenecked

## Related Documentation

- [Recipe Structure](../09-recipes/recipe-structure.md) - Directory layout
- [Agent Configuration](../09-recipes/agent-configuration.md) - NexAU agents
- [Recipe Configuration](../09-recipes/recipe-configuration.md) - YAML config
- [Evaluators](../05-rollout-workers/evaluators.md) - Evaluation patterns
