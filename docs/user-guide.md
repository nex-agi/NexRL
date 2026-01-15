# NexRL User Guide for Algorithm Developers

This guide helps you develop and integrate new RL algorithms and tasks into the NexRL framework. It covers recipe preparation, rollout worker implementation, evaluator design, and trainer architecture.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Deployment Modes](#deployment-modes)
3. [Recipe Structure](#recipe-structure)
4. [Implementing Rollout Workers](#implementing-rollout-workers)
5. [Implementing Evaluators](#implementing-evaluators)
6. [Training Backends & Trainers](#training-backends--trainers)
7. [Configuration Guide](#configuration-guide)
8. [Best Practices](#best-practices)
9. [Common Patterns](#common-patterns)

---

## Quick Start

### Prerequisites

Before starting, ensure you have:
- Python 3.12+
- kubectl configured with access to a Kubernetes cluster
- [Volcano Scheduler](https://github.com/volcano-sh/volcano) installed in the cluster
- High-performance shared storage (e.g., NFS, GPFS)

Install NexRL:

```bash
git clone https://github.com/nex-agi/NexRL.git
cd NexRL
pip install -e .
```

### Zero-Setup Quick Start

Run immediately with built-in defaults (uses public images and /tmp storage):

```bash
# Self-hosted mode
nexrl -m self-hosted -c recipe/my_task/my_task.yaml --run-nexrl

# Training-service mode (with Tinker/Weaver)
nexrl -m training-service -c recipe/my_task/my_task.yaml --run-nexrl
```

### Minimal Setup for a New Task

1. **Create recipe directory structure**:
```bash
mkdir -p recipe/my_task/agent_workspace
```

2. **Create configuration** (`recipe/my_task/my_task.yaml`):
```yaml
# Import common configuration
defaults:
  - self_hosted_nexau_common
  - _self_

project_name: "NexRL-MyTask"
experiment_name: "my-task-training"

# Task-specific data
data:
  data_files:
    - "/path/to/train.parquet"

# Rollout worker configuration
rollout_worker:
  type: "nexau"
  nexau_agent_config_path: "recipe/my_task/agent_workspace/agent_config.yaml"
  evaluator_module_path: "recipe/my_task/agent_workspace/evaluator.py:MyEvaluator"
  task_name: "my_task"
```

3. **Create agent config** (`recipe/my_task/agent_workspace/agent_config.yaml`):
```yaml
type: agent
name: my_task_agent
max_context_tokens: 100000
system_prompt: "You are a helpful assistant..."
system_prompt_type: string
llm_config:
  temperature: 0.7
  max_tokens: 8192
  api_type: openai_chat_completion

tracers:
  - import: nexau.archs.tracer.adapters.in_memory:InMemoryTracer
```

4. **Create evaluator** (`recipe/my_task/agent_workspace/evaluator.py`):
```python
from nexrl.rollout_worker import Evaluator, BaseEvaluationTarget, EvaluationRunResult

class MyEvaluator(Evaluator):
    def evaluate(self, data, evaluation_target):
        # Your evaluation logic
        reward = 1.0 if evaluation_target.final_answer == data["answer"] else 0.0
        return EvaluationRunResult(reward=reward)
```

5. **Run training**:
```bash
# With the CLI (recommended)
nexrl -m self-hosted -c recipe/my_task/my_task.yaml --run-nexrl

# Or using Python directly
python -m nexrl.main recipe=my_task/my_task
```

---

## Deployment Modes

NexRL supports two deployment modes via the `cli/` launcher:

### Self-Hosted Mode

Runs all training infrastructure (training backend, inference service) on your Kubernetes cluster. Full control over the entire stack.

```bash
nexrl --mode self-hosted --train-config recipe/my_task.yaml --run-nexrl
```

**Components launched:**
- Training API server and GPU workers
- Inference service (SGLang/vLLM)
- NexRL driver pod

**Use when:**
- You want full control over training infrastructure
- You have GPU resources available
- You need to customize training backend configuration

### Training-Service Mode

Uses external training services (Tinker/Weaver). Only launches the NexRL driver pod locally.

```bash
nexrl --mode training-service --train-config recipe/my_task.yaml --run-nexrl
```

**Components launched:**
- NexRL driver pod
- Inference service (uses external training API)

**Use when:**
- You want to use managed training services
- You want lighter resource footprint
- You have access to Tinker or Weaver training APIs

### CLI Options

```bash
nexrl [OPTIONS]

Required:
  -m, --mode              Deployment mode: 'self-hosted' or 'training-service'
  -c, --train-config      Path to training config YAML

Optional:
  -r, --run-nexrl         Auto-start training immediately
  -t, --tag               Custom tag for job names
  --inference-url URL     [self-hosted] Use existing inference service (skip launching)
```

**Examples:**

```bash
# Self-hosted with auto-start
nexrl -m self-hosted -c recipe/math/config.yaml -r

# Training-service with custom tag
nexrl -m training-service -c recipe/math/tinker.yaml -r -t exp-v2

# Self-hosted with external inference
nexrl -m self-hosted -c recipe/math/config.yaml -r --inference-url my-service:8000
```

### Configuration Setup

NexRL supports three configuration methods with automatic fallback:

| Method | When to Use | Setup |
|--------|-------------|-------|
| **Built-in Defaults** | Testing, demos | None - just run! |
| **Environment Variables** | Development | `source cli/setup_env.sh` |
| **Kubernetes ConfigMaps** | Production, teams | `kubectl apply -f cli/setup/` |

**Development Setup (Environment Variables):**

```bash
# Quick setup using provided script
source cli/setup_env.sh

# Or set manually
export NEXRL_STORAGE_PATH="/your/persistent/storage"
export NEXRL_WORKER_IMAGE="your-registry/nexrl:tag"
export NEXRL_CONTROLLER_IMAGE="your-registry/nexrl:tag"
export WANDB_KEY="your-wandb-key"
```

**Production Setup (ConfigMaps):**

```bash
# One-time cluster setup
cd cli/setup
kubectl apply -f 01-namespace.yaml
kubectl apply -f 02-admin-config.yaml  # Edit first!
kubectl apply -f 03-user-config.yaml   # Edit first!
kubectl apply -f 04-volcano-queue.yaml
kubectl apply -f 09-nexrl-job-rbac.yaml
```

See [`cli/README.md`](../cli/README.md) and [`cli/setup/README.md`](../cli/setup/README.md) for detailed configuration instructions.

---

## Recipe Structure

### Directory Layout

A complete recipe follows this structure:

```
recipe/
└── my_task_name/
    ├── my_task_name.yaml          # Main configuration file
    ├── my_task_name.env.sh        # Environment setup (optional)
    └── agent_workspace/           # Agent-specific files
        ├── agent_config.yaml      # NexAU agent configuration
        ├── evaluator.py          # Task evaluator implementation
        ├── my_worker.py          # Custom rollout worker (optional)
        └── custom_tools.py       # Task-specific tools (optional)
```

### File Descriptions

#### 1. Main Configuration (`my_task_name.yaml`)

Defines task-specific settings:

```yaml
# Import base configuration
defaults:
  - self_hosted_nexau_common  # or tinker_nexau_common, weaver_nexau_common
  - _self_

# Project identification
project_name: "NexRL-MyTask"
experiment_name: "my-task-v1"

# Environment setup (optional)
environment:
  setup_script: "recipe/my_task/my_task.env.sh"

# Data configuration
data:
  type: "torch"
  data_files:
    - "/path/to/train_data.parquet"
  shuffle: true
  seed: 42
  max_prompt_length: ${rollout_worker.max_prompt_length}

# Rollout worker configuration
rollout_worker:
  type: "nexau"  # Use default NexAU worker

  # Optional: Use custom worker
  # custom_rollout_worker_module_path: "recipe/my_task/agent_workspace/my_worker.py"
  # custom_rollout_worker_class_name: "MyCustomWorker"

  # NexAU agent settings
  nexau_agent_config_path: "recipe/my_task/agent_workspace/agent_config.yaml"
  evaluator_module_path: "recipe/my_task/agent_workspace/evaluator.py:MyEvaluator"
  nexau_agent_workspace: "recipe/my_task/agent_workspace"  # Optional: for local imports
  task_name: "my_task"

  # Model configuration
  max_prompt_length: 16384
  tokenizer: "Qwen/Qwen2.5-7B-Instruct"

# Trainer configuration
trainer:
  type: "self_hosted_grpo"  # or "remote_api_grpo" for Tinker/Weaver
  total_train_steps: 1000
  save_freq: 100

# Algorithm configuration (for self_hosted_grpo only)
algorithm:
  type: "grpo"
  batch_size: ${trajectory_pool.batch_size}
  do_old_log_prob_compute: true
  use_kl_in_reward: false

# Validation configuration
validate:
  validate_before_train: true
  data:
    data_files:
      - "/path/to/test_data.parquet"

# Resource configuration
resource:
  inference:
    served_model_name: "my-task-model"
```

#### 2. Agent Configuration (`agent_config.yaml`)

Defines the NexAU agent behavior:

```yaml
type: agent
name: my_task_agent
max_context_tokens: 100000

# System prompt - defines agent's role and behavior
system_prompt: |
  You are an expert assistant for [task description].

  Your goal is to [objective].

  Output your answer in the following format:
  <reasoning>Your step-by-step thinking</reasoning>
  <answer>Your final answer</answer>

system_prompt_type: string  # or 'list' for multi-turn prompts

# Tool configuration (optional)
tool_call_mode: openai  # Enable tool calling

tools:
  - name: calculator
    binding: custom_tools:calculator  # Import from agent_workspace
    description: "Perform mathematical calculations"

# LLM configuration
llm_config:
  temperature: 0.7
  max_tokens: 8192
  api_type: openai_chat_completion

# Tracing configuration
tracers:
  - import: nexau.archs.tracer.adapters.in_memory:InMemoryTracer
```

**Key Points:**
- `system_prompt`: Defines agent behavior and output format
- `tools`: Optional tool bindings for agent capabilities
- `llm_config`: Model sampling parameters
- `tracers`: Execution tracking (required for NexRL)

#### 3. Environment Setup (`my_task.env.sh`)

Optional script for environment preparation:

```bash
#!/bin/bash
# Set task-specific environment variables
export MY_TASK_DATA_PATH="/path/to/data"
export MY_TASK_CONFIG_PATH="/path/to/config"

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Install task-specific dependencies
# pip install -q custom-library
```

---

## Implementing Rollout Workers

Rollout workers convert data items into trajectories for training. NexRL provides `BaseNexAURolloutWorker` with sensible defaults.

### Using Default Worker

For most tasks, the default implementation works out of the box:

```yaml
rollout_worker:
  type: "nexau"
  nexau_agent_config_path: "recipe/my_task/agent_workspace/agent_config.yaml"
  evaluator_module_path: "recipe/my_task/agent_workspace/evaluator.py:MyEvaluator"
  task_name: "my_task"
```

The default worker:
1. Passes data directly to agent as query
2. Executes agent with NexAU
3. Processes trace to extract trajectories
4. Evaluates output using your evaluator
5. Generates training tokens and loss masks
6. Creates complete trajectory for algorithm processing

### Custom Query Formatting

Override `format_task_query` to customize how data is presented to the agent:

```python
# recipe/my_task/agent_workspace/my_worker.py

from nexrl.rollout_worker import BaseNexAURolloutWorker
from typing import Any

class MyTaskWorker(BaseNexAURolloutWorker):
    """Custom worker with specialized query formatting."""

    def format_task_query(self, data_item: dict[str, Any]) -> str:
        """
        Convert raw data into agent-compatible query.

        Args:
            data_item: Dictionary from dataloader with task data

        Returns:
            Formatted query string for agent
        """
        # Example: Math problem formatting
        problem = data_item.get("problem", "")
        context = data_item.get("context", "")

        query = f"Problem: {problem}\n"
        if context:
            query += f"Context: {context}\n"
        query += "\nProvide your solution step by step."

        return query
```

**Configuration:**
```yaml
rollout_worker:
  type: "nexau"
  custom_rollout_worker_module_path: "recipe/my_task/agent_workspace/my_worker.py"
  custom_rollout_worker_class_name: "MyTaskWorker"
  # ... other settings
```

### Advanced: Custom Loss Masking

Control which tokens contribute to training loss:

```python
class MyTaskWorker(BaseNexAURolloutWorker):
    def get_train_loss_mask(self, trajectory_infos: list[dict]) -> list[bool]:
        """
        Define loss mask for training.

        Args:
            trajectory_infos: List of trajectory dicts from trace processing
                Each contains: prompt_messages, tools, response_message, etc.

        Returns:
            Boolean mask - True for tokens to train on, False to skip
        """
        # Example 1: Train on all response tokens (default)
        return [True] * len(trajectory_infos)

        # Example 2: Skip tool call tokens, only train on final answer
        mask = []
        for traj in trajectory_infos:
            response_msg = traj.get("response_message", {})
            # Skip if message contains tool calls
            has_tool_calls = bool(response_msg.get("tool_calls"))
            mask.append(not has_tool_calls)
        return mask

        # Example 3: Only train on tokens after certain keyword
        # (Requires more complex token-level analysis)
```

### Complete Custom Worker Example

For complex tasks requiring full control:

```python
from nexrl.rollout_worker import BaseNexAURolloutWorker, NexAUEvaluationTarget
from typing import Any
from nexau.archs.tracer.adapters import InMemoryTracer

class ComplexTaskWorker(BaseNexAURolloutWorker):
    """Fully customized worker with pre/post processing."""

    def __init__(self, config):
        super().__init__(config)
        # Initialize custom components
        self.preprocessor = self._init_preprocessor()
        self.postprocessor = self._init_postprocessor()

    def format_task_query(self, data_item: dict[str, Any]) -> str:
        """Format query with preprocessing."""
        # Preprocess data
        processed_data = self.preprocessor(data_item)

        # Build query
        query = self._build_query(processed_data)

        return query

    def run_agent(self, task: dict[str, Any]) -> str | None:
        """
        Override full execution pipeline if needed.

        Use this when you need to:
        - Modify agent execution flow
        - Add custom pre/post processing
        - Implement multi-stage agent interactions
        """
        try:
            # 1. Format query
            query = self.format_task_query(task)

            # 2. Execute agent
            tracer = InMemoryTracer()
            final_answer = self.agent.run(query, client=self.client, tracer=tracer)

            # 3. Post-process output
            final_answer = self.postprocessor(final_answer)

            # 4. Process trace
            trace = tracer.get_trace()
            trajectories = []
            for child in trace.get("children", []):
                self.child_processor(child, trajectories)

            # 5. Evaluate
            evaluation_target = NexAUEvaluationTarget(
                final_answer=final_answer,
                observation=[{"trace": trace}]
            )
            eval_result = self.evaluator.evaluate(task, evaluation_target)

            # 6. Generate tokens and create trajectory
            tokens = self._generate_tokens(trajectories)
            loss_mask = self.get_train_loss_mask(trajectories)

            trajectory = {
                "tokens": tokens,
                "loss_mask": loss_mask,
                "reward": eval_result.reward,
                "final_answer": final_answer,
                "metrics": eval_result.metrics,
                **task
            }

            # 7. Submit to pool
            return self._put_trajectory(trajectory)

        except Exception as e:
            self.logger.error(f"Error in run_agent: {e}")
            return None

    def _init_preprocessor(self):
        # Your preprocessing logic
        return lambda x: x

    def _init_postprocessor(self):
        # Your postprocessing logic
        return lambda x: x

    def _build_query(self, data):
        # Your query building logic
        return str(data)
```

---

## Implementing Evaluators

Evaluators compute rewards and metrics from agent outputs.

### Basic Evaluator Structure

```python
# recipe/my_task/agent_workspace/evaluator.py

from typing import Any
from nexrl.rollout_worker import (
    Evaluator,
    BaseEvaluationTarget,
    EvaluationRunResult
)

class MyEvaluator(Evaluator):
    """Evaluator for my task."""

    def __init__(self):
        """Initialize evaluator with any needed resources."""
        super().__init__()
        # Load external resources if needed
        # self.reference_model = load_model()

    def evaluate(
        self,
        data: dict[str, Any],
        evaluation_target: BaseEvaluationTarget
    ) -> EvaluationRunResult:
        """
        Evaluate agent output.

        Args:
            data: Original task data (may contain ground truth)
            evaluation_target: Agent output
                - final_answer: Agent's final output string
                - observation: Complete execution trace (for NexAU agents)

        Returns:
            EvaluationRunResult with:
                - reward: Primary RL signal (float)
                - ground_truth: Reference answer (string)
                - metrics: Additional metrics (dict[str, float])
                - extra_info: Any other info (dict[str, Any])
        """
        # Extract agent output
        agent_output = evaluation_target.final_answer

        # Get ground truth
        ground_truth = data.get("answer", "")

        # Compute reward (main training signal)
        reward = self._compute_reward(agent_output, ground_truth)

        # Compute additional metrics (must be floats)
        metrics = self._compute_metrics(agent_output, ground_truth, data)

        # Store extra information (any type)
        extra_info = {
            "agent_output": agent_output,
            "data_id": data.get("id", ""),
        }

        return EvaluationRunResult(
            reward=reward,
            ground_truth=str(ground_truth),
            metrics=metrics,
            extra_info=extra_info
        )

    def _compute_reward(self, output: str, ground_truth: str) -> float:
        """Compute primary reward signal."""
        # Exact match
        return 1.0 if output.strip() == ground_truth.strip() else 0.0

    def _compute_metrics(
        self,
        output: str,
        ground_truth: str,
        data: dict
    ) -> dict[str, float]:
        """Compute additional metrics for logging."""
        return {
            "exact_match": float(output.strip() == ground_truth.strip()),
            "output_length": float(len(output)),
        }
```

### Common Evaluator Patterns

#### Pattern 1: Answer Extraction with Regex

```python
import re

class ExtractorEvaluator(Evaluator):
    """Evaluator that extracts structured answers."""

    def extract_answer(self, response: str) -> str | None:
        """Extract answer from formatted response."""
        # Example: Extract content between tags
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def evaluate(self, data, evaluation_target):
        # Extract structured answer
        extracted = self.extract_answer(evaluation_target.final_answer)

        if extracted is None:
            # No valid answer found - penalize
            return EvaluationRunResult(
                reward=0.0,
                metrics={"extraction_failed": 1.0}
            )

        # Compare extracted answer
        ground_truth = data.get("answer", "")
        reward = float(extracted == ground_truth)

        return EvaluationRunResult(
            reward=reward,
            ground_truth=ground_truth,
            metrics={
                "extraction_success": 1.0,
                "exact_match": reward,
            }
        )
```

#### Pattern 2: Multi-Criteria Evaluation

```python
class MultiCriteriaEvaluator(Evaluator):
    """Evaluator with multiple weighted criteria."""

    def __init__(self):
        super().__init__()
        # Define criterion weights
        self.weights = {
            "correctness": 0.5,
            "completeness": 0.3,
            "clarity": 0.2,
        }

    def evaluate(self, data, evaluation_target):
        output = evaluation_target.final_answer
        ground_truth = data.get("answer", "")

        # Evaluate each criterion
        scores = {
            "correctness": self._eval_correctness(output, ground_truth),
            "completeness": self._eval_completeness(output, data),
            "clarity": self._eval_clarity(output),
        }

        # Compute weighted reward
        reward = sum(
            scores[k] * self.weights[k]
            for k in self.weights
        )

        return EvaluationRunResult(
            reward=reward,
            ground_truth=ground_truth,
            metrics=scores  # Log individual scores
        )

    def _eval_correctness(self, output: str, ground_truth: str) -> float:
        """Evaluate correctness (0-1)."""
        return float(output.strip() == ground_truth.strip())

    def _eval_completeness(self, output: str, data: dict) -> float:
        """Evaluate completeness (0-1)."""
        required_points = data.get("required_points", [])
        covered = sum(
            1 for point in required_points
            if point.lower() in output.lower()
        )
        return covered / max(len(required_points), 1)

    def _eval_clarity(self, output: str) -> float:
        """Evaluate clarity (0-1)."""
        # Simple heuristic - can be more sophisticated
        word_count = len(output.split())
        # Penalize very short or very long outputs
        if word_count < 10:
            return 0.3
        elif word_count > 500:
            return 0.6
        else:
            return 1.0
```

#### Pattern 3: LLM-as-Judge Evaluation

```python
import openai

class LLMJudgeEvaluator(Evaluator):
    """Use an LLM to evaluate output quality."""

    def __init__(self):
        super().__init__()
        self.judge_client = openai.OpenAI()
        self.judge_model = "gpt-4"

    def evaluate(self, data, evaluation_target):
        output = evaluation_target.final_answer
        question = data.get("question", "")
        reference = data.get("reference_answer", "")

        # Create evaluation prompt
        prompt = f"""Evaluate the following answer on a scale of 0-1.

Question: {question}

Reference Answer: {reference}

Student Answer: {output}

Provide a score from 0 to 1, where:
- 0: Completely incorrect
- 0.5: Partially correct
- 1: Fully correct

Output only the numeric score."""

        # Query judge model
        response = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        # Parse score
        try:
            score_text = response.choices[0].message.content.strip()
            reward = float(score_text)
            reward = max(0.0, min(1.0, reward))  # Clamp to [0, 1]
        except (ValueError, AttributeError):
            reward = 0.0

        return EvaluationRunResult(
            reward=reward,
            ground_truth=reference,
            metrics={"llm_judge_score": reward}
        )
```

### Best Practices for Evaluators

1. **Reward Design**:
   - Primary reward should be in [0, 1] range
   - Use sparse rewards for binary tasks (0 or 1)
   - Use dense rewards for gradual improvement tasks
   - Consider reward shaping for complex tasks

2. **Metrics**:
   - All metrics must be scalar floats
   - Use descriptive metric names
   - Include intermediate metrics for debugging
   - Avoid expensive computations in metrics

3. **Ground Truth**:
   - Always populate the ground_truth field
   - Use string representation for consistency
   - Store original type in extra_info if needed

4. **Error Handling**:
   - Handle malformed outputs gracefully
   - Return 0 reward for invalid outputs
   - Log failures in metrics for analysis

---

## Training Backends & Trainers

NexRL provides a flexible trainer architecture that supports different training backends and algorithms. The key distinction is between self-hosted training (where you run the training infrastructure) and remote API training (where you use external training services like Tinker or Weaver).

> 💡 **TL;DR:** Use `self_hosted_grpo` for self-hosted deployments, or `remote_api_grpo` for Tinker/Weaver backends.

### Trainer Architecture Overview

NexRL trainers integrate algorithm-specific logic directly into the trainer class, eliminating the need for a separate algorithm processor layer. This design provides a clean, straightforward path for implementing custom training algorithms.

#### Self-Hosted Trainers (Self-Hosted Backend)

For deployments where you manage your own training infrastructure:

```
BaseTrainer
  ↓
SelfHostedTrainer (base class with abstract _prepare_batch)
  ↓
SelfHostedGrpoTrainer (implements GRPO algorithm)
```

**Pipeline:**
```
Trajectories → Process Trajectories → _prepare_batch (algorithm logic) → Training Service API
```

**Key characteristics:**
- **Full control:** Run your own training workers on your GPUs
- **Algorithm flexibility:** Implement custom algorithms by extending `SelfHostedTrainer`
- **Batch processing:** Works with `Batch` objects containing tensors
- **Integration:** Uses training service APIs (e.g., forward_backward, optim_step)

#### Remote API Trainers (Tinker/Weaver Backend)

For deployments using external training services:

```
BaseTrainer
  ↓
RemoteApiTrainer (base class with abstract _prepare_trajectories)
  ├── RemoteApiGrpoTrainer (GRPO algorithm)
  └── RemoteApiCrossEntropyTrainer (supervised learning)
```

**Pipeline:**
```
Trajectories → _prepare_trajectories (algorithm logic) → Service Datums → Tinker/Weaver API
```

**Key characteristics:**
- **Managed infrastructure:** Leverage external training services
- **Lightweight:** No local GPU workers needed
- **Trajectory processing:** Works directly with trajectory dictionaries
- **Service integration:** Uses Tinker/Weaver service holder pattern

### Available Trainers

| Trainer Type | Config Value | Backend | Algorithm | Use Case |
|--------------|--------------|---------|-----------|----------|
| SelfHostedTrainer | `self_hosted` | NexTrainer | Custom | Base for custom self-hosted algorithms |
| SelfHostedGrpoTrainer | `self_hosted_grpo` | NexTrainer | GRPO | Self-hosted GRPO training |
| RemoteApiGrpoTrainer | `remote_api_grpo` | Tinker/Weaver | GRPO | Cloud GRPO training |
| RemoteApiCrossEntropyTrainer | `remote_api_cross_entropy` | Tinker/Weaver | Cross-Entropy | Cloud supervised training |

### Configuration Examples

#### Remote API GRPO (Most Common)

For Tinker or Weaver backends:

```yaml
trainer:
  type: "remote_api_grpo"
  total_train_steps: 1000
  max_prompt_length: 15000
  max_response_length: 13000

service:
  train_service:
    backend: "tinker"  # or "weaver"
    config:
      loss_fn: "importance_sampling"
      learning_rate: 2e-6
      beta1: 0.9
      beta2: 0.95
      eps: 1e-8

  tinker_service:  # or weaver_service
    lora_rank: 32
    api_key: "your-api-key"
```

**Features:**
- GRPO advantage computation built into trainer
- Automatic trajectory grouping by `run_id`
- No custom code needed

#### Self-Hosted GRPO

For self-hosted NexTrainer backend:

```yaml
trainer:
  type: "self_hosted_grpo"
  total_train_steps: 1000
  max_prompt_length: 15000
  max_response_length: 13000

algorithm:
  type: "grpo"
  batch_size: 32
  do_old_log_prob_compute: true
  use_kl_in_reward: false

  # Optional: KL penalty configuration
  critic:
    kl_ctrl:
      type: "adaptive"  # or "fixed"
      kl_reward_coef: 0.1
      target_kl: 6.0
      horizon: 10000

  inference_service:
    model: "Qwen/Qwen2.5-7B-Instruct"
    tokenizer: "Qwen/Qwen2.5-7B-Instruct"

train_service:
  backend: "nextrainer"
  url: "http://train-service:8000"
  world_size: 4
```

**Features:**
- Full control over GRPO implementation
- Integrated advantage computation, KL penalty, and metrics logging
- Supports adaptive and fixed KL controllers

### Implementing Custom Trainers

#### Extending SelfHostedTrainer

Create a custom algorithm by extending `SelfHostedTrainer` and implementing `_prepare_batch`:

```python
# my_custom_trainer.py
from nexrl.trainer import SelfHostedTrainer
from nexrl.nexrl_types import Batch

class MyCustomTrainer(SelfHostedTrainer):
    """Custom trainer with my algorithm."""

    def __init__(self, config):
        super().__init__(config)
        # Initialize algorithm-specific components
        self._my_param = config.algorithm.get("my_param", 1.0)

    def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
        """
        Implement custom batch preparation logic.

        Args:
            batch: Batch from trajectory pool

        Returns:
            Tuple of (prepared_batch, metrics_dict)
        """
        metrics = {}

        # 1. Your preprocessing
        # ... remove padding, compute features ...

        # 2. Compute your algorithm-specific values
        # ... advantages, value estimates, etc ...
        batch.values["advantages"] = compute_my_advantages(batch)

        # 3. Log metrics
        metrics["my_algorithm/mean_advantage"] = batch.values["advantages"].mean().item()

        # 4. Return prepared batch and metrics
        return batch, metrics
```

**Register in controller:**
```python
# In nexrl/controller.py MODULE_REGISTRY
NexRLRole.TRAINER: {
    "my_custom": MyCustomTrainer,
    # ...
}
```

**Use in configuration:**
```yaml
trainer:
  type: "my_custom"

algorithm:
  my_param: 2.0
```

#### Extending RemoteApiTrainer

Create a custom remote trainer by extending `RemoteApiTrainer` and implementing `_prepare_trajectories`:

```python
# my_remote_trainer.py
from nexrl.trainer import RemoteApiTrainer
from nexrl.nexrl_types import Trajectory

class MyRemoteTrainer(RemoteApiTrainer):
    """Custom remote API trainer."""

    def _prepare_trajectories(
        self,
        trajectories: list[Trajectory],
        metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """
        Prepare trajectories for remote API.

        Args:
            trajectories: List of trajectories from rollout
            metrics: Dictionary to populate with metrics

        Returns:
            Processed trajectories
        """
        # Your trajectory processing logic
        processed = []
        for traj in trajectories:
            # Compute custom features
            traj["custom_feature"] = compute_feature(traj)
            processed.append(traj)

        # Log metrics
        metrics["my_metric"] = compute_my_metric(processed)

        return processed
```

### GRPO Implementation Details

#### RemoteApiGrpoTrainer

For Tinker/Weaver backends, GRPO logic is implemented in `RemoteApiGrpoTrainer._prepare_trajectories`:

```python
def _prepare_trajectories(self, trajectories, metrics):
    """Prepare trajectories with GRPO advantage computation."""
    from ..algorithm.core_algos import compute_grpo_advantage_for_trajectories
    from ..utils.logging_utils import log_grpo_metrics

    # Compute GRPO advantages (groups by run_id)
    trajectories = compute_grpo_advantage_for_trajectories(
        trajectories, logger=logger, use_run_ids=True
    )

    # Log GRPO statistics
    log_grpo_metrics(trajectories, metrics)

    return trajectories
```

**What it does:**
1. Groups trajectories by `run_id` (same prompt)
2. Computes group-relative advantages (normalized by std within group)
3. Logs GRPO statistics (mean, std per group)

#### SelfHostedGrpoTrainer

For NexTrainer backend, GRPO logic is implemented in `SelfHostedGrpoTrainer._prepare_batch`:

```python
def _prepare_batch(self, batch, metrics):
    """Prepare batch using GRPO algorithm."""

    # 1. Log rollout metrics
    self._log_rollout_metrics(batch)

    # 2. Remove redundant padding
    batch = Batch.remove_redundant_left_padding(...)
    batch = Batch.remove_redundant_right_padding(...)

    # 3. Recompute old log probabilities
    old_log_probs = self._compute_old_log_probs(batch)
    batch.values["old_log_probs"] = old_log_probs

    # 4. Compute token-level rewards
    reward_tensor = self._reward_fn(batch)
    batch.values["token_level_scores"] = reward_tensor

    # 5. Apply KL penalty (optional)
    if self._use_kl_in_reward:
        batch, kl_metrics = self._apply_kl_penalty(batch, ...)
        metrics.update(kl_metrics)

    # 6. Compute GRPO advantages
    batch = self._compute_advantage(batch)

    # 7. Compute and log metrics
    metrics.update(self._compute_data_metrics(batch))

    return batch, metrics
```

**What it does:**
1. Preprocesses batch (padding removal, metric logging)
2. Recomputes old log probabilities from current policy
3. Converts scalar rewards to token-level rewards
4. Optionally applies KL penalty to rewards
5. Computes group-relative advantages (GRPO)
6. Computes comprehensive metrics for monitoring

### Best Practices

1. **Choosing a Trainer:**
   - Use `remote_api_grpo` for Tinker/Weaver (easiest)
   - Use `self_hosted_grpo` for self-hosted NexTrainer
   - Create custom trainer only for new algorithms

2. **Implementing Custom Trainers:**
   - Start from existing GRPO implementation
   - Override only `_prepare_batch` or `_prepare_trajectories`
   - Keep algorithm logic in the prepare method
   - Log all important metrics for monitoring
   - Document your algorithm clearly

3. **Debugging:**
   - Log advantage/reward statistics (mean, std, min, max)
   - Monitor for NaN or inf values
   - Check trajectory grouping is correct
   - Validate reward shapes and masks

4. **Configuration:**
   - Make algorithm parameters configurable
   - Provide sensible defaults
   - Document parameter effects
   - Version your configurations

---

## Configuration Guide

### Full Configuration Example

```yaml
# recipe/my_task/my_task.yaml

# Hydra configuration
hydra:
  searchpath:
    - file://recipe/

# Import base configuration
defaults:
  - self_hosted_nexau_common
  - _self_

#==========================================
# PROJECT CONFIGURATION
#==========================================

project_name: "NexRL-MyTask"
experiment_name: "my-task-v1"

#==========================================
# ENVIRONMENT
#==========================================

environment:
  setup_script: "recipe/my_task/my_task.env.sh"

#==========================================
# DATA LOADER
#==========================================

data:
  type: "torch"
  data_files:
    - "/path/to/train.parquet"
  shuffle: true
  seed: 42
  max_prompt_length: ${rollout_worker.max_prompt_length}

  # Optional: Custom dataloader parameters
  batch_size: 1
  num_workers: 4

#==========================================
# ROLLOUT WORKER
#==========================================

rollout_worker:
  type: "nexau"

  # Custom worker (optional)
  custom_rollout_worker_module_path: "recipe/my_task/agent_workspace/my_worker.py"
  custom_rollout_worker_class_name: "MyTaskWorker"

  # NexAU configuration
  nexau_agent_config_path: "recipe/my_task/agent_workspace/agent_config.yaml"
  evaluator_module_path: "recipe/my_task/agent_workspace/evaluator.py:MyEvaluator"
  nexau_agent_workspace: "recipe/my_task/agent_workspace"
  task_name: "my_task"

  # Model configuration
  tokenizer: "Qwen/Qwen2.5-7B-Instruct"
  max_prompt_length: 16384

  # Worker scaling
  num_workers: 8

#==========================================
# TRAJECTORY POOL
#==========================================

trajectory_pool:
  type: "default"
  batch_size: 32

  # Grouping configuration (optional)
  group_size: 4  # Group 4 trajectories per query
  key_list: ["query"]  # Group by query field

  # Batch readiness
  check_batch_ready_function: "batch_size"

#==========================================
# ALGORITHM (Self-Hosted GRPO only)
#==========================================

# NOTE: Algorithm section is only used with 'self_hosted_grpo' trainer.
# For 'remote_api_grpo' (Tinker/Weaver), algorithm logic is built into the trainer
# and configuration is done via service.train_service.config section.

algorithm:
  type: "grpo"
  batch_size: ${trajectory_pool.batch_size}

  # GRPO-specific parameters
  do_old_log_prob_compute: true
  use_kl_in_reward: false

  # KL penalty configuration (optional)
  critic:
    kl_ctrl:
      type: "adaptive"  # or "fixed"
      kl_reward_coef: 0.1
      target_kl: 6.0
      horizon: 10000

  # Reference model for computing old log probabilities
  inference_service:
    model: "Qwen/Qwen2.5-7B-Instruct"
    tokenizer: "Qwen/Qwen2.5-7B-Instruct"

#==========================================
# TRAINER
#==========================================

trainer:
  type: "self_hosted_grpo"  # Options: "self_hosted_grpo", "remote_api_grpo", "remote_api_cross_entropy"
  total_train_steps: 10000

  # For self-hosted mode: specify checkpoint paths
  checkpoint_path: "/path/to/checkpoints/${experiment_name}"
  sync_weight_path: "/path/to/sync_weights/${experiment_name}"
  save_freq: 100
  remove_previous_ckpt: false

  # Note: For remote_api trainers (Tinker/Weaver), checkpointing is handled by the service

#==========================================
# WEIGHT SYNCHRONIZATION
#==========================================

weight:
  type: "default"
  sync_mode: "sync"  # or "fully-async", "batch-async"
  staleness_threshold: 2

#==========================================
# VALIDATION
#==========================================

validate:
  validate_before_train: true

  data:
    type: "torch"
    data_files:
      - "/path/to/test.parquet"
    shuffle: false

  eval:
    type: "default"
    compute_interval: 500  # Validate every N steps

#==========================================
# SERVICES
#==========================================

service:
  # Training service - Choose ONE backend
  train_service:
    # Option 1: NexTrainer (SelfHostedGrpoTrainer)
    backend: "nextrainer"
    url: "http://train-service:8000"
    model_tag: "policy"
    world_size: 4  # Number of GPUs

    # Option 2: Tinker (RemoteApiGrpoTrainer)
    # backend: "tinker"
    # url: "http://tinker-api:8000"
    # config:
    #   loss_fn: "importance_sampling"
    #   learning_rate: 2e-6
    #   beta1: 0.9
    #   beta2: 0.95
    #   eps: 1e-8

    # Option 3: Weaver (RemoteApiGrpoTrainer)
    # backend: "weaver"
    # url: "http://weaver-api:8000"
    # config:
    #   loss_fn: "importance_sampling"
    #   learning_rate: 2e-6
    #   beta1: 0.9
    #   beta2: 0.95
    #   eps: 1e-8

  # Inference service
  inference_service:
    backend: "vllm"
    url: "http://inference-service:8001"
    model_tag: "policy"
    api_key: "dummy"
    max_retries: 3
    freeze_for_weight_sync: true

#==========================================
# LOGGING
#==========================================

logger:
  backend: "wandb"
  wandb:
    project: ${project_name}
    name: ${experiment_name}
    tags: ["my_task", "grpo"]

#==========================================
# RUNTIME MONITORING
#==========================================

runtime_monitor:
  exception_handling:
    enabled: true
    check_interval: 10
    policy: "stop_on_error"

  health_check:
    enabled: true
    check_interval: 30
    timeout: 5.0

#==========================================
# RESOURCE CONFIGURATION
#==========================================

resource:
  inference:
    served_model_name: "my-task-model-v1"
    gpu_count: 2
    cpu_count: 8
    memory_gb: 64

  training:
    gpu_count: 4
    cpu_count: 16
    memory_gb: 128
```

### Configuration Inheritance

Use Hydra's composition to reuse common settings:

```yaml
# recipe/common/base_nexau.yaml
# Common settings for all NexAU tasks

rollout_worker:
  tokenizer: "Qwen/Qwen2.5-7B-Instruct"
  max_prompt_length: 16384

trainer:
  type: "self_hosted_grpo"
  save_freq: 100

algorithm:
  type: "grpo"
```

```yaml
# recipe/my_task/my_task.yaml
# Task-specific config

defaults:
  - common/base_nexau
  - _self_

# Override specific values
rollout_worker:
  task_name: "my_task"
  # ... task-specific settings
```

---

## Best Practices

### 1. Recipe Organization

**DO**:
- Keep all task files in one recipe directory
- Use descriptive naming (task_name, not "test1")
- Version your recipes (my_task_v1, my_task_v2)
- Document configuration changes in comments

**DON'T**:
- Mix multiple tasks in one recipe
- Use absolute paths in configs (use ${env.VAR} or relative paths)
- Hardcode API keys or secrets

### 2. Rollout Worker Design

**DO**:
- Start with default BaseNexAURolloutWorker
- Only override what you need
- Use clear, task-specific query formatting
- Handle edge cases (empty outputs, errors)
- Log important events

**DON'T**:
- Override run_agent unless necessary
- Perform expensive computation in format_task_query
- Ignore exceptions silently

### 3. Evaluator Design

**DO**:
- Make evaluation logic clear and documented
- Handle malformed outputs gracefully
- Return meaningful metrics for debugging
- Normalize rewards to [0, 1] range
- Consider partial credit for partially correct answers

**DON'T**:
- Use non-deterministic evaluation (adds noise)
- Perform expensive operations (slows rollout)
- Return non-scalar metrics
- Crash on unexpected input

### 4. Trainer Selection and Configuration

**DO**:
- Use `remote_api_grpo` for Tinker/Weaver (managed services)
- Use `self_hosted_grpo` for self-hosted deployments
- Start with existing GRPO trainers before creating custom ones
- Document your custom trainer logic clearly
- Validate computed advantages/rewards (check for NaN, inf)
- Log algorithm statistics for monitoring
- Make hyperparameters configurable via config files

**DON'T**:
- Implement custom trainers from scratch without understanding the base classes
- Ignore numerical stability in advantage computation
- Hard-code algorithm hyperparameters
- Mix trainer types and backends incorrectly (e.g., self_hosted_grpo with Tinker)

### 5. Configuration Management

**DO**:
- Use Hydra composition for common settings
- Document all configuration options
- Use environment variables for paths
- Provide sensible defaults
- Version your configurations

**DON'T**:
- Duplicate configuration across recipes
- Use absolute paths
- Commit secrets to version control

---

## Common Patterns

### Pattern 1: Multi-Turn Conversation Task

```python
class MultiTurnWorker(BaseNexAURolloutWorker):
    """Worker for multi-turn conversation tasks."""

    def format_task_query(self, data_item: dict[str, Any]) -> str:
        """Format conversation history into query."""
        conversation = data_item.get("conversation", [])

        # Format each turn
        formatted_turns = []
        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            formatted_turns.append(f"{role}: {content}")

        # Add current query
        current_query = data_item.get("query", "")
        formatted_turns.append(f"user: {current_query}")
        formatted_turns.append("assistant:")

        return "\n".join(formatted_turns)
```

### Pattern 2: Code Generation Task

```python
class CodeGenEvaluator(Evaluator):
    """Evaluator for code generation with test execution."""

    def evaluate(self, data, evaluation_target):
        generated_code = self.extract_code(evaluation_target.final_answer)
        test_cases = data.get("test_cases", [])

        # Execute test cases
        passed = 0
        failed = 0
        for test in test_cases:
            if self.run_test(generated_code, test):
                passed += 1
            else:
                failed += 1

        # Reward based on pass rate
        reward = passed / max(len(test_cases), 1)

        return EvaluationRunResult(
            reward=reward,
            metrics={
                "test_pass_rate": reward,
                "tests_passed": float(passed),
                "tests_failed": float(failed),
            }
        )

    def extract_code(self, response: str) -> str:
        """Extract code block from response."""
        match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        return match.group(1) if match else response

    def run_test(self, code: str, test: dict) -> bool:
        """Run a single test case."""
        try:
            # Execute code safely (use sandbox in production)
            exec_globals = {}
            exec(code, exec_globals)

            # Run test
            test_func = exec_globals.get(test["function_name"])
            result = test_func(*test["inputs"])

            return result == test["expected_output"]
        except Exception:
            return False
```

### Pattern 3: Reasoning Task with Chain of Thought

```python
class ReasoningWorker(BaseNexAURolloutWorker):
    """Worker that encourages step-by-step reasoning."""

    def format_task_query(self, data_item: dict[str, Any]) -> str:
        """Format query to encourage reasoning."""
        question = data_item.get("question", "")

        query = f"""Question: {question}

Please solve this step by step:
1. First, identify what is being asked
2. Then, break down the problem
3. Show your reasoning at each step
4. Finally, provide your answer

Format your response as:
<reasoning>
Your step-by-step thinking here
</reasoning>
<answer>
Your final answer here
</answer>"""

        return query

class ReasoningEvaluator(Evaluator):
    """Evaluator that rewards both correctness and reasoning."""

    def evaluate(self, data, evaluation_target):
        response = evaluation_target.final_answer

        # Extract components
        reasoning = self.extract_reasoning(response)
        answer = self.extract_answer(response)

        # Evaluate correctness
        ground_truth = data.get("answer", "")
        correctness = float(answer == ground_truth)

        # Evaluate reasoning quality (simple heuristic)
        reasoning_quality = self.eval_reasoning_quality(reasoning)

        # Combined reward
        reward = 0.7 * correctness + 0.3 * reasoning_quality

        return EvaluationRunResult(
            reward=reward,
            ground_truth=ground_truth,
            metrics={
                "correctness": correctness,
                "reasoning_quality": reasoning_quality,
                "has_reasoning": float(reasoning is not None),
            }
        )

    def extract_reasoning(self, response: str) -> str | None:
        match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        return match.group(1).strip() if match else None

    def extract_answer(self, response: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        return match.group(1).strip() if match else ""

    def eval_reasoning_quality(self, reasoning: str | None) -> float:
        """Simple reasoning quality heuristic."""
        if reasoning is None:
            return 0.0

        # Check for step markers
        has_steps = any(marker in reasoning for marker in ["1.", "Step", "First"])

        # Check length (reasonable reasoning should be substantial)
        reasonable_length = 50 < len(reasoning) < 1000

        return 1.0 if (has_steps and reasonable_length) else 0.5
```

---

## Debugging Tips

### Enable Debug Logging

```python
# In your worker or evaluator
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.debug(f"Processing task: {task}")
logger.debug(f"Generated query: {query}")
logger.debug(f"Agent output: {output}")
```

### Test Components Independently

```python
# Test evaluator standalone
from recipe.my_task.agent_workspace.evaluator import MyEvaluator
from nexrl.rollout_worker import BaseEvaluationTarget

evaluator = MyEvaluator()
target = BaseEvaluationTarget(final_answer="Test output")
data = {"answer": "Expected"}
result = evaluator.evaluate(data, target)
print(f"Reward: {result.reward}, Metrics: {result.metrics}")
```

### Validate Configuration

```bash
# Check configuration is valid
python -m nexrl.validator recipe=my_task/my_task
```

### Monitor Training

```python
# Add custom logging in workers
self._activity_tracker.experiment_logger_post(
    backend="wandb",
    data={
        "custom_metric": value,
        "agent_calls": count,
    },
    step=current_step
)
```

---

## Additional Resources

- **Developer Guide**: Detailed API reference and architecture overview
- **Example Recipes**: See `recipe/nexau_news_qwen3_8b` for complete example
- **Core Algorithms**: See `nexrl/algorithm/core_algos.py` for GRPO implementation
- **NexAU Documentation**: For agent framework details

---

## Quick Reference

### File Checklist for New Task

- [ ] `recipe/my_task/my_task.yaml` - Main configuration
- [ ] `recipe/my_task/agent_workspace/agent_config.yaml` - Agent config
- [ ] `recipe/my_task/agent_workspace/evaluator.py` - Evaluator implementation
- [ ] `recipe/my_task/agent_workspace/my_worker.py` - Custom worker (optional)
- [ ] `recipe/my_task/my_task.env.sh` - Environment setup (optional)
- [ ] Data files referenced in configuration

### Common Commands

```bash
# Run training (CLI - recommended)
nexrl -m self-hosted -c recipe/my_task/my_task.yaml --run-nexrl

# Run with custom tag
nexrl -m self-hosted -c recipe/my_task/my_task.yaml -r -t my-experiment

# Training-service mode with Tinker/Weaver
nexrl -m training-service -c recipe/my_task/tinker.yaml --run-nexrl

# Use existing inference service (skip launching)
nexrl -m self-hosted -c recipe/my_task/my_task.yaml -r --inference-url my-service:8000

# Python module invocation (alternative)
python -m nexrl.main recipe=my_task/my_task

# Monitor job logs
kubectl logs -f -l app=JOB_NAME-driver -n nexrl

# List all resources
kubectl get all -n nexrl
```

---

**Happy Training! 🚀**
