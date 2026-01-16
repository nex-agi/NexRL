# NexAU Rollout Worker

The `BaseNexAURolloutWorker` provides integration with the NexAU agent framework, enabling complex multi-step agent tasks with tool calling, trace processing, and comprehensive evaluation capabilities.

## Overview

**Location:** `nexrl/rollout_worker/base_nexau_rollout_worker.py`

```python
from nexrl.rollout_worker import BaseNexAURolloutWorker

class BaseNexAURolloutWorker(BaseRolloutWorker):
    """Base class for NexAU-based rollout workers."""
```

**Key features:**
- NexAU agent framework integration
- Automatic trace processing and trajectory extraction
- Integrated evaluation system
- Token/loss mask generation
- Workspace management for local imports
- Trace prefix merging for multi-turn interactions

## Architecture

```
Task → Format Query → NexAU Agent → Trace Processor → Evaluator → Trajectories
                         ↓
                   InMemoryTracer
                         ↓
                  LLM Call Tree
```

## Constructor

```python
def __init__(self, config: DictConfig):
    """
    Initialize NexAU worker.

    Args:
        config: Configuration with keys:
            - nexau_agent_config_path: Path to agent config YAML
            - evaluator_module_path: Path to evaluator (format: "path.py:ClassName")
            - nexau_agent_workspace: Workspace directory for imports
            - tokenizer: Path to tokenizer
            - trace_path: Directory to save traces (optional)
            - save_trace: Whether to save traces (default: False)
            - temperature: Sampling temperature
            - enable_trace_prefix_merge: Merge trajectory prefixes (default: True)
            - _config_file_path: Recipe config file path (auto-set by Hydra)
    """
```

**Initialization process:**
1. Resolves paths relative to recipe config file
2. Loads tokenizer for token processing
3. Sets up trace directory if enabled
4. Adds workspace to `sys.path` for local imports
5. Dynamically loads evaluator class

## Configuration

### Minimal Configuration

```yaml
rollout_worker:
  type: "nexau"
  num_workers: 4

  # Required: Agent configuration
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"

  # Required: Evaluator
  evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"

  # Required: Tokenizer
  tokenizer: "meta-llama/Llama-3.1-8B-Instruct"

  # Optional: Workspace for imports
  nexau_agent_workspace: "agent_workspace"

  # Inference service
  inference_service:
    backend: "vllm"
    url: "http://localhost:8000/v1"
    model_tag: "default"
    api_key: "EMPTY"

  # Worker settings
  need_llm_inference: true
  temperature: 0.7
  max_tokens: 4096
```

### Full Configuration

```yaml
rollout_worker:
  type: "nexau"
  num_workers: 4

  # NexAU configuration
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
  evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"
  nexau_agent_workspace: "agent_workspace"
  task_name: "my_task"

  # Tokenizer
  tokenizer: "meta-llama/Llama-3.1-8B-Instruct"

  # Trace saving
  save_trace: true
  trace_path: "outputs/traces"

  # Trajectory processing
  enable_trace_prefix_merge: true

  # Inference service
  inference_service:
    backend: "vllm"
    url: "http://localhost:8000/v1"
    model_tag: "default"
    api_key: "EMPTY"
    max_retries: 3
    freeze_for_weight_sync: true

  # Generation parameters
  need_llm_inference: true
  temperature: 0.7
  max_tokens: 4096
  top_p: 0.9
```

## Core Methods

### load_agent_from_config()

Loads NexAU agent from configuration file:

```python
def load_agent_from_config(self, custom_llm_client_provider=None):
    """
    Load NexAU agent from config.

    Args:
        custom_llm_client_provider: Function that returns custom LLM client

    Returns:
        Tuple of (agent, client_provider_func)
    """
```

**Usage:**

```python
# Load agent with custom LLM client
agent, client_provider_func = self.load_agent_from_config(
    custom_llm_client_provider=lambda: self._inference_client
)

# Run agent
response = agent.run(query, custom_llm_client_provider=client_provider_func)
```

### run_agent()

Executes agent and evaluates result:

```python
def run_agent(self, task: dict[str, Any]) -> tuple[Any, EvaluationRunResult]:
    """
    Run agent and evaluate result.

    Args:
        task: Task dictionary (must contain 'prompt', 'query', or 'question')

    Returns:
        Tuple of (agent_output, evaluation_result)
    """
```

**Default implementation:**
1. Extracts query from task (`prompt`, `query`, or `question` field)
2. Loads agent with custom LLM client
3. Runs agent with tracer enabled
4. Extracts traces from InMemoryTracer
5. Processes traces into trajectories
6. Evaluates agent output
7. Attaches reward to each trajectory

**Override this method** for custom query formatting:

```python
class MyWorker(BaseNexAURolloutWorker):
    def run_agent(self, task):
        # Custom query formatting
        query = self._format_custom_query(task)

        # Load and run agent
        agent, client_provider = self.load_agent_from_config(
            custom_llm_client_provider=lambda: self._inference_client
        )
        response = agent.run(query, custom_llm_client_provider=client_provider)

        # Extract traces
        traces = self._extract_traces(agent)
        trajectories = self.trace_processor(traces)

        # Create agent output
        agent_output = AgentOutput(
            final_answer=response,
            observation=agent.history,
            rl_params={"trajectory": trajectories}
        )

        # Evaluate
        eval_result = self.evaluator.evaluate(task, evaluation_target)

        # Attach reward
        for traj in trajectories:
            traj["reward"] = eval_result.reward
            traj["score"] = {"reward_score": eval_result.reward, **eval_result.metrics}

        return agent_output, eval_result
```

### rollout()

Main rollout execution (usually not overridden):

```python
def rollout(self, task: dict[str, Any]) -> str | None:
    """
    Execute rollout: run agent, process traces, create trajectories.

    Args:
        task: Task dictionary

    Returns:
        Result status from _put_trajectory
    """
```

**Process:**
1. Calls `run_agent(task)` to get agent output and evaluation
2. Saves trace to disk (if enabled)
3. Processes trajectories: extracts tokens, creates loss masks
4. Merges trajectory prefixes (if enabled)
5. Creates `Trajectory` objects
6. Submits trajectories to pool (returns last submission result)

## Trace Processing

### trace_processor()

Extracts trajectory information from NexAU traces:

```python
def trace_processor(self, traces: list[dict]) -> list[dict]:
    """
    Process traces and extract trajectory information.

    Args:
        traces: Raw traces from InMemoryTracer

    Returns:
        List of trajectory dictionaries containing:
            - prompt_messages: Input messages
            - tools: Available tools
            - response_message: LLM response
            - response_tokens: Token IDs
            - finish_reason: Completion status
            - nexrl_train: Training data (if available)
    """
```

**How it works:**
1. Recursively traverses trace tree
2. Finds all LLM call nodes
3. Extracts prompt messages, tools, and response
4. Extracts response tokens from `nexrl_train` or `logprobs`
5. Returns list of trajectory info dictionaries

### child_processor()

Processes individual trace nodes:

```python
def child_processor(self, child: dict, trajectories: list[dict]):
    """
    Process single trace child node.

    Args:
        child: Trace node
        trajectories: List to append trajectory info to
    """
```

**Extraction logic:**
- **With `nexrl_train`**: Uses pre-extracted tokens from inference service
- **Without `nexrl_train`**: Extracts from OpenAI-style logprobs
- Handles nested children recursively

### add_loss_mask()

Generates loss mask and combines tokens:

```python
def add_loss_mask(
    self,
    prompt_tokens: list[int],
    response_tokens: list[int],
    response_logprobs: list[float] | None = None,
) -> dict[str, list]:
    """
    Generate loss mask and combine prompt+response tokens.

    Returns:
        Dictionary with:
            - tokens: Combined token sequence
            - loss_mask: 0 for prompt, 1 for response
            - logprobs: Log probabilities
    """
```

**Default behavior:**
- All prompt tokens: `loss_mask=0`
- All response tokens: `loss_mask=1`

**Override `get_train_loss_mask()`** for custom masking:

```python
class MyWorker(BaseNexAURolloutWorker):
    def get_train_loss_mask(self, trajectory_infos: list[dict]) -> list[bool]:
        """
        Custom loss masking logic.

        Args:
            trajectory_infos: List of trajectory dicts

        Returns:
            Boolean mask for each token
        """
        # Example: Mask out tool calls
        mask = []
        for traj in trajectory_infos:
            response_msg = traj.get("response_message", {})
            if response_msg.get("tool_calls"):
                # Mask tool call tokens
                mask.extend([False] * len(traj["response_tokens"]))
            else:
                # Keep regular response tokens
                mask.extend([True] * len(traj["response_tokens"]))
        return mask
```

### trace_prefix_merge()

Merges trajectories with common prefixes:

```python
def trace_prefix_merge(self, trajectories: list[dict]) -> list[dict]:
    """
    Merge trajectories that share a common prefix.

    When a trajectory's prompt matches the full sequence of the
    previous trajectory, merge them by appending new tokens.

    Args:
        trajectories: List of trajectory dicts

    Returns:
        List of merged trajectories
    """
```

**Use case:** Multi-turn agent interactions where each turn builds on previous context

**Example:**
```
Turn 1: [prompt_tokens] + [response_1_tokens]
Turn 2: [prompt_tokens + response_1_tokens] + [response_2_tokens]

After merge:
[prompt_tokens] + [response_1_tokens] + [response_2_tokens]
```

**Control:** Set `enable_trace_prefix_merge: false` to disable

### save_trace()

Saves trace to disk for debugging:

```python
def save_trace(self, group_id: str, run_id: int, step: int, logs: dict):
    """
    Save trace to disk.

    Saves JSON file with naming: group_{group_id}_run_{run_id}_step_{step}-uuid-{uuid}.json
    """
```

**Enable:** Set `save_trace: true` and `trace_path: "outputs/traces"`

## Evaluation System

### Evaluator Base Class

```python
from nexrl.rollout_worker import Evaluator, BaseEvaluationTarget, EvaluationRunResult

class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        data: dict[str, Any],
        evaluation_target: BaseEvaluationTarget,
    ) -> EvaluationRunResult:
        """Evaluate agent output."""
```

### Evaluation Target

```python
@dataclass
class BaseEvaluationTarget:
    final_answer: str

@dataclass
class NexAUEvaluationTarget(BaseEvaluationTarget):
    final_answer: str
    observation: list[dict[str, Any]]  # Agent execution history
```

### Evaluation Result

```python
@dataclass
class EvaluationRunResult:
    reward: float = 0.0                          # Primary RL signal [0.0, 1.0]
    ground_truth: str = ""                       # Reference answer
    metrics: dict[str, float] = field(default_factory=dict)  # Must be scalar floats
    extra_info: dict[str, Any] = field(default_factory=dict)  # Can be any type
```

**Guidelines:**
- `reward`: Primary training signal, should be in [0.0, 1.0] range
- `metrics`: Additional metrics for logging (must be scalar floats for aggregation)
- `extra_info`: Debugging/analysis data (any type, not logged as metrics)

## Example: Custom Worker

Here's a complete example for a news classification task:

```python
from nexrl.rollout_worker import (
    BaseNexAURolloutWorker,
    EvaluationRunResult,
    NexAUEvaluationTarget
)
from nexau.archs.tracer.adapters import InMemoryTracer
from dataclasses import dataclass, field
from typing import Any
import logging

logger = logging.getLogger(__name__)

class NewsNexAURolloutWorker(BaseNexAURolloutWorker):
    """News-specific rollout worker with custom query formatting."""

    def _format_news_query(self, data_item: dict[str, Any]) -> str:
        """Format news data into query string."""
        company_name = data_item.get("company_name", "")
        news_title = "新闻标题: " + data_item.get("title", "")
        news_summary = "新闻摘要: " + data_item.get("summary", "")
        time_now = "现在时间：" + data_item.get("created_at", "")
        time_news = "新闻时间：" + data_item.get("date_str", "")
        goal_entity = "目标企业/人/组织名: " + company_name
        news_tags = "新闻标签: " + data_item.get("tags", "")
        news_source = "新闻来源: " + data_item.get("source", "")

        news_to_judge = (
            f"<需要判断的新闻>\n"
            f"    {time_now}\n"
            f"    {time_news}\n"
            f"    {goal_entity}\n"
            f"    {news_title}\n"
            f"    {news_summary}\n"
            f"    {news_tags}\n"
            f"    {news_source}\n"
            f"</需要判断的新闻>"
        )

        return news_to_judge

    def run_agent(self, task: dict[str, Any]) -> tuple[Any, EvaluationRunResult]:
        """Run agent with custom query formatting."""
        # Custom query formatting
        query = self._format_news_query(task)

        # Load agent
        agent, client_provider_func = self.load_agent_from_config(
            custom_llm_client_provider=lambda: self._inference_client
        )

        # Run agent
        response = agent.run(query, custom_llm_client_provider=client_provider_func)

        # Extract traces
        traces = []
        for tracer in agent.config.tracers:
            if isinstance(tracer, InMemoryTracer):
                traces = tracer.dump_traces()
                break

        # Process traces
        trajectories = self.trace_processor(traces)

        # Create agent output
        @dataclass
        class AgentOutput:
            final_answer: str
            observation: list
            rl_params: dict = field(default_factory=dict)

        agent_output = AgentOutput(
            final_answer=response,
            observation=agent.history,
            rl_params={"trajectory": trajectories}
        )

        # Evaluate
        evaluation_result = self.evaluator.evaluate(
            task,
            NexAUEvaluationTarget(
                final_answer=agent_output.final_answer,
                observation=agent_output.observation
            ),
        )

        # Attach reward to trajectories
        for traj in trajectories:
            traj["reward"] = evaluation_result.reward
            traj["score"] = {
                "reward_score": evaluation_result.reward,
                **evaluation_result.metrics,
            }

        return agent_output, evaluation_result
```

## Recipe Structure

Typical NexAU-based recipe layout:

```
recipe/
└── my_task/
    ├── my_task.yaml                    # Main recipe config
    ├── my_task.env.sh                  # Environment setup
    └── agent_workspace/                # Agent files
        ├── agent_config.yaml           # NexAU agent config
        ├── evaluator.py                # Task evaluator
        ├── my_worker.py                # Custom worker (optional)
        └── custom_tools.py             # Task-specific tools (optional)
```

### Recipe Configuration

```yaml
# my_task.yaml
defaults:
  - base_config

rollout_worker:
  type: "nexau"

  # Agent configuration
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
  evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"
  nexau_agent_workspace: "agent_workspace"

  # Optional: Custom worker
  custom_rollout_worker_module_path: "agent_workspace/my_worker.py"
  custom_rollout_worker_class_name: "MyWorker"

  # ... rest of config
```

### Agent Configuration

```yaml
# agent_workspace/agent_config.yaml
name: "MyAgent"
version: "1.0"

system_prompt: |
  You are a helpful assistant that solves tasks step by step.

tools:
  - name: "web_search"
    description: "Search the web for information"
    # ... tool definition

llm:
  model: "meta-llama/Llama-3.1-8B-Instruct"
  temperature: 0.7
  max_tokens: 4096

tracers:
  - type: "in_memory"  # Required for NexRL
```

## Advanced Features

### Custom Loss Masking

Override `get_train_loss_mask()` for fine-grained control:

```python
class MyWorker(BaseNexAURolloutWorker):
    def get_train_loss_mask(self, trajectory_infos: list[dict]) -> list[bool]:
        """
        Custom masking: only train on final answer, not tool calls.
        """
        mask = []
        for traj in trajectory_infos:
            response_msg = traj.get("response_message", {})

            # Check if this is a tool call
            if response_msg.get("tool_calls"):
                # Mask out tool call tokens
                num_tokens = len(traj.get("response_tokens", []))
                mask.extend([False] * num_tokens)
            else:
                # Keep final answer tokens
                num_tokens = len(traj.get("response_tokens", []))
                mask.extend([True] * num_tokens)

        return mask
```

### Workspace Management

The worker automatically adds workspace to `sys.path`:

```python
# In agent_workspace/custom_tools.py
def my_custom_tool():
    return "Result"

# In agent_config.yaml, can import directly
tools:
  - module: "custom_tools"
    function: "my_custom_tool"
```

### Trace Analysis

Save traces for debugging:

```yaml
rollout_worker:
  save_trace: true
  trace_path: "outputs/traces"
```

Traces include:
- Full agent execution history
- All LLM calls and responses
- Reward and metrics
- Extra evaluation info

## Performance Considerations

### Token Processing

Token extraction happens per trajectory:
- Use `enable_trace_prefix_merge: true` to reduce redundant tokens
- This is especially beneficial for multi-turn interactions

### Trace Saving

Trace saving can be I/O intensive:
- Disable in production: `save_trace: false`
- Enable for debugging specific issues

### Evaluator Performance

Evaluators run synchronously during rollout:
- Keep evaluation logic efficient
- Move expensive analysis to `extra_info` (not used for training)

## Troubleshooting

### Issue: Agent config not found

**Error:** `FileNotFoundError: Agent config not found`

**Solution:** Ensure path is relative to recipe config file location

### Issue: Evaluator class not found

**Error:** `AttributeError: Class 'MyEvaluator' not found`

**Solution:**
- Check evaluator_module_path format: `"path.py:ClassName"`
- Ensure class name matches exactly
- Verify file is in workspace

### Issue: No trajectories generated

**Symptoms:** Worker completes but no trajectories submitted

**Causes:**
- Agent returned no LLM calls
- Trace processing failed
- No `nexrl_train` data or logprobs available

**Solution:**
- Check agent execution
- Enable trace saving to inspect traces
- Verify inference service returns token IDs

### Issue: Token length mismatch

**Error:** `AssertionError: Length mismatch: tokens=X, loss_mask=Y`

**Cause:** Inconsistency in token extraction

**Solution:** Override `add_loss_mask()` with custom logic

## Next Steps

- [Custom Workers](./custom-workers.md) - Creating task-specific workers
- [Evaluators](./evaluators.md) - Implementing evaluation logic
- [Recipes](../10-recipes/recipe-structure.md) - Recipe organization

## Related Documentation

- [NexAU Documentation](https://github.com/Nex-AGI/NexAU) - Agent framework
- [Data Types](../02-core-architecture/data-types.md) - Trajectory structure
- [Configuration Reference](../12-configuration-reference/rollout-config.md) - Full config options
