# Rollout Worker Configuration

Configuration options for rollout workers.

## Rollout Worker Types

### SimpleRolloutWorker (`type: "simple"`)

Basic LLM completion worker.

```yaml
rollout_worker:
  type: "simple"
  num_workers: 32
  need_llm_inference: true
  temperature: 1.0
```

### AgentRolloutWorker (`type: "agent"`)

Agent-based worker with tool calling support.

```yaml
rollout_worker:
  type: "agent"
  num_workers: 128
  need_llm_inference: true
  temperature: 0.7
  agent_cls: "single_turn_math"

  agent:
    math:
      judge_mode: "rule"
```

### NexAURolloutWorker (`type: "nexau"`)

NexAU-based worker for complex agent tasks.

```yaml
rollout_worker:
  type: "nexau"
  num_workers: 128
  need_llm_inference: true
  temperature: 0.7

  # Custom worker (optional)
  custom_rollout_worker_module_path: "agent_workspace/my_worker.py"
  custom_rollout_worker_class_name: "MyWorker"

  # NexAU configuration
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
  evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"
  nexau_agent_workspace: "agent_workspace"
  task_name: "my_task"

  max_prompt_length: ${data.max_prompt_length}
  max_response_length: ${data.max_response_length}
```

## Configuration Options

### type

**Type:** `string`
**Options:** `"simple"`, `"agent"`, `"nexau"`, `"mock"`
**Default:** Required

Rollout worker implementation.

- `simple`: Basic LLM completion
- `agent`: Multi-turn agent with tools
- `nexau`: NexAU-based agent framework
- `mock`: Mock worker for testing

### num_workers

**Type:** `int`
**Default:** Required

Number of rollout worker instances.

**Guidelines:**
- CPU-bound: 4-16 workers
- I/O-bound (LLM API): 32-256 workers
- Ray mode: Scale across cluster nodes
- Consider data throughput needs

**Example:**
```yaml
# Small scale
num_workers: 8

# Medium scale
num_workers: 64

# Large scale
num_workers: 256
```

### need_llm_inference

**Type:** `bool`
**Default:** `true`

Whether worker needs LLM inference service.

**When `true`:**
- Worker uses LLMServiceClient
- Requires inference service configured
- Supports weight synchronization

**When `false`:**
- Worker doesn't call LLM
- Used for testing or non-LLM tasks

### temperature

**Type:** `float`
**Default:** `1.0`

Sampling temperature for LLM generation.

**Values:**
- `0.0`: Greedy (deterministic)
- `0.1-0.5`: Low variance
- `0.7-1.0`: Balanced (common)
- `>1.0`: High variance

**Use Cases:**
```yaml
# Deterministic (evaluation)
temperature: 0.0

# Low variance (precise tasks)
temperature: 0.3

# Balanced (general training)
temperature: 0.7

# Exploration (diverse responses)
temperature: 1.0
```

## NexAU-Specific Options

### custom_rollout_worker_module_path

**Type:** `string`
**Default:** None (optional)

Path to custom worker implementation.

**Format:** Relative to recipe directory or absolute path

**Example:**
```yaml
custom_rollout_worker_module_path: "agent_workspace/my_worker.py"
```

**Custom Worker:**
```python
from nexrl.rollout_worker import BaseNexAURolloutWorker

class MyWorker(BaseNexAURolloutWorker):
    def format_task_query(self, data_item):
        return f"Task: {data_item['query']}"
```

### custom_rollout_worker_class_name

**Type:** `string`
**Default:** None (required if custom_rollout_worker_module_path set)

Class name in custom worker module.

**Example:**
```yaml
custom_rollout_worker_class_name: "MyWorker"
```

### nexau_agent_config_path

**Type:** `string`
**Default:** Required (for type: "nexau")

Path to NexAU agent configuration YAML.

**Format:** Relative to recipe directory

**Example:**
```yaml
nexau_agent_config_path: "agent_workspace/agent_config.yaml"
```

**Agent Config Structure:**
```yaml
system_prompt: "You are a helpful assistant."
tools:
  - name: "WebSearch"
    config_path: "tools/WebSearch.yaml"
llm_config:
  model: "gpt-4"
  temperature: 0.7
```

### evaluator_module_path

**Type:** `string`
**Default:** Required (for type: "nexau")

Path to evaluator module and class.

**Format:** `"path/to/file.py:ClassName"`

**Example:**
```yaml
evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"
```

**Evaluator Implementation:**
```python
from nexrl.rollout_worker import Evaluator, EvaluationRunResult

class MyEvaluator(Evaluator):
    def evaluate(self, data, evaluation_target):
        reward = compute_reward(data, evaluation_target.final_answer)
        return EvaluationRunResult(reward=reward, ground_truth=data["answer"])
```

### nexau_agent_workspace

**Type:** `string`
**Default:** None (optional)

Workspace directory for agent modules.

**Purpose:**
- Added to `sys.path` for local imports
- Contains agent tools and utilities
- Enables self-contained recipes

**Example:**
```yaml
nexau_agent_workspace: "agent_workspace"
```

**Directory Structure:**
```
agent_workspace/
├── agent_config.yaml
├── evaluator.py
├── my_worker.py
├── tools/
│   ├── WebSearch.yaml
│   └── custom_tool.py
└── utils/
    └── helpers.py
```

### task_name

**Type:** `string`
**Default:** Required (for type: "nexau")

Task identifier for logging and tracking.

**Example:**
```yaml
task_name: "math_reasoning"
```

## Agent-Specific Options

### agent_cls

**Type:** `string`
**Default:** Required (for type: "agent")

Agent class name.

**Options:**
- `"single_turn_math"`: Math reasoning agent

**Example:**
```yaml
rollout_worker:
  type: "agent"
  agent_cls: "single_turn_math"

  agent:
    math:
      judge_mode: "rule"  # Agent-specific config
```

## Length Constraints

### max_prompt_length

**Type:** `int`
**Default:** Inherited from `data.max_prompt_length`

Maximum prompt length in tokens.

**Example:**
```yaml
max_prompt_length: ${data.max_prompt_length}
```

### max_response_length

**Type:** `int`
**Default:** Inherited from `data.max_response_length`

Maximum response length in tokens.

**Example:**
```yaml
max_response_length: ${data.max_response_length}
```

## Common Patterns

### Simple LLM Task

```yaml
rollout_worker:
  type: "simple"
  num_workers: 32
  need_llm_inference: true
  temperature: 0.7
```

### Math Reasoning

```yaml
rollout_worker:
  type: "agent"
  num_workers: 128
  need_llm_inference: true
  temperature: 1.0
  agent_cls: "single_turn_math"

  agent:
    math:
      judge_mode: "rule"
```

### Custom NexAU Task

```yaml
rollout_worker:
  type: "nexau"
  num_workers: 64
  temperature: 0.7

  custom_rollout_worker_module_path: "agent_workspace/custom_worker.py"
  custom_rollout_worker_class_name: "CustomWorker"

  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
  evaluator_module_path: "agent_workspace/evaluator.py:CustomEvaluator"
  nexau_agent_workspace: "agent_workspace"
  task_name: "custom_task"

  max_prompt_length: 4096
  max_response_length: 8192
```

### Testing Configuration

```yaml
rollout_worker:
  type: "mock"
  num_workers: 4
  need_llm_inference: false
```

## Worker Scaling Guidelines

### Development (Single Node)

```yaml
num_workers: 8  # Small scale for debugging
```

### Production (Small Cluster)

```yaml
# 4 nodes × 16 workers/node
num_workers: 64
```

### Production (Large Cluster)

```yaml
# 16 nodes × 16 workers/node
num_workers: 256
```

## Temperature Guidelines

### Task-Based

```yaml
# Exact matching (code, math)
temperature: 0.0

# Reasoning tasks
temperature: 0.7

# Creative tasks
temperature: 1.0

# Exploration training
temperature: 1.0
```

## Troubleshooting

### Workers Idle

**Symptom:** Low throughput, workers not processing

**Solutions:**
- Check data loader has items available
- Verify LLM service is accessible
- Review worker logs for errors
- Check trajectory pool not blocking

### High Error Rate

**Symptom:** Many failed trajectories

**Solutions:**
- Check LLM service status
- Verify evaluator implementation
- Review prompt format
- Check max_response_length sufficient

### Custom Worker Not Found

**Symptom:** `ModuleNotFoundError` or `AttributeError`

**Solutions:**
- Verify `custom_rollout_worker_module_path` is correct
- Check `custom_rollout_worker_class_name` matches class
- Ensure file is in correct location
- Check `nexau_agent_workspace` if using local imports

## Related Documentation

- [Rollout Workers](../05-rollout-workers/overview.md) - Worker architecture
- [Custom Workers](../05-rollout-workers/custom-workers.md) - Implementing custom workers
- [Evaluators](../05-rollout-workers/evaluators.md) - Evaluation patterns
- [Complete Config](./complete-config.md) - Full configuration example
