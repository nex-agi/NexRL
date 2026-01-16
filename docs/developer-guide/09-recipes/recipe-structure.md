# Recipe Structure

A recipe in NexRL is a self-contained package that defines a complete training task, including data, rollout workers, evaluators, and configuration.

## Overview

Recipes organize all task-specific code and configuration in a dedicated directory under `recipe/`. This design keeps tasks modular and portable - you can share recipes as standalone units.

## Directory Layout

### Basic Recipe Structure

```
recipe/
└── my_task/
    ├── common.yaml              # Shared configuration
    ├── self_hosted.yaml         # Self-hosted mode config
    ├── tinker.yaml              # Tinker service config
    ├── weaver.yaml              # Weaver service config
    ├── self_hosted.env.sh       # Self-hosted environment setup
    ├── tinker.env.sh            # Tinker environment setup
    ├── weaver.env.sh            # Weaver environment setup
    └── README.md                # Recipe documentation
```

### NexAU Recipe Structure

For tasks using NexAU agents, add an `agent_workspace/` directory:

```
recipe/
└── my_task/
    ├── common.yaml
    ├── self_hosted.yaml
    ├── tinker.yaml
    ├── weaver.yaml
    ├── self_hosted.env.sh
    ├── tinker.env.sh
    ├── weaver.env.sh
    ├── README.md
    └── agent_workspace/
        ├── agent_config.yaml    # NexAU agent configuration
        ├── evaluator.py         # Task evaluator implementation
        ├── my_rollout_worker.py # (Optional) Custom rollout worker
        └── tools/               # (Optional) Custom agent tools
            └── MyTool.yaml
```

## File Purposes

### Configuration Files

**`common.yaml`**
- Contains shared configuration used across all deployment modes
- Defines data loading, rollout workers, trajectory pool, validation
- Uses Hydra environment variable interpolation: `${oc.env:VAR_NAME}`

**`{mode}.yaml`** (self_hosted, tinker, weaver)
- Mode-specific overrides for deployment
- Inherits from `common.yaml` using Hydra defaults
- Configures trainers, services, and resources for that mode

**`{mode}.env.sh`**
- Environment setup script for that deployment mode
- Exports necessary environment variables
- Optional but recommended for reproducible setups

### Agent Workspace Files

**`agent_workspace/agent_config.yaml`**
- Configures the NexAU agent (system prompt, LLM settings, tools)
- Referenced from recipe config: `rollout_worker.nexau_agent_config_path`

**`agent_workspace/evaluator.py`**
- Implements task-specific evaluation logic
- Must define an `Evaluator` subclass
- Referenced from recipe config: `rollout_worker.evaluator_module_path`

**`agent_workspace/*.py`** (optional)
- Custom rollout workers for special task requirements
- Custom tools or utilities for the agent
- Loaded via `rollout_worker.custom_rollout_worker_module_path`

## Real Examples

### Simple Task (Math)

The math task demonstrates a minimal recipe without agent workspace:

```
recipe/math/
├── common.yaml              # Basic configuration
├── self_hosted.yaml         # Self-hosted training
├── tinker.yaml              # Tinker service
├── weaver.yaml              # Weaver service
└── README.md
```

Data format: Parquet with `prompt` and `ground_truth` columns.

### NexAU Task (News)

The news task shows a complete NexAU recipe structure:

```
recipe/nexau_news/
├── common.yaml
├── self_hosted.yaml
├── tinker.yaml
├── weaver.yaml
├── self_hosted.env.sh
├── tinker.env.sh
├── weaver.env.sh
├── README.md
└── agent_workspace/
    ├── agent_config.yaml       # News agent prompt and config
    ├── evaluator.py            # NewsEvaluator with F1/precision
    └── news_rollout_worker.py  # Custom query formatting
```

### NexAU Task with Tools (Deep Search)

The deep search task includes custom tools:

```
recipe/nexau_deepsearch/
├── common.yaml
├── self_hosted.yaml
├── tinker.yaml
├── weaver.yaml
├── self_hosted.env.sh
├── tinker.env.sh
├── weaver.env.sh
├── README.md
└── agent_workspace/
    ├── agent_config.yaml
    ├── evaluator.py
    ├── web_tool.py             # Custom tool implementation
    └── tools/
        └── WebSearch.yaml      # Tool definition
```

## Path References

### Relative Paths in Configuration

Recipe configuration files use paths relative to the recipe directory:

```yaml
rollout_worker:
  # Relative to recipe directory
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
  evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"
  custom_rollout_worker_module_path: "agent_workspace/my_worker.py"
  custom_rollout_worker_class_name: "MyWorker"
```

### Module Path Format

For Python modules, use the format: `path/to/file.py:ClassName`

```yaml
evaluator_module_path: "agent_workspace/evaluator.py:NewsEvaluator"
```

The controller will:
1. Resolve path relative to `NEXRL_PATH` environment variable
2. Import the module
3. Extract the specified class

## Best Practices

### Organization

1. **Keep recipes self-contained** - All task-specific code in recipe directory
2. **Use common.yaml** - Share configuration across deployment modes
3. **Document in README.md** - Explain task, data format, usage
4. **Version control recipes** - Recipes are portable and shareable

### Configuration

1. **Use environment variables** - For paths, API keys, URLs
2. **Leverage Hydra composition** - Inherit from common, override in mode-specific
3. **Provide all modes** - self_hosted, tinker, weaver for flexibility
4. **Include environment scripts** - Make setup reproducible

### Code

1. **Implement custom evaluators** - Task-specific reward computation
2. **Extend workers when needed** - Override `format_task_query` for custom prompting
3. **Add tools carefully** - Only include necessary agent tools
4. **Follow naming conventions** - Match file names to class names

### Data

1. **Use shared storage paths** - Environment variables like `NEXRL_DATA_PATH`
2. **Document data format** - Required columns and structure
3. **Provide train/test splits** - For validation
4. **Use standard formats** - Parquet recommended for performance

## Creating a New Recipe

### Step 1: Create Directory Structure

```bash
mkdir -p recipe/my_task/agent_workspace
```

### Step 2: Create Configuration Files

Start with `common.yaml`:

```yaml
project_name: "NexRL-MyTask"
experiment_name: "my-task-training"
launch_mode: "local"

data:
  type: "torch"
  data_files:
    - "${oc.env:NEXRL_DATA_PATH}/my_task/train.parquet"
  batch_size: 16

rollout_worker:
  type: "nexau"
  num_workers: 128
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
  evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"
```

### Step 3: Create Agent Configuration

Create `agent_workspace/agent_config.yaml`:

```yaml
type: agent
name: my_task_agent
max_context_tokens: 100000
system_prompt: "Your task-specific prompt here..."
system_prompt_type: string
llm_config:
  temperature: 0.7
  max_tokens: 8192
  api_type: openai_chat_completion

tracers:
  - import: nexau.archs.tracer.adapters.in_memory:InMemoryTracer
```

### Step 4: Implement Evaluator

Create `agent_workspace/evaluator.py`:

```python
from nexrl.rollout_worker import Evaluator, BaseEvaluationTarget, EvaluationRunResult

class MyEvaluator(Evaluator):
    def evaluate(self, data, evaluation_target):
        # Your evaluation logic
        reward = 1.0 if self._check_correctness(data, evaluation_target) else 0.0
        return EvaluationRunResult(
            reward=reward,
            ground_truth=data.get("answer", ""),
            metrics={"accuracy": reward},
            extra_info={}
        )

    def _check_correctness(self, data, evaluation_target):
        # Implement your logic
        return True
```

### Step 5: Create Mode-Specific Configs

Create `self_hosted.yaml`:

```yaml
defaults:
  - common
  - _self_

environment:
  setup_script: "self_hosted.env.sh"
  require_setup_script: false

trainer:
  type: "self_hosted_grpo"
  # ... trainer configuration

service:
  inference_service:
    base_url: "${oc.env:INFERENCE_BASE_URL}"
    # ... service configuration
```

### Step 6: Document in README.md

```markdown
# My Task

## Task Description
Brief description of the task.

## Data Format
Expected data columns and format.

## Usage
```bash
nexrl -m self-hosted -c recipe/my_task/self_hosted.yaml --run-nexrl
```

## Configuration
Required environment variables.
```

## Related Documentation

- [Agent Configuration](./agent-configuration.md) - NexAU agent setup details
- [Recipe Configuration](./recipe-configuration.md) - YAML configuration and Hydra
- [Environment Setup](./environment-setup.md) - Environment scripts and variables
- [Custom Workers](../05-rollout-workers/custom-workers.md) - Implementing custom rollout workers
- [Evaluators](../05-rollout-workers/evaluators.md) - Evaluation patterns

---

**Next**: [Agent Configuration](./agent-configuration.md) - Learn how to configure NexAU agents for your task.
