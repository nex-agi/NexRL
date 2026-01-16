# Creating Custom Rollout Workers

Custom rollout workers allow you to implement task-specific logic while leveraging NexRL's infrastructure. This guide covers common patterns and best practices for creating custom workers.

## When to Create Custom Workers

Create custom workers when you need to:

1. **Custom query formatting**: Task-specific prompt templates
2. **Multi-step processing**: Chain multiple LLM calls or computations
3. **Specialized evaluation**: Domain-specific reward calculation
4. **Custom token handling**: Special tokenization or loss masking
5. **Integration with external services**: APIs, databases, tools

## Worker Types

### Type 1: Simple Custom Worker (from BaseRolloutWorker)

For basic LLM tasks with custom logic:

```python
from nexrl.rollout_worker import BaseRolloutWorker
from nexrl.nexrl_types import Trajectory
from typing import Any
import logging

logger = logging.getLogger(__name__)

class MySimpleWorker(BaseRolloutWorker):
    """Custom worker for simple tasks."""

    def rollout(self, task: dict[str, Any]) -> str | None:
        # 1. Extract and validate task data
        if "question" not in task:
            logger.error("Task missing 'question' field")
            return None

        question = task["question"]
        context = task.get("context", "")

        # 2. Format prompt
        prompt = self._format_prompt(question, context)

        # 3. Call LLM
        result = self._inference_client.completion(
            prompt,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens
        )

        # 4. Process response
        answer = self._extract_answer(result["response"])

        # 5. Evaluate
        reward = self._compute_reward(answer, task)

        # 6. Create trajectory
        trajectory = Trajectory(
            tokens=result["prompt_tokens"] + result["response_tokens"],
            loss_mask=[0] * len(result["prompt_tokens"]) + [1] * len(result["response_tokens"]),
            reward=reward,
            is_val=task.get("is_val", False),
            extra_fields={
                "ground_truth": task.get("answer", ""),
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
                "extracted_answer": answer,
            }
        )

        # 7. Submit
        return self._put_trajectory(trajectory)

    def _format_prompt(self, question: str, context: str) -> str:
        """Format task-specific prompt."""
        if context:
            return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        return f"Question: {question}\n\nAnswer:"

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response."""
        # Task-specific extraction logic
        import re
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

    def _compute_reward(self, answer: str, task: dict) -> float:
        """Compute reward."""
        ground_truth = task.get("answer", "")
        return 1.0 if answer.strip() == ground_truth.strip() else 0.0
```

### Type 2: NexAU Custom Worker (from BaseNexAURolloutWorker)

For agent tasks with custom query formatting:

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

class MyAgentWorker(BaseNexAURolloutWorker):
    """Custom NexAU worker with specialized query formatting."""

    def run_agent(self, task: dict[str, Any]) -> tuple[Any, EvaluationRunResult]:
        """
        Run agent with custom query formatting and processing.

        Override this method to customize:
        - Query formatting
        - Agent configuration
        - Trace processing
        - Evaluation
        """
        # 1. Custom query formatting
        query = self._format_query(task)

        # 2. Load agent with custom LLM client
        agent, client_provider_func = self.load_agent_from_config(
            custom_llm_client_provider=lambda: self._inference_client
        )

        # 3. Run agent
        response = agent.run(query, custom_llm_client_provider=client_provider_func)

        # 4. Extract traces
        traces = []
        for tracer in agent.config.tracers:
            if isinstance(tracer, InMemoryTracer):
                traces = tracer.dump_traces()
                break

        # 5. Process traces
        trajectories = self.trace_processor(traces)

        # 6. Create agent output
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

        # 7. Evaluate
        evaluation_result = self.evaluator.evaluate(
            task,
            NexAUEvaluationTarget(
                final_answer=agent_output.final_answer,
                observation=agent_output.observation
            ),
        )

        # 8. Attach reward to trajectories
        for traj in trajectories:
            traj["reward"] = evaluation_result.reward
            traj["score"] = {
                "reward_score": evaluation_result.reward,
                **evaluation_result.metrics,
            }

        return agent_output, evaluation_result

    def _format_query(self, task: dict[str, Any]) -> str:
        """
        Custom query formatting for specific task.

        This is where task-specific prompt engineering happens.
        """
        # Example: Structured query format
        instruction = task.get("instruction", "")
        context = task.get("context", "")
        examples = task.get("examples", [])

        query_parts = []

        if instruction:
            query_parts.append(f"Instruction: {instruction}")

        if context:
            query_parts.append(f"\nContext: {context}")

        if examples:
            query_parts.append("\nExamples:")
            for i, ex in enumerate(examples, 1):
                query_parts.append(f"{i}. {ex}")

        query_parts.append("\nPlease solve the task:")

        return "\n".join(query_parts)
```

### Type 3: Multi-Step Worker

For complex multi-step reasoning:

```python
from nexrl.rollout_worker import BaseRolloutWorker
from nexrl.nexrl_types import Trajectory
from typing import Any
import logging

logger = logging.getLogger(__name__)

class MultiStepWorker(BaseRolloutWorker):
    """Worker with multi-step processing."""

    def rollout(self, task: dict[str, Any]) -> str | None:
        # Step 1: Generate reasoning
        reasoning_prompt = self._format_reasoning_prompt(task)
        reasoning_result = self._inference_client.completion(
            reasoning_prompt,
            temperature=0.8,  # Higher temperature for exploration
            max_tokens=1024
        )
        reasoning = reasoning_result["response"]

        # Step 2: Generate final answer based on reasoning
        answer_prompt = self._format_answer_prompt(task, reasoning)
        answer_result = self._inference_client.completion(
            answer_prompt,
            temperature=0.3,  # Lower temperature for final answer
            max_tokens=512
        )

        # Step 3: Combine trajectories
        # Merge prompt from step 1 with response from step 1 and prompt+response from step 2
        all_tokens = (
            reasoning_result["prompt_tokens"] +
            reasoning_result["response_tokens"] +
            answer_result["response_tokens"]  # Prompt overlaps, use only response
        )

        # Step 4: Create loss mask
        # Only train on final answer, mask out reasoning tokens
        loss_mask = (
            [0] * len(reasoning_result["prompt_tokens"]) +  # Mask prompt
            [0] * len(reasoning_result["response_tokens"]) +  # Mask reasoning
            [1] * len(answer_result["response_tokens"])  # Train on answer
        )

        # Step 5: Evaluate
        answer = self._extract_answer(answer_result["response"])
        reward = self._compute_reward(answer, task)

        # Step 6: Create trajectory
        trajectory = Trajectory(
            tokens=all_tokens,
            loss_mask=loss_mask,
            reward=reward,
            is_val=task.get("is_val", False),
            extra_fields={
                "ground_truth": task.get("answer", ""),
                "reasoning": reasoning,
                "final_answer": answer,
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
            }
        )

        return self._put_trajectory(trajectory)

    def _format_reasoning_prompt(self, task: dict) -> str:
        """Format prompt for reasoning step."""
        return f"Think step by step about: {task['question']}\n\nReasoning:"

    def _format_answer_prompt(self, task: dict, reasoning: str) -> str:
        """Format prompt for answer step."""
        return f"Question: {task['question']}\n\nReasoning: {reasoning}\n\nFinal Answer:"

    def _extract_answer(self, response: str) -> str:
        """Extract final answer."""
        # Implementation
        return response.strip()

    def _compute_reward(self, answer: str, task: dict) -> float:
        """Compute reward."""
        # Implementation
        return 1.0 if answer == task.get("answer", "") else 0.0
```

## Common Customization Patterns

### Pattern 1: Custom Prompt Templates

```python
class TemplateWorker(BaseRolloutWorker):
    def __init__(self, config):
        super().__init__(config)
        # Load prompt templates
        self.system_prompt = self._load_system_prompt()
        self.few_shot_examples = self._load_examples()

    def rollout(self, task):
        # Build prompt from templates
        prompt = self._build_prompt(task)
        result = self._inference_client.completion(prompt)
        # ... process result

    def _build_prompt(self, task):
        """Build prompt from templates."""
        parts = [self.system_prompt]

        # Add few-shot examples
        for example in self.few_shot_examples:
            parts.append(f"Q: {example['question']}")
            parts.append(f"A: {example['answer']}")

        # Add current task
        parts.append(f"Q: {task['question']}")
        parts.append("A:")

        return "\n\n".join(parts)

    def _load_system_prompt(self):
        """Load system prompt from file."""
        # Implementation
        return "You are a helpful assistant."

    def _load_examples(self):
        """Load few-shot examples."""
        # Implementation
        return []
```

### Pattern 2: External Tool Integration

```python
class ToolIntegratedWorker(BaseRolloutWorker):
    def __init__(self, config):
        super().__init__(config)
        self.api_client = self._init_api_client()

    def rollout(self, task):
        # Step 1: Query external API
        api_result = self.api_client.query(task["query"])

        # Step 2: Format prompt with API result
        prompt = self._format_prompt_with_context(task, api_result)

        # Step 3: Generate response
        llm_result = self._inference_client.completion(prompt)

        # Step 4: Create trajectory
        trajectory = self._create_trajectory(llm_result, task)
        return self._put_trajectory(trajectory)

    def _init_api_client(self):
        """Initialize external API client."""
        # Implementation
        return None
```

### Pattern 3: Custom Token Processing

```python
class CustomTokenWorker(BaseNexAURolloutWorker):
    def get_train_loss_mask(self, trajectory_infos: list[dict]) -> list[bool]:
        """
        Custom loss masking logic.

        Example: Only train on tokens after specific marker.
        """
        mask = []

        for traj in trajectory_infos:
            response_tokens = traj.get("response_tokens", [])
            response_text = self.tokenizer.decode(response_tokens)

            # Find marker position
            marker = "<final_answer>"
            if marker in response_text:
                # Tokenize up to marker to find split point
                marker_tokens = self.tokenizer.encode(marker, add_special_tokens=False)

                # Find marker position in tokens
                marker_pos = self._find_sublist(response_tokens, marker_tokens)

                if marker_pos >= 0:
                    # Mask tokens before marker, keep tokens after
                    mask.extend([False] * marker_pos)
                    mask.extend([True] * (len(response_tokens) - marker_pos))
                else:
                    # Marker not found, train on all
                    mask.extend([True] * len(response_tokens))
            else:
                # No marker, train on all
                mask.extend([True] * len(response_tokens))

        return mask

    def _find_sublist(self, full_list, sublist):
        """Find position of sublist in full list."""
        sublist_len = len(sublist)
        for i in range(len(full_list) - sublist_len + 1):
            if full_list[i:i+sublist_len] == sublist:
                return i
        return -1
```

### Pattern 4: Conditional Evaluation

```python
class ConditionalWorker(BaseRolloutWorker):
    def rollout(self, task):
        # Generate response
        result = self._inference_client.completion(task["prompt"])
        response = result["response"]

        # Conditional evaluation based on task type
        task_type = task.get("type", "default")

        if task_type == "exact_match":
            reward = self._exact_match_reward(response, task)
        elif task_type == "contains":
            reward = self._contains_reward(response, task)
        elif task_type == "similarity":
            reward = self._similarity_reward(response, task)
        else:
            reward = 0.0

        trajectory = Trajectory(
            tokens=result["prompt_tokens"] + result["response_tokens"],
            loss_mask=[0] * len(result["prompt_tokens"]) + [1] * len(result["response_tokens"]),
            reward=reward,
            extra_fields={"task_type": task_type}
        )

        return self._put_trajectory(trajectory)

    def _exact_match_reward(self, response, task):
        """Exact match evaluation."""
        return 1.0 if response.strip() == task["answer"].strip() else 0.0

    def _contains_reward(self, response, task):
        """Substring match evaluation."""
        return 1.0 if task["answer"] in response else 0.0

    def _similarity_reward(self, response, task):
        """Similarity-based evaluation."""
        # Implement similarity metric
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, response, task["answer"]).ratio()
        return float(ratio)
```

## Recipe Integration

### Directory Structure

```
recipe/
└── my_task/
    ├── my_task.yaml                    # Main config
    ├── my_task.env.sh                  # Environment setup
    └── agent_workspace/
        ├── my_worker.py                # Custom worker
        ├── evaluator.py                # Evaluator
        ├── agent_config.yaml           # Agent config (if NexAU)
        └── custom_tools.py             # Additional modules
```

### Configuration

```yaml
# my_task.yaml
rollout_worker:
  # For simple workers
  type: "simple"  # Or "nexau" for NexAU-based

  # Custom worker override
  custom_rollout_worker_module_path: "agent_workspace/my_worker.py"
  custom_rollout_worker_class_name: "MyCustomWorker"

  # Worker config
  num_workers: 4
  temperature: 0.7
  max_tokens: 2048

  # NexAU-specific (if applicable)
  nexau_agent_config_path: "agent_workspace/agent_config.yaml"
  evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"
  nexau_agent_workspace: "agent_workspace"
```

### Loading Custom Modules

Custom modules in workspace can import each other:

```python
# agent_workspace/custom_tools.py
def my_tool_function():
    return "result"

# agent_workspace/my_worker.py
from custom_tools import my_tool_function  # Works because workspace in sys.path

class MyWorker(BaseNexAURolloutWorker):
    def rollout(self, task):
        result = my_tool_function()
        # ... use result
```

## Best Practices

### 1. Error Handling

Always handle errors gracefully:

```python
def rollout(self, task):
    try:
        # Main logic
        result = self._process_task(task)
        trajectory = self._create_trajectory(result, task)
        return self._put_trajectory(trajectory)
    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        return None
    except ValueError as e:
        logger.error(f"Invalid task data: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error in rollout: {e}")
        # Activity tracker will auto-report this
        return None
```

### 2. Logging

Use structured logging:

```python
logger.debug(f"Processing task_id={task.get('task_id')}")
logger.info(f"Generated trajectory with reward={reward:.3f}")
logger.warning(f"No ground truth for task_id={task.get('task_id')}")
logger.error(f"Failed to extract answer from response")
```

### 3. Configuration

Make behavior configurable:

```python
class MyWorker(BaseRolloutWorker):
    def __init__(self, config):
        super().__init__(config)
        # Load worker-specific config
        self.use_few_shot = config.get("use_few_shot", True)
        self.num_examples = config.get("num_examples", 3)
        self.answer_format = config.get("answer_format", "xml")
```

```yaml
rollout_worker:
  custom_rollout_worker_module_path: "agent_workspace/my_worker.py"
  custom_rollout_worker_class_name: "MyWorker"

  # Custom worker config
  use_few_shot: true
  num_examples: 5
  answer_format: "json"
```

### 4. Testing

Test worker logic independently:

```python
# test_my_worker.py
import pytest
from omegaconf import DictConfig
from my_worker import MyWorker

def test_query_formatting():
    config = DictConfig({"temperature": 0.7})
    worker = MyWorker(config)

    task = {"question": "What is 2+2?", "context": "Math problem"}
    query = worker._format_query(task)

    assert "What is 2+2?" in query
    assert "Math problem" in query

def test_answer_extraction():
    config = DictConfig({"temperature": 0.7})
    worker = MyWorker(config)

    response = "The answer is <answer>4</answer>"
    answer = worker._extract_answer(response)

    assert answer == "4"
```

### 5. Reusability

Create utility functions for common operations:

```python
# utils.py (in workspace)
import re

def extract_xml_tag(text: str, tag: str) -> str | None:
    """Extract content from XML-style tags."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_json(text: str) -> dict:
    """Extract JSON from text."""
    import json
    # Implementation
    return {}

# my_worker.py
from utils import extract_xml_tag

class MyWorker(BaseRolloutWorker):
    def _extract_answer(self, response):
        return extract_xml_tag(response, "answer") or response
```

## Debugging Tips

### Enable Trace Saving

For NexAU workers:

```yaml
rollout_worker:
  save_trace: true
  trace_path: "outputs/traces"
```

### Add Debug Logging

```python
def rollout(self, task):
    logger.debug(f"Task: {task}")

    result = self._inference_client.completion(prompt)
    logger.debug(f"LLM result: {result}")

    answer = self._extract_answer(result["response"])
    logger.debug(f"Extracted answer: {answer}")

    reward = self._compute_reward(answer, task)
    logger.debug(f"Computed reward: {reward}")
```

### Inspect Trajectories

Add extra fields for debugging:

```python
trajectory = Trajectory(
    tokens=tokens,
    loss_mask=loss_mask,
    reward=reward,
    extra_fields={
        # Required
        "ground_truth": task["answer"],

        # Debug info
        "raw_response": result["response"],
        "extracted_answer": answer,
        "prompt": prompt,
        "task_id": task.get("task_id"),
    }
)
```

## Common Pitfalls

### Pitfall 1: Not Handling Missing Fields

```python
# BAD
prompt = task["prompt"]  # Crashes if missing

# GOOD
prompt = task.get("prompt")
if prompt is None:
    logger.error("Task missing prompt")
    return None
```

### Pitfall 2: Incorrect Loss Mask Length

```python
# BAD - mask length doesn't match tokens
loss_mask = [1] * 100  # Fixed length
tokens = prompt_tokens + response_tokens  # Variable length

# GOOD - mask length matches tokens
loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
assert len(loss_mask) == len(tokens)
```

### Pitfall 3: Not Returning Submission Result

```python
# BAD - doesn't handle re-rollout
def rollout(self, task):
    trajectory = self._create_trajectory(task)
    self._put_trajectory(trajectory)
    return "success"  # Always returns success

# GOOD - returns actual result
def rollout(self, task):
    trajectory = self._create_trajectory(task)
    return self._put_trajectory(trajectory)  # Returns actual status
```

## Next Steps

- [Evaluators](./evaluators.md) - Implementing evaluation logic
- [Base Rollout Worker](./base-rollout-worker.md) - Core worker interface
- [NexAU Rollout Worker](./nexau-rollout-worker.md) - Agent framework details
- [Examples](../14-examples/simple-task.md) - Complete examples

## Related Documentation

- [Recipe Structure](../10-recipes/recipe-structure.md) - Organizing recipes
- [Data Types](../02-core-architecture/data-types.md) - Trajectory structure
- [Configuration Reference](../12-configuration-reference/rollout-config.md) - Full config options
