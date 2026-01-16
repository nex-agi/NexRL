# Evaluators

Evaluators compute rewards and metrics for agent outputs. They are the critical link between task execution and reinforcement learning, translating task success into training signals.

## Overview

**Location:** `nexrl/rollout_worker/base_nexau_rollout_worker.py`

```python
from nexrl.rollout_worker import (
    Evaluator,
    BaseEvaluationTarget,
    NexAUEvaluationTarget,
    EvaluationRunResult
)
```

## Core Classes

### Evaluator Base Class

The abstract base class that all evaluators must extend:

```97:135:nexrl/rollout_worker/base_nexau_rollout_worker.py
class Evaluator(ABC):
    """
    Abstract base class for evaluators.

    Subclasses must implement the evaluate method to define specific evaluation logic.
    """

    def __init__(self) -> None:
        """Initialize the evaluator."""

    @abstractmethod
    def evaluate(
        self,
        data: dict[str, Any],
        evaluation_target: BaseEvaluationTarget,
    ) -> EvaluationRunResult:
        """
        Perform evaluation.

        Args:
            data: Input data, should contain at least query information
            evaluation_target: Agent output target for evaluation

        Returns:
            EvaluationRunResult: Evaluation result
        """
        # Default implementation: simple exact match evaluation
        ground_truth = data.get("ground_truth", "")
        reward = 0.0

        if isinstance(ground_truth, str):
            reward = float(evaluation_target.final_answer == ground_truth)

        return EvaluationRunResult(
            reward=reward,
            ground_truth=str(ground_truth),
            metrics={},
            extra_info={},
        )
```

### Evaluation Target

```49:63:nexrl/rollout_worker/base_nexau_rollout_worker.py
@dataclass
class BaseEvaluationTarget:
    """Base class for evaluation targets."""

    final_answer: str


@dataclass
class NexAUEvaluationTarget(BaseEvaluationTarget):
    """Evaluation target for NexAU agents."""

    final_answer: str  # Final answer produced by the agent
    observation: list[dict[str, Any]]  # Complete execution trajectory containing
    # all intermediate steps and observations
```

- **`BaseEvaluationTarget`**: Used for simple rollout workers, contains only the final answer
- **`NexAUEvaluationTarget`**: Used for NexAU agents, includes the complete execution trajectory with intermediate steps

### Evaluation Result

```70:89:nexrl/rollout_worker/base_nexau_rollout_worker.py
@dataclass
class EvaluationRunResult:
    """
    Evaluator execution result.

    Attributes:
        reward: Evaluation score
        ground_truth: Ground truth answer
        metrics: Additional metrics (must be scalar floats)
        extra_info: Additional information (can be any type)
    """

    reward: float = 0.0
    ground_truth: str = ""
    metrics: dict[str, float] = field(
        default_factory=dict
    )  # pyright: ignore[reportUnknownVariableType]
    extra_info: dict[str, Any] = field(
        default_factory=dict
    )  # pyright: ignore[reportUnknownVariableType]
```

**Important:**
- **`reward`**: Primary RL signal, should be in [0.0, 1.0] range
- **`ground_truth`**: Reference answer for logging/debugging
- **`metrics`**: Must be scalar floats for aggregation across trajectories
- **`extra_info`**: Can be any type, used for debugging/analysis

## Real-World Examples

### Example 1: Classification Evaluator (NewsEvaluator)

This evaluator is used in the `recipe/nexau_news` task for binary classification with flexible accuracy:

```python
# recipe/nexau_news/agent_workspace/evaluator.py
import re
from typing import Any
import pandas as pd
from nexrl.rollout_worker import BaseEvaluationTarget, EvaluationRunResult, Evaluator


class NewsEvaluator(Evaluator):
    def extract_answer(self, response: str) -> str | None:
        answer_match = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer_match:
            return answer_match[-1].strip()
        else:
            return None

    def parse_answer_to_bool(self, answer_str):
        """Convert answer string to boolean value."""
        if answer_str is None:
            return None
        answer_lower = answer_str.lower().strip()
        if answer_lower in ["true", "是", "yes", "1", "t"]:
            return True
        elif answer_lower in ["false", "否", "no", "0", "f"]:
            return False
        else:
            return None

    def evaluate(self, data: Any, evaluation_target: BaseEvaluationTarget) -> EvaluationRunResult:
        extracted_answer_str = self.extract_answer(evaluation_target.final_answer)
        extracted_answer_bool = self.parse_answer_to_bool(extracted_answer_str)

        # Handle annotation field for flexible accuracy
        annotation = data.get("标注")
        if annotation is None or pd.isna(annotation):
            true_label = bool(data.get("通过", False))
            if extracted_answer_bool == true_label:
                accuracy_flex = 1
            else:
                accuracy_flex = 0
        else:
            true_label = annotation in ("必须推", "都可以")
            if extracted_answer_bool == true_label or annotation == "都可以":
                accuracy_flex = 1
            else:
                accuracy_flex = 0

        # Calculate metrics
        tp = 0
        fp = 0
        fn = 0
        accuracy = 0

        answer_extracted = extracted_answer_bool is not None
        if answer_extracted:
            if extracted_answer_bool == true_label:
                accuracy = 1
                if true_label:
                    tp = 1
            else:
                if extracted_answer_bool and not true_label:
                    fp = 1
                elif not extracted_answer_bool and true_label:
                    fn = 1
        else:
            if true_label:
                fn = 1
            else:
                accuracy = 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            "accuracy_strict": accuracy,
            "accuracy_flex": accuracy_flex,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        return EvaluationRunResult(
            reward=accuracy_flex,
            ground_truth=str(true_label),
            metrics=metrics,
            extra_info={},
        )
```

**Key features:**
- Regex-based answer extraction from `<answer>` tags
- Boolean parsing with multiple language support
- Flexible accuracy with annotation support
- Classification metrics: precision, recall, F1
- Graceful handling of extraction failures

### Example 2: F1 Score Evaluator (DeepResearchEvaluator)

This evaluator is used in the `recipe/nexau_deepsearch` task for text-based F1 scoring:

```python
# recipe/nexau_deepsearch/agent_workspace/evaluator.py
import re
import string
from typing import Any
from nexrl.rollout_worker import EvaluationRunResult, Evaluator, NexAUEvaluationTarget


def extract_answer(response: str) -> str:
    answer_match = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        return answer_match[-1].strip()
    else:
        return None


def f1_preprocess_text(text: str) -> str:
    """Preprocess text for F1 scoring."""
    text = text.lower()
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def calc_score(answer_content: str, ground_truth: str) -> float:
    """Calculate F1 score between the answer and ground truth."""
    answer_content = f1_preprocess_text(answer_content)

    # Handle multiple ground truths separated by <|answer_split|>
    ground_truths = [f1_preprocess_text(ground_truth)]
    if isinstance(ground_truth, str) and "<|answer_split|>" in ground_truth:
        ground_truths = [f1_preprocess_text(_) for _ in ground_truth.split("<|answer_split|>")]

    max_score = 0.0

    for gt_option in ground_truths:
        # Tokenize answer and ground truth
        pred_tokens = set(answer_content.split())
        gt_tokens = set(gt_option.split())

        if not gt_tokens or not pred_tokens:
            continue

        # Calculate common tokens
        common_tokens = pred_tokens & gt_tokens

        # Calculate precision and recall
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
        recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0

        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            max_score = max(max_score, f1)

    return max_score


def reward_function(response: str, ground_truth: str) -> float:
    extracted_answer = extract_answer(response)
    if extracted_answer is None:
        print(f"No answer found in response: {response}")
        return 0.0
    return calc_score(extracted_answer, ground_truth)


class DeepResearchEvaluator(Evaluator):
    def evaluate(self, data: Any, evaluation_target: NexAUEvaluationTarget) -> EvaluationRunResult:
        reward = reward_function(evaluation_target.final_answer, data["ground_truth"])
        ground_truth = data["ground_truth"]
        return EvaluationRunResult(
            reward=reward, ground_truth=ground_truth, metrics={}, extra_info={}
        )
```

**Key features:**
- Answer extraction from `<answer>` tags
- Text preprocessing for F1 scoring (lowercase, remove punctuation)
- Token-based F1 score calculation
- Support for multiple valid ground truths
- Returns continuous reward in [0.0, 1.0]

## Recipe Integration

### File Structure

Evaluators are placed in the recipe's agent workspace:

```
recipe/
└── my_task/
    ├── my_task.yaml                  # Main recipe config
    └── agent_workspace/
        └── evaluator.py              # Evaluator implementation
```

### Configuration

Reference the evaluator in your recipe config:

```yaml
# my_task.yaml
rollout_worker:
  type: "nexau"
  evaluator_module_path: "agent_workspace/evaluator.py:MyEvaluator"
  # ... other config
```

The path is relative to the config file location.

### Implementation Template

```python
# agent_workspace/evaluator.py
from typing import Any
from nexrl.rollout_worker import (
    Evaluator,
    BaseEvaluationTarget,
    EvaluationRunResult
)


class MyEvaluator(Evaluator):
    """Task-specific evaluator."""

    def __init__(self):
        """Initialize evaluator with any required resources."""
        super().__init__()
        # Load reference data, models, etc.

    def evaluate(
        self,
        data: dict[str, Any],
        evaluation_target: BaseEvaluationTarget
    ) -> EvaluationRunResult:
        """
        Evaluate agent output.

        Args:
            data: Original task data (may contain ground truth)
            evaluation_target: Agent output to evaluate

        Returns:
            EvaluationRunResult with reward, metrics, and extra info
        """
        # Extract ground truth
        ground_truth = data.get("answer", "")

        # Extract agent answer
        agent_answer = evaluation_target.final_answer

        # Compute reward (your logic here)
        reward = self._compute_reward(agent_answer, ground_truth)

        # Compute additional metrics
        metrics = {
            "accuracy": reward,
            # Add more metrics as needed
        }

        return EvaluationRunResult(
            reward=reward,
            ground_truth=ground_truth,
            metrics=metrics,
            extra_info={
                "agent_answer": agent_answer,
                # Add debugging info as needed
            }
        )

    def _compute_reward(self, answer: str, ground_truth: str) -> float:
        """Compute reward logic."""
        # Implement your evaluation logic
        return 1.0 if answer == ground_truth else 0.0
```

## Best Practices

### Reward Design

**Keep rewards in [0.0, 1.0] range:**

```python
# GOOD - normalized
reward = 1.0 if correct else 0.0
reward = float(score) / max_score

# BAD - unbounded
reward = num_correct * 10
```

**Use meaningful scales:**

```python
# Binary tasks
reward = 1.0 if correct else 0.0

# Partial credit
reward = 0.0   # completely wrong
reward = 0.5   # partially correct
reward = 1.0   # fully correct

# Continuous (e.g., F1 score)
reward = f1_score  # Already in [0, 1]
```

### Metrics Design

**All metrics must be scalar floats:**

```python
# GOOD
metrics = {
    "accuracy": 1.0,
    "precision": 0.85,
    "f1": 0.875,
}

# BAD - not scalar
metrics = {
    "predictions": [0, 1, 1, 0],  # List
}
```

**Save complex data in extra_info:**

```python
extra_info = {
    "predictions": [0, 1, 1, 0],
    "raw_response": evaluation_target.final_answer,
}
```

### Error Handling

**Handle missing or malformed data gracefully:**

```python
def evaluate(self, data, evaluation_target):
    # Validate data
    if "answer" not in data:
        return EvaluationRunResult(
            reward=0.0,
            ground_truth="",
            metrics={"validation_error": 1.0},
            extra_info={"error": "Missing ground truth"}
        )

    # Extract answer with fallback
    agent_answer = evaluation_target.final_answer
    if not agent_answer:
        return EvaluationRunResult(
            reward=0.0,
            ground_truth=data["answer"],
            metrics={"empty_response": 1.0},
            extra_info={}
        )

    # Normal evaluation
    # ...
```

### Determinism

**Make evaluation deterministic when possible:**

```python
# GOOD - deterministic
reward = 1.0 if answer == ground_truth else 0.0

# Avoid non-deterministic evaluation unless necessary
```

## Testing Evaluators

### Unit Tests

```python
# test_evaluator.py
import pytest
from evaluator import MyEvaluator
from nexrl.rollout_worker import BaseEvaluationTarget

def test_correct_answer():
    evaluator = MyEvaluator()

    data = {"answer": "42"}
    target = BaseEvaluationTarget(final_answer="42")

    result = evaluator.evaluate(data, target)

    assert result.reward == 1.0
    assert result.ground_truth == "42"

def test_incorrect_answer():
    evaluator = MyEvaluator()

    data = {"answer": "42"}
    target = BaseEvaluationTarget(final_answer="43")

    result = evaluator.evaluate(data, target)

    assert result.reward == 0.0

def test_missing_ground_truth():
    evaluator = MyEvaluator()

    data = {}
    target = BaseEvaluationTarget(final_answer="42")

    result = evaluator.evaluate(data, target)

    # Should handle gracefully
    assert isinstance(result, EvaluationRunResult)
```

## Common Issues

### Issue 1: Reward Not in [0, 1]

```python
# BAD
reward = num_correct  # Could be any value

# GOOD
reward = num_correct / total_questions
reward = min(1.0, max(0.0, reward))  # Clamp to range
```

### Issue 2: Non-Scalar Metrics

```python
# BAD
metrics = {
    "predictions": [0, 1, 1, 0],  # Not scalar
}

# GOOD
metrics = {
    "accuracy": 0.75,  # Scalar
}
extra_info = {
    "predictions": [0, 1, 1, 0],  # Store details here
}
```

### Issue 3: Case Sensitivity

```python
# BAD
reward = 1.0 if answer == ground_truth else 0.0  # "Yes" != "yes"

# GOOD
reward = 1.0 if answer.lower() == ground_truth.lower() else 0.0
```

## Next Steps

- [Custom Workers](./custom-workers.md) - Integrating evaluators with workers
- [Overview](./overview.md) - Rollout worker architecture
- [Base Rollout Worker](./base-rollout-worker.md) - Worker base classes

## Related Documentation

- [Data Types](../02-core-architecture/data-types.md) - Trajectory and evaluation types
- [NexAU Rollout Worker](./nexau-rollout-worker.md) - NexAU agent integration
