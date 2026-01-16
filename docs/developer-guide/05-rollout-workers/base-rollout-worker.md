# Base Rollout Worker

The `BaseRolloutWorker` is the abstract base class for all rollout workers in NexRL. It provides the core infrastructure for task processing, LLM interaction, and trajectory generation.

## Class Definition

**Location:** `nexrl/rollout_worker/base_rollout_worker.py`

```python
from nexrl.rollout_worker import BaseRolloutWorker

class BaseRolloutWorker(NexRLModule, ABC):
    """Base class for rollout workers."""
```

## Constructor

```python
def __init__(self, config: DictConfig):
    """
    Initialize the rollout worker.

    Args:
        config: Worker configuration containing:
            - inference_service: Inference service config
            - need_llm_inference: Whether LLM inference is needed
            - temperature: Sampling temperature
            - max_tokens: Maximum tokens
            - ... worker-specific config
    """
```

The constructor:
1. Initializes threading components
2. Sets up stop event for graceful shutdown
3. Prepares module reference placeholders (set later)
4. Does NOT initialize inference client (deferred to `init_inference_service_client()`)

## Abstract Method

### rollout()

The core method that derived classes must implement:

```python
@abstractmethod
def rollout(self, task: dict[str, Any]) -> str | None:
    """
    Execute a single rollout operation.

    Args:
        task: Task dictionary from data loader

    Returns:
        - "success": Trajectory submitted successfully
        - "fail": Failed to submit trajectory
        - "re-rollout": Weight sync in progress, retry this task
        - None: Processing failed before trajectory creation
    """
```

**Implementation guidelines:**
1. Extract required fields from `task`
2. Perform task-specific processing (LLM calls, agent execution, etc.)
3. Create `Trajectory` object with results
4. Call `self._put_trajectory(trajectory)` and return its result

## Setup Methods

### set_module_references()

Called by controller to establish inter-module connections:

```python
def set_module_references(
    self,
    trajectory_pool: TrajectoryPool,
    dataloader: BaseDataLoader,
    weight_sync_controller: WeightSyncController,
    validate_dataloader: BaseDataLoader,
    validator: Validator,
):
    """Set module references for the worker."""
```

**When called:** During controller initialization, before `run()`

### init_inference_service_client()

Initializes the LLM inference client:

```python
def init_inference_service_client(self, service_holder=None):
    """
    Initialize inference service client.

    Args:
        service_holder: Shared service holder (Tinker/Weaver) if applicable
    """
```

**Features:**
- Only initializes if `need_llm_inference=True`
- Supports multiple backends (vLLM, Tinker, Weaver, etc.)
- Sets up weight sync coordination

**When called:** During controller initialization, after module references are set

### set_activity_tracker()

Sets the activity monitoring proxy:

```python
def set_activity_tracker(self, tracker: ActivityTrackerProxy):
    """Set activity tracker for monitoring."""
```

Inherited from `NexRLModule`.

## Lifecycle Methods

### run()

Starts the worker thread:

```python
def run():
    """Start worker execution in background thread."""
```

**Process:**
1. Validates that module references are set
2. Clears stop event
3. Creates and starts worker thread running `_main_loop()`

**Preconditions:**
- `set_module_references()` must have been called
- `set_activity_tracker()` must have been called
- `init_inference_service_client()` must have been called (if needed)

### stop()

Gracefully stops the worker:

```python
def stop():
    """Stop the worker and wait for thread completion."""
```

**Process:**
1. Sets stop event
2. Waits for worker thread to finish

### _main_loop()

Internal method that runs in the worker thread:

```python
def _main_loop(self):
    """Main worker loop."""
```

**Process:**
```python
while not self._stop_event.is_set():
    task = self._get_rollout_task()
    if task is None:
        sleep(0.1)
        continue

    with self._activity_tracker.track(self._module_name, "rollout"):
        while task is not None:
            result = self.rollout(task)

            if result != "re-rollout":
                task = self._get_rollout_task()
            # else: retry same task
```

**Key features:**
- Tracks activity for monitoring
- Handles re-rollout logic (weight sync coordination)
- Non-blocking with sleep when no tasks available

## Data Flow Methods

### _get_rollout_task()

Fetches the next task from appropriate data loader:

```python
def _get_rollout_task(self) -> dict[str, Any] | None:
    """
    Get next rollout task from data loader.

    Returns:
        Task dictionary or None if no tasks available
    """
```

**Behavior:**
- Uses validation data loader if `_is_running_validate=True`
- Uses training data loader otherwise
- Sleeps briefly if no tasks to prevent busy waiting
- Non-blocking operation

### _put_trajectory()

Submits trajectory to appropriate destination:

```python
def _put_trajectory(self, trajectory: Trajectory) -> str:
    """
    Submit trajectory to pool or validator.

    Args:
        trajectory: Trajectory to submit

    Returns:
        - "success": Submitted successfully
        - "fail": Failed to submit
        - "re-rollout": Weight sync in progress, should retry
    """
```

**Routing:**
- **Validation mode** (`_is_running_validate=True`): Routes to validator
- **Training mode**: Routes to trajectory pool

### _put_rollout_task()

Returns a task to the data loader:

```python
def _put_rollout_task(self, task: dict[str, Any]) -> bool:
    """
    Return task to data loader for reprocessing.

    Args:
        task: Task to return

    Returns:
        True if successfully returned, False otherwise
    """
```

**Use case:** When a task needs to be retried later (e.g., due to transient errors)

## Validation Mode Methods

### begin_validate()

Switches worker to validation mode:

```python
def begin_validate():
    """Switch to validation mode."""
```

**Effects:**
- `_is_running_validate` set to `True`
- Subsequent tasks from validation data loader
- Trajectories routed to validator

**When called:** By controller at start of validation cycle

### end_validate()

Switches worker back to training mode:

```python
def end_validate():
    """Switch back to training mode."""
```

**Effects:**
- `_is_running_validate` set to `False`
- Subsequent tasks from training data loader
- Trajectories routed to trajectory pool

**When called:** By controller at end of validation cycle

## Inference Client Interface

Workers access LLM through `_inference_client`:

### completion()

Single-turn completion:

```python
result = self._inference_client.completion(
    prompt="Your prompt here",
    temperature=0.7,
    max_tokens=512,
    # ... additional parameters
)
```

**Returns:**
```python
{
    "prompt": str,           # Original prompt
    "response": str,         # Generated text
    "finish_reason": str,    # "stop", "length", etc.
    "prompt_tokens": list[int],     # Token IDs (if available)
    "response_tokens": list[int],   # Token IDs (if available)
    "response_logprobs": list[float],  # Log probabilities (if available)
    # ... additional fields from kwargs
}
```

### generate()

Chat-style generation with messages:

```python
result = self._inference_client.generate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=512,
    tools=[...],  # Optional tool definitions
)
```

**Returns:**
```python
{
    "messages": list[dict],  # Original messages
    "response": str,         # Generated text
    "tool_calls": list[dict],  # Tool calls made (if any)
    "finish_reason": str,    # Completion status
    # ... additional fields
}
```

### Weight Sync Coordination

The inference client automatically handles weight sync:
- Blocks during weight synchronization if `freeze_for_weight_sync=True`
- This is transparent to the worker implementation

## Example Implementation: SimpleRolloutWorker

Here's the complete implementation of `SimpleRolloutWorker`:

```python
from nexrl.rollout_worker import BaseRolloutWorker
from nexrl.nexrl_types import Trajectory
import logging
import re

class SimpleRolloutWorker(BaseRolloutWorker):
    """Simple worker for basic LLM completion tasks."""

    def rollout(self, task: dict[str, Any]) -> str | None:
        # 1. Validate task
        if "prompt" not in task:
            logger.error(f"Task missing 'prompt' field: {task}")
            return None

        prompt = task["prompt"]

        # 2. Call LLM
        assert self._inference_client is not None
        completion_result = self._inference_client.completion(prompt)

        # 3. Extract tokens
        prompt_tokens = completion_result.get("prompt_tokens", [])
        response_tokens = completion_result.get("response_tokens", [])
        response_logprobs = completion_result.get("response_logprobs", [])

        # 4. Create token sequence and loss mask
        tokens = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)

        # 5. Extract answer (task-specific)
        response = completion_result.get("response", "")
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        extracted_answer = answer_match.group(1).strip() if answer_match else ""

        # 6. Get ground truth
        ground_truth = task.get("ground_truth", "")

        # 7. Calculate reward
        reward = 1.0 if extracted_answer == ground_truth else 0.0

        # 8. Create trajectory
        trajectory = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=reward,
            is_val=task.get("is_val", False),
            extra_fields={
                "ground_truth": ground_truth,
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
                "task_id": task.get("task_id", 0),
                "temperature": self._config.temperature,
                "finish_reason": completion_result.get("finish_reason", "stop"),
                "logprobs": [0.0] * len(prompt_tokens) + response_logprobs,
                "response": response,
                "extracted_answer": extracted_answer,
            },
        )

        # 9. Submit trajectory
        return self._put_trajectory(trajectory)
```

## Configuration Example

```yaml
rollout_worker:
  type: "simple"
  num_workers: 4

  # Inference service
  inference_service:
    backend: "vllm"
    url: "http://localhost:8000/v1"
    model_tag: "default"
    api_key: "EMPTY"
    max_retries: 3
    freeze_for_weight_sync: true

  # Worker config
  need_llm_inference: true
  temperature: 0.7
  max_tokens: 2048
```

## Common Patterns

### Pattern 1: Basic Completion

```python
class MyWorker(BaseRolloutWorker):
    def rollout(self, task):
        prompt = task["prompt"]
        result = self._inference_client.completion(prompt)

        trajectory = self._create_trajectory(result, task)
        return self._put_trajectory(trajectory)
```

### Pattern 2: Multi-step Processing

```python
class MyWorker(BaseRolloutWorker):
    def rollout(self, task):
        # Step 1: Generate reasoning
        reasoning_prompt = self._format_reasoning_prompt(task)
        reasoning = self._inference_client.completion(reasoning_prompt)

        # Step 2: Generate answer
        answer_prompt = self._format_answer_prompt(task, reasoning)
        answer = self._inference_client.completion(answer_prompt)

        # Step 3: Evaluate and create trajectory
        trajectory = self._create_trajectory(answer, task)
        return self._put_trajectory(trajectory)
```

### Pattern 3: Error Handling

```python
class MyWorker(BaseRolloutWorker):
    def rollout(self, task):
        try:
            result = self._process_task(task)
            trajectory = self._create_trajectory(result, task)
            return self._put_trajectory(trajectory)
        except ValueError as e:
            logger.error(f"Invalid task: {e}")
            return None  # Processing failed
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            # Try to return task for retry
            self._put_rollout_task(task)
            return None
```

## Threading Considerations

**Thread safety:**
- Each worker runs in its own thread
- Module references are read-only after initialization
- Activity tracker handles thread-safe tracking
- Inference client handles concurrent requests

**Best practices:**
- Don't modify shared state in `rollout()`
- Don't block indefinitely in `rollout()`
- Use activity tracker for long operations

## Performance Tips

1. **Keep rollout() fast**: Minimize non-essential work
2. **Batch when possible**: Some inference services support batching
3. **Handle re-rollout efficiently**: Don't recompute expensive operations
4. **Use async patterns**: For I/O-heavy operations

## Debugging

### Logging

```python
import logging
logger = logging.getLogger(__name__)

class MyWorker(BaseRolloutWorker):
    def rollout(self, task):
        logger.debug(f"Processing task: {task.get('task_id')}")
        # ... processing
        logger.info(f"Generated trajectory with reward: {reward}")
```

### Activity Tracking

The main loop automatically tracks activity:

```python
with self._activity_tracker.track(self._module_name, "rollout"):
    result = self.rollout(task)
```

Check worker activity:

```python
# In controller or monitoring
if activity_tracker.is_rollout_worker_quiescent():
    print("All workers idle")
```

## Common Issues

### Issue: Inference client not initialized

**Symptoms:** `AttributeError: 'NoneType' object has no attribute 'completion'`

**Solution:** Ensure `need_llm_inference=true` in config

### Issue: Re-rollout loop

**Symptoms:** Same task repeatedly returns "re-rollout"

**Cause:** Weight sync taking too long or stuck

**Solution:** Check weight sync controller status, verify sync mode configuration

### Issue: Tasks not appearing

**Symptoms:** Worker idle, but data loader has data

**Solution:** Check data loader `can_return_item()` and `is_finished()` methods

## Next Steps

- [NexAU Rollout Worker](./nexau-rollout-worker.md) - Agent framework integration
- [Custom Workers](./custom-workers.md) - Creating task-specific workers
- [Evaluators](./evaluators.md) - Implementing evaluation logic

## Related Documentation

- [Data Types](../02-core-architecture/data-types.md) - Trajectory structure
- [Activity Tracking](../02-core-architecture/activity-tracking.md) - Monitoring details
- [Weight Synchronization](../09-features/weight-synchronization.md) - Sync coordination
