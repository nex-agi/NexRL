# Rollout Workers Overview

Rollout workers are the components responsible for generating trajectories by executing tasks and interacting with LLM services. They form the data generation pipeline that feeds the training process.

## Architecture

```
DataLoader → RolloutWorker → TrajectoryPool
                ↓
            Evaluator
```

Rollout workers:
1. Fetch tasks from the data loader
2. Execute the task (interact with LLM/agent)
3. Evaluate the result
4. Create trajectories with tokens, rewards, and metadata
5. Submit trajectories to the trajectory pool

## Worker Hierarchy

```
NexRLModule
    ↓
BaseRolloutWorker (abstract)
    ↓
    ├── SimpleRolloutWorker (basic LLM completion)
    ├── AgentRolloutWorker (multi-turn with tools)
    └── BaseNexAURolloutWorker (agent framework)
            ↓
        Custom Workers (task-specific)
```

## When to Use Each Worker

### SimpleRolloutWorker
- **Use case**: Basic single-turn LLM completion tasks
- **Features**: Prompt → LLM → Response
- **Example**: Question answering, text generation

### AgentRolloutWorker
- **Use case**: Multi-turn conversations with tool calling
- **Features**: Message history, tool calls, stateful interactions
- **Example**: Conversational agents, complex reasoning

### BaseNexAURolloutWorker
- **Use case**: Complex agent tasks with NexAU framework
- **Features**: Agent framework integration, trace processing, evaluation
- **Example**: Web search, code generation, multi-step tasks

### Custom Workers
- **Use case**: Task-specific behavior
- **Features**: Custom query formatting, specialized evaluation, domain logic
- **Example**: News classification, specialized domains

## Core Workflow

### 1. Main Loop

```python
while not stopped:
    task = get_next_task()  # From data loader

    if validation_mode:
        trajectory = process_validation_task(task)
        submit_to_validator(trajectory)
    else:
        trajectory = process_training_task(task)
        result = submit_to_trajectory_pool(trajectory)

        if result == "re-rollout":
            # Weight sync in progress, retry same task
            continue
```

### 2. Rollout Execution

Each worker implements the `rollout()` method:

```python
def rollout(self, task: dict[str, Any]) -> str | None:
    """
    Execute single rollout operation.

    Returns:
        - "success": Trajectory submitted successfully
        - "fail": Failed to submit
        - "re-rollout": Weight sync in progress, retry
        - None: Processing failed before trajectory creation
    """
```

### 3. Trajectory Creation

Workers create trajectories containing:

```python
Trajectory(
    tokens=[...],           # Token IDs (prompt + response)
    loss_mask=[...],        # 0 for prompt, 1 for response
    reward=float,           # Primary RL signal (0.0-1.0)
    is_val=bool,           # Validation or training
    extra_fields={          # Additional metadata
        "ground_truth": str,
        "group_id": str,
        "run_id": int,
        "task_id": int,
        # ... task-specific fields
    }
)
```

## Key Concepts

### Activity Tracking

Workers track their activity for monitoring:

```python
with self._activity_tracker.track(self._module_name, "rollout"):
    result = self.rollout(task)
```

This enables:
- System health monitoring
- Quiescence detection (all workers idle)
- Performance profiling

### Weight Synchronization Coordination

Workers coordinate with weight sync:

```python
result = self._put_trajectory(trajectory)
if result == "re-rollout":
    # Weight sync in progress
    # Keep current task and retry
    continue
```

**Sync modes:**
- **sync**: All workers block until sync complete
- **fully-async**: No blocking, opportunistic sync
- **batch-async**: Block if staleness threshold exceeded

### Validation Mode

Workers support both training and validation:

```python
def begin_validate(self):
    """Switch to validation mode"""
    self._is_running_validate = True

def end_validate(self):
    """Switch back to training mode"""
    self._is_running_validate = False
```

When in validation mode:
- Tasks come from validation data loader
- Trajectories go to validator (not trajectory pool)
- Validation metrics computed after collection

### Inference Service Integration

Workers access LLM services through inference client:

```python
# Set during initialization
self._inference_client = create_inference_service_client(
    backend=config.inference_service.backend,
    config=config
)

# Use in rollout
result = self._inference_client.completion(
    prompt=prompt,
    temperature=0.7,
    max_tokens=512
)
```

## Module References

Workers need references to other components:

```python
def set_module_references(
    self,
    trajectory_pool: TrajectoryPool,
    dataloader: BaseDataLoader,
    weight_sync_controller: WeightSyncController,
    validate_dataloader: BaseDataLoader,
    validator: Validator,
):
    """Set module references before run()"""
```

These are set by the controller during initialization.

## Configuration

Basic worker configuration:

```yaml
rollout_worker:
  type: "nexau"  # Worker type
  num_workers: 4  # Number of parallel workers

  # Inference service
  inference_service:
    backend: "vllm"
    url: "http://localhost:8000/v1"
    model_tag: "default"
    api_key: "EMPTY"

  # Custom worker (optional)
  custom_rollout_worker_module_path: "recipe/my_task/my_worker.py"
  custom_rollout_worker_class_name: "MyWorker"

  # Worker-specific config
  temperature: 0.7
  max_tokens: 2048
  need_llm_inference: true
```

## Common Patterns

### Basic Completion Pattern

```python
class MyWorker(BaseRolloutWorker):
    def rollout(self, task):
        # Get prompt
        prompt = task["prompt"]

        # Call LLM
        result = self._inference_client.completion(prompt)

        # Create trajectory
        trajectory = Trajectory(
            tokens=result["tokens"],
            loss_mask=result["loss_mask"],
            reward=self._evaluate(result, task),
            extra_fields={"ground_truth": task["answer"]}
        )

        # Submit
        return self._put_trajectory(trajectory)
```

### Agent Pattern with Evaluation

```python
class MyAgentWorker(BaseNexAURolloutWorker):
    def run_agent(self, task):
        # Format query
        query = self.format_query(task)

        # Load and run agent
        agent, client_provider = self.load_agent_from_config(
            custom_llm_client_provider=lambda: self._inference_client
        )
        response = agent.run(query, custom_llm_client_provider=client_provider)

        # Evaluate
        eval_result = self.evaluator.evaluate(task, evaluation_target)

        return agent_output, eval_result
```

## Error Handling

Workers should handle errors gracefully:

```python
def rollout(self, task):
    try:
        # Process task
        result = self._process(task)
        return self._put_trajectory(result)
    except Exception as e:
        logger.error(f"Rollout failed: {e}")
        # Error is auto-reported via activity tracker
        return None  # Indicates failure
```

Errors are automatically reported through the activity tracker when using the `track()` context manager.

## Performance Considerations

### Parallel Workers

Multiple workers run in parallel:

```yaml
rollout_worker:
  num_workers: 8  # 8 parallel workers
```

**Guidelines:**
- More workers = higher throughput
- Balance with GPU memory (if using local inference)
- Consider inference service capacity

### Batching

Workers process tasks individually, but batching happens in:
1. **Inference service**: Batches multiple requests
2. **Trajectory pool**: Batches trajectories for training

### Resource Management

In Ray mode, workers can be co-located:

```python
resource_pool_mapping:
  rollout_worker_pool:
    - [rollout_worker, 4]  # 4 workers in this pool
    - [weight_sync_controller, 1]  # Co-located with workers
```

## Next Steps

- [Base Rollout Worker](./base-rollout-worker.md) - Core worker interface and methods
- [NexAU Rollout Worker](./nexau-rollout-worker.md) - Agent framework integration
- [Custom Workers](./custom-workers.md) - Creating task-specific workers
- [Evaluators](./evaluators.md) - Implementing evaluation logic

## Related Documentation

- [Data Loader](../03-data-loader/data-loader.md) - Task source
- [Trajectory Pool](../04-trajectory-pool/trajectory-pool.md) - Trajectory destination
- [Activity Tracking](../02-core-architecture/activity-tracking.md) - Monitoring
- [Configuration Reference](../12-configuration-reference/rollout-config.md) - Full config options
