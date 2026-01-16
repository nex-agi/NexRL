# Self-Hosted Trainers

Self-hosted trainers are used with NexTrainer backend for full control over the training algorithm. These trainers handle batch preparation, tokenization, and algorithm-specific logic.

## Architecture

```python
BaseTrainer
    ↓
SelfHostedTrainer (abstract)
    ↓
SelfHostedGrpoTrainer
```

## SelfHostedTrainer

Base class for self-hosted training backends. Located in `nexrl/trainer/self_hosted_trainer.py`.

### Purpose

Provides core training loop infrastructure for self-hosted backends:
1. Trajectory tokenization and padding
2. Batch conversion and tensor preparation
3. Training service communication
4. Checkpointing and metrics tracking

### Constructor

```python
def __init__(self, config: DictConfig)
```

**Key Initialization:**
- Creates train service client (NexTrainer backend)
- Initializes tokenizer from model path
- Sets up max prompt/response lengths
- Configures world size for distributed training

### Main Training Flow

The `train()` method implements the complete training pipeline:

```python
def train(self, trajectories: list[Trajectory]) -> dict:
    # 1. Process trajectories (padding, tokenization)
    trajectories = self._process_trajectories(trajectories)

    # 2. Convert to batch
    batch = Batch.from_trajectories(trajectories, model_tag=self._model_tag)
    batch = batch.pad_to_world_size(world_size=self.world_size)

    # 3. Prepare batch (algorithm-specific, implemented by subclass)
    batch, preparation_metrics = self._prepare_batch(batch)

    # 4. Convert to NexTrainer format
    nextrainer_batch = batch.to_nextrainer_batch()

    # 5. Execute training step
    train_result = self._train_service_client.train_step(nextrainer_batch)

    # 6. Handle checkpointing if needed
    # ...

    return train_metrics
```

### Key Methods

#### _prepare_batch() (Abstract)

```python
@abstractmethod
def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]
```

Algorithm-specific batch preparation. Subclasses must implement this.

**Parameters:**
- `batch`: Batch with tokenized trajectory data

**Returns:** Tuple of (prepared_batch, metrics_dict)

**Purpose:** Add algorithm-specific fields (advantages, rewards, etc.) to the batch.

#### _process_trajectories()

```python
def _process_trajectories(self, trajectories: list[Trajectory]) -> list[Trajectory]
```

Converts raw trajectories into tensors with proper padding.

**Process:**
1. Separates prompt and response tokens using `loss_mask`
2. Applies left padding to prompts, right padding to responses
3. Computes position IDs from attention masks
4. Creates tensor fields: `input_ids`, `attention_mask`, `position_ids`, `loss_mask`, `prompts`, `responses`

**Key Implementation Detail:**
- `loss_mask` determines prompt vs response: 0 for prompt tokens, 1 for response tokens
- Handles variable-length prompts and responses
- Uses tokenizer's pad token for padding

## SelfHostedGrpoTrainer

Implements GRPO (Group Relative Policy Optimization) algorithm for self-hosted training. Located in `nexrl/trainer/self_hosted_grpo_trainer.py`.

### Constructor

```python
def __init__(self, config: DictConfig)
```

**Additional Configuration:**
- `do_old_log_prob_compute`: Whether to recompute old log probabilities (default: True)
- `use_kl_in_reward`: Whether to apply KL penalty to rewards (default: False)
- Initializes KL controller (Adaptive or Fixed)

### _prepare_batch() Implementation

Implements the abstract method with GRPO-specific processing:

```python
def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
    metrics = {}

    # 1. Log rollout metrics (reward distribution, etc.)
    self._log_rollout_metrics(batch)

    # 2. Remove redundant padding for efficiency
    batch = Batch.remove_redundant_left_padding(batch, ...)
    batch = Batch.remove_redundant_right_padding(batch, ...)

    # 3. Recompute old log probabilities (if enabled)
    if self._do_old_log_prob_compute:
        old_log_probs = self._compute_old_log_probs(batch)
        batch.values["old_log_probs"] = old_log_probs

    # 4. Convert scalar rewards to token-level rewards
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

### GRPO-Specific Methods

#### _compute_advantage()

```python
def _compute_advantage(self, batch: Batch) -> Batch
```

Computes group-relative advantages using the GRPO algorithm.

**Process:**
1. Extract group IDs (`uid` or `group_id`) and run IDs
2. Call `core_algos.compute_grpo_outcome_advantage()`:
   - Groups trajectories by (group_id, run_id)
   - Computes group mean and std of rewards
   - Normalizes: `advantage = (reward - mean) / (std + 1e-8)`
3. Stores advantages and returns in batch metadata

#### _reward_fn()

```python
def _reward_fn(self, batch: Batch) -> torch.Tensor
```

Converts scalar rewards to token-level reward tensors.

**Process:**
1. Creates zero tensor matching response shape
2. For each trajectory, assigns reward to last valid token
3. Uses `loss_mask` to identify valid response tokens

#### _compute_old_log_probs()

```python
def _compute_old_log_probs(self, batch: Batch) -> torch.Tensor
```

Recomputes log probabilities using the current policy (before training).

**Purpose:** Used for importance sampling and KL divergence computation.

#### _apply_kl_penalty()

```python
def _apply_kl_penalty(self, batch, kl_ctrl, kl_penalty) -> tuple[Batch, dict]
```

Applies KL divergence penalty to rewards.

**Process:**
1. Computes KL between current and reference policy
2. Scales by KL controller coefficient
3. Modifies rewards: `rewards = scores - beta * kld`
4. Updates KL controller based on current KL

## Configuration

Example configuration for self-hosted GRPO trainer:

```yaml
trainer:
  total_train_steps: 100
  max_prompt_length: 4096
  max_response_length: 2048

  train_service:
    backend: "self-hosted"
    url: "http://localhost:5000"
    world_size: 8  # Number of GPUs

  algorithm:
    # Trainer is auto-selected as SelfHostedGrpoTrainer

    # GRPO parameters
    do_old_log_prob_compute: true
    use_kl_in_reward: false

    # KL controller (if use_kl_in_reward: true)
    critic:
      kl_ctrl:
        type: "adaptive"  # or "fixed"
        kl_reward_coef: 0.1
        target_kl: 6.0
        horizon: 10000

    # Inference service for tokenizer
    inference_service:
      model: "meta-llama/Llama-3.1-8B-Instruct"
      tokenizer: "meta-llama/Llama-3.1-8B-Instruct"
```

## Key Features

### Batch Preparation Pipeline

The `_prepare_batch()` pattern allows clean separation of algorithm logic:

```python
# In SelfHostedGrpoTrainer
def _prepare_batch(self, batch):
    # GRPO-specific processing
    batch = self._compute_advantage(batch)
    return batch, metrics

# In a custom trainer (e.g., PPO)
def _prepare_batch(self, batch):
    # PPO-specific processing
    batch = self._compute_ppo_objectives(batch)
    return batch, metrics
```

### Padding Removal

GRPO trainer removes redundant padding to improve training efficiency:

- **Left padding removal**: Removes common prefix padding from prompts
- **Right padding removal**: Removes common suffix padding from responses

This is done after tokenization but before sending to the train service.

### Checkpointing

Self-hosted trainer handles checkpoint saving in a background thread:

```python
# Automatic checkpoint saving based on config
if self._train_step % checkpoint_interval == 0:
    self._save_checkpoint_async()
```

## Usage Example

The trainer is automatically instantiated by the controller. For custom tasks, you typically only need to configure it in YAML:

```yaml
# recipe/my_task/my_task.yaml
trainer:
  total_train_steps: 100
  train_service:
    backend: "self-hosted"
    world_size: 8
```

For advanced use cases, you can extend `SelfHostedTrainer` to implement a custom algorithm:

```python
from nexrl.trainer import SelfHostedTrainer
from nexrl.nexrl_types import Batch

class MyCustomTrainer(SelfHostedTrainer):
    def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
        metrics = {}

        # Your custom algorithm logic here
        # - Compute rewards
        # - Compute advantages
        # - Add any custom fields to batch

        return batch, metrics
```

See [Custom Trainers](./custom-trainers.md) for more details.

## Related Documentation

- [Overview](./overview.md) - Trainer architecture overview
- [GRPO Algorithm](../07-algorithms/grpo.md) - GRPO implementation details
- [Training Service](../08-services/training-service.md) - NexTrainer backend integration
- [Custom Trainers](./custom-trainers.md) - Creating custom algorithm trainers
