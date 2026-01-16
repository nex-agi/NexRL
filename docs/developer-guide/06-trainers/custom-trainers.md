# Custom Trainers

This guide explains how to create custom trainers to implement new RL algorithms or training strategies in NexRL.

## When to Create a Custom Trainer

Create a custom trainer when you need to:

- Implement a new RL algorithm (PPO, A2C, DPO, etc.)
- Modify advantage computation logic
- Add custom reward shaping
- Implement specialized batch processing
- Integrate a novel training technique

## Choosing a Base Class

Select the appropriate base class based on your backend:

| Base Class | When to Use |
|------------|-------------|
| `SelfHostedTrainer` | NexTrainer backend, need full control over batching |
| `RemoteApiTrainer` | Tinker/Weaver backend, service handles batching |
| `BaseTrainer` | Completely custom training loop needed |

**Recommendation:** Extend `SelfHostedTrainer` or `RemoteApiTrainer` for most use cases. They provide infrastructure for checkpointing, metrics, and service communication.

## Pattern 1: Custom Self-Hosted Trainer

### Step 1: Create Trainer Class

Create a new file in your recipe directory:

```python
# recipe/my_task/my_trainer.py
from nexrl.trainer import SelfHostedTrainer
from nexrl.nexrl_types import Batch
import torch

class MyCustomTrainer(SelfHostedTrainer):
    """Custom trainer implementing MyAlgorithm."""

    def __init__(self, config):
        super().__init__(config)

        # Algorithm-specific initialization
        self._my_param = config.algorithm.get("my_param", 0.5)
        self._use_custom_advantage = config.algorithm.get("use_custom_advantage", True)

    def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
        """Implement custom algorithm logic."""
        metrics = {}

        # 1. Log rollout metrics
        self._log_rollout_metrics(batch)

        # 2. Remove redundant padding (optional but recommended)
        batch = Batch.remove_redundant_left_padding(
            batch,
            pad_token_id=self._pad_token_id,
            anchor_field="input_ids",
            fields=["input_ids", "attention_mask", "position_ids", "loss_mask"]
        )

        # 3. Compute rewards
        rewards = self._compute_custom_rewards(batch)
        batch.values["token_level_scores"] = rewards

        # 4. Compute advantages (custom algorithm)
        if self._use_custom_advantage:
            batch = self._compute_custom_advantages(batch)
            metrics["advantage_mean"] = float(torch.mean(batch.values["advantages"]))

        # 5. Add any other custom fields
        batch.values["my_custom_field"] = self._compute_custom_field(batch)

        return batch, metrics

    def _compute_custom_advantages(self, batch: Batch) -> Batch:
        """Implement your advantage computation logic."""
        # Example: simple per-trajectory advantage
        rewards = batch.metadata["rewards"]
        advantages = []

        for reward in rewards:
            # Your custom advantage calculation
            advantage = reward * self._my_param
            advantages.append(advantage)

        # Store advantages in batch
        batch.metadata["advantages"] = advantages
        return batch

    def _compute_custom_rewards(self, batch: Batch) -> torch.Tensor:
        """Convert scalar rewards to token-level rewards."""
        # Similar to SelfHostedGrpoTrainer._reward_fn()
        rewards = batch.metadata["rewards"]
        loss_mask = batch.values["loss_mask"]

        reward_tensor = torch.zeros_like(loss_mask, dtype=torch.float32)

        for i, reward in enumerate(rewards):
            # Assign reward to last valid token
            valid_positions = torch.where(loss_mask[i] == 1)[0]
            if len(valid_positions) > 0:
                last_pos = valid_positions[-1]
                reward_tensor[i, last_pos] = reward

        return reward_tensor

    def _compute_custom_field(self, batch: Batch) -> torch.Tensor:
        """Add any custom processing."""
        # Your custom logic here
        return torch.zeros(len(batch))
```

### Step 2: Register Trainer

Add to recipe configuration:

```yaml
# recipe/my_task/my_task.yaml
trainer:
  # Custom trainer registration
  custom_trainer_module_path: "recipe/my_task/my_trainer.py"
  custom_trainer_class_name: "MyCustomTrainer"

  # Standard trainer config
  total_train_steps: 100
  train_service:
    backend: "self-hosted"
    world_size: 8

  # Your custom algorithm config
  algorithm:
    my_param: 0.5
    use_custom_advantage: true
```

### Step 3: Test Your Trainer

```bash
# Run training with custom trainer
nexrl -m self-hosted -c recipe/my_task/my_task.yaml --run-nexrl
```

## Pattern 2: Custom Remote API Trainer

### Step 1: Create Trainer Class

```python
# recipe/my_task/my_remote_trainer.py
from nexrl.trainer import RemoteApiTrainer
from nexrl.nexrl_types import Trajectory
from typing import Any

class MyRemoteTrainer(RemoteApiTrainer):
    """Custom trainer for remote API with custom algorithm."""

    def __init__(self, config):
        super().__init__(config)

        # Algorithm parameters
        self._advantage_weight = config.algorithm.get("advantage_weight", 1.0)

    def _prepare_trajectories(
        self, trajectories: list[Trajectory], metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """Add custom fields to trajectories."""

        # 1. Group trajectories by run_id (if needed)
        groups = self._group_by_run_id(trajectories)

        # 2. Compute custom advantages
        for run_id, group in groups.items():
            # Your custom grouping/advantage logic
            group_advantages = self._compute_group_advantages(group)

            for traj, adv in zip(group, group_advantages):
                traj.advantage = adv * self._advantage_weight

        # 3. Log metrics
        avg_advantage = sum(t.advantage for t in trajectories) / len(trajectories)
        metrics["custom/advantage_mean"] = avg_advantage

        return trajectories

    def _group_by_run_id(self, trajectories: list[Trajectory]) -> dict:
        """Group trajectories by run_id."""
        groups = {}
        for traj in trajectories:
            run_id = traj.get("run_id", "default")
            if run_id not in groups:
                groups[run_id] = []
            groups[run_id].append(traj)
        return groups

    def _compute_group_advantages(self, group: list[Trajectory]) -> list[float]:
        """Compute advantages for a group of trajectories."""
        # Your custom advantage logic
        rewards = [traj.reward for traj in group]
        mean_reward = sum(rewards) / len(rewards)

        # Simple advantage: reward - mean
        advantages = [r - mean_reward for r in rewards]
        return advantages
```

### Step 2: Register and Configure

```yaml
# recipe/my_task/my_task.yaml
trainer:
  custom_trainer_module_path: "recipe/my_task/my_remote_trainer.py"
  custom_trainer_class_name: "MyRemoteTrainer"

  total_train_steps: 100
  train_service:
    backend: "tinker"
    url: "http://tinker-service:8000"
    config:
      loss_fn: "importance_sampling"
      learning_rate: 2e-6

  algorithm:
    advantage_weight: 1.0
```

## Pattern 3: Algorithm Modules

For complex algorithms, separate the algorithm logic from the trainer:

### Step 1: Create Algorithm Module

```python
# recipe/my_task/my_algorithm.py
import torch
from nexrl.nexrl_types import Batch

def compute_my_advantages(
    batch: Batch,
    discount_factor: float = 0.99,
    gae_lambda: float = 0.95
) -> Batch:
    """
    Compute advantages using custom algorithm (e.g., GAE).

    Args:
        batch: Batch with rewards and values
        discount_factor: Discount factor gamma
        gae_lambda: GAE lambda parameter

    Returns:
        Batch with advantages added
    """
    rewards = batch.metadata["rewards"]
    # Your algorithm implementation
    # ...

    advantages = []  # Computed advantages
    batch.metadata["advantages"] = advantages
    return batch

def compute_custom_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2
) -> torch.Tensor:
    """Compute custom loss (e.g., PPO clipped loss)."""
    # Your loss computation
    pass
```

### Step 2: Use in Trainer

```python
# recipe/my_task/my_trainer.py
from nexrl.trainer import SelfHostedTrainer
from nexrl.nexrl_types import Batch
from .my_algorithm import compute_my_advantages

class MyTrainer(SelfHostedTrainer):
    def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
        metrics = {}

        # Use algorithm module
        batch = compute_my_advantages(
            batch,
            discount_factor=self._config.algorithm.discount_factor,
            gae_lambda=self._config.algorithm.gae_lambda
        )

        metrics["advantage_mean"] = float(torch.mean(batch.metadata["advantages"]))
        return batch, metrics
```

## Common Patterns

### Pattern: KL Penalty

Add KL divergence penalty to rewards:

```python
def _apply_kl_penalty(self, batch: Batch, beta: float) -> Batch:
    """Apply KL penalty to rewards."""
    # Compute KL divergence
    log_probs = batch.values["log_probs"]
    ref_log_probs = batch.values["ref_log_probs"]
    kl = log_probs - ref_log_probs

    # Apply penalty
    rewards = batch.values["token_level_scores"]
    penalized_rewards = rewards - beta * kl
    batch.values["token_level_scores"] = penalized_rewards

    return batch
```

### Pattern: Value Function

Add value network for actor-critic algorithms:

```python
def _compute_values(self, batch: Batch) -> torch.Tensor:
    """Compute value estimates using value network."""
    # Call inference service to get value estimates
    values = self._value_network.predict(batch.values["input_ids"])
    return values
```

### Pattern: Multi-Step Returns

Compute n-step returns:

```python
def _compute_nstep_returns(
    self, rewards: list[float], n: int, gamma: float
) -> list[float]:
    """Compute n-step returns."""
    returns = []
    for i in range(len(rewards)):
        n_step_return = 0
        for j in range(min(n, len(rewards) - i)):
            n_step_return += (gamma ** j) * rewards[i + j]
        returns.append(n_step_return)
    return returns
```

## Best Practices

### 1. Preserve Required Fields

Ensure your trainer maintains required batch fields:

```python
# Required for NexTrainer
- input_ids
- attention_mask
- position_ids
- loss_mask
- token_level_scores (rewards)
- advantages (for policy gradient methods)
```

### 2. Log Metrics

Log algorithm-specific metrics for monitoring:

```python
def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
    metrics = {}

    # Compute things
    advantages = self._compute_advantages(batch)

    # Log statistics
    metrics["algorithm/advantage_mean"] = float(advantages.mean())
    metrics["algorithm/advantage_std"] = float(advantages.std())
    metrics["algorithm/advantage_max"] = float(advantages.max())

    return batch, metrics
```

### 3. Handle Edge Cases

```python
def _compute_advantages(self, batch: Batch) -> torch.Tensor:
    """Compute advantages with proper handling."""
    rewards = batch.metadata["rewards"]

    # Handle empty batch
    if len(rewards) == 0:
        return torch.zeros(0)

    # Handle single sample (no std)
    if len(rewards) == 1:
        return torch.zeros(1)

    # Normal computation
    mean = torch.mean(rewards)
    std = torch.std(rewards)
    advantages = (rewards - mean) / (std + 1e-8)  # Avoid division by zero

    return advantages
```

### 4. Use Existing Utilities

Leverage NexRL utilities:

```python
from nexrl.utils.torch_functional import padding_data, compute_position_id_with_mask
from nexrl.utils.logging_utils import log_rollout_metrics, log_grpo_metrics
from nexrl.algorithm.core_algos import compute_grpo_advantage_for_trajectories

# Use in your trainer
batch = padding_data(batch, pad_token_id=self._pad_token_id)
log_rollout_metrics(trajectories, metrics)
```

## Testing Your Custom Trainer

### 1. Unit Test

Create a test file:

```python
# recipe/my_task/test_my_trainer.py
import pytest
from omegaconf import OmegaConf
from nexrl.nexrl_types import Batch
from .my_trainer import MyCustomTrainer

def test_custom_trainer_prepare_batch():
    config = OmegaConf.create({
        "algorithm": {"my_param": 0.5},
        "train_service": {"world_size": 1, "backend": "self-hosted", "url": "http://localhost"}
    })

    trainer = MyCustomTrainer(config)

    # Create dummy batch
    batch = create_dummy_batch()

    # Test _prepare_batch
    prepared_batch, metrics = trainer._prepare_batch(batch)

    assert "advantages" in prepared_batch.values
    assert "advantage_mean" in metrics
```

### 2. Integration Test

Test with a small training run:

```bash
# Use small config for testing
nexrl -m self-hosted -c recipe/my_task/test_config.yaml --run-nexrl
```

```yaml
# recipe/my_task/test_config.yaml
trainer:
  total_train_steps: 2  # Just 2 steps for testing
  custom_trainer_module_path: "recipe/my_task/my_trainer.py"
  custom_trainer_class_name: "MyCustomTrainer"
  # ... rest of config
```

## Troubleshooting

### Common Issues

1. **Missing required fields in batch:**
   - Check that you're not removing required fields (input_ids, attention_mask, etc.)
   - Use `batch.values.keys()` to inspect available fields

2. **Dimension mismatches:**
   - Ensure tensor shapes match: `(batch_size, seq_len)`
   - Check that padding is applied correctly

3. **Metrics not appearing:**
   - Verify metric keys don't have typos
   - Check that metrics are floats, not tensors

4. **Custom trainer not loaded:**
   - Verify file path is relative to workspace root
   - Check class name matches exactly
   - Ensure file has no import errors

## Examples from Existing Trainers

### From SelfHostedGrpoTrainer

Key patterns to reuse:

```python
# Padding removal
batch = Batch.remove_redundant_left_padding(batch, ...)
batch = Batch.remove_redundant_right_padding(batch, ...)

# Token-level rewards
reward_tensor = self._reward_fn(batch)

# Old log probabilities
old_log_probs = self._compute_old_log_probs(batch)

# Advantage computation
batch = self._compute_advantage(batch)
```

### From RemoteApiGrpoTrainer

Key patterns for remote trainers:

```python
# Use existing GRPO algorithm
from nexrl.algorithm.core_algos import compute_grpo_advantage_for_trajectories

trajectories = compute_grpo_advantage_for_trajectories(
    trajectories, logger=logger, use_run_ids=True
)

# Log metrics
from nexrl.utils.logging_utils import log_grpo_metrics
log_grpo_metrics(trajectories, metrics)
```

## Related Documentation

- [Overview](./overview.md) - Trainer architecture
- [Self-Hosted Trainers](./self-hosted-trainers.md) - SelfHostedTrainer details
- [Remote API Trainers](./remote-api-trainers.md) - RemoteApiTrainer details
- [GRPO Algorithm](../07-algorithms/grpo.md) - GRPO implementation reference
- [Data Types](../02-core-architecture/data-types.md) - Trajectory and Batch types
