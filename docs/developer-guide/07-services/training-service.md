# Training Service

The training service provides the training backend for self-hosted trainers. NexRL uses a client-server architecture where trainers communicate with training workers via the TrainServiceClient.

## Architecture

```
TrainServiceClient (Abstract)
        ↓
    ┌────────────────────┐
    ↓                    ↓
HTTPTrainServiceClient  MockTrainServiceClient
  (NexTrainer backend)    (Testing only)
```

## Multi-Service Support

NexRL supports multiple train services for advanced training methods:
- **On-Policy Distillation (OPD)**: Student + Teacher services
- **Multi-model training**: Multiple student models

Each service is identified by:
- **Name**: Service name in configuration (e.g., `student_service`, `teacher_service`)
- **Role**: Service role (`actor`, `teacher`)
- **Identifier**: Unique identifier for the service group

## TrainServiceClient

Abstract base class for training service clients. Located in `nexrl/train_service_client.py`.

### Purpose

Provides a unified interface for:
- Worker initialization and configuration
- Model initialization
- Training step execution (forward + backward + optim)
- Log probability computation
- Checkpoint management

### Core Methods

#### initialize_worker()

```python
def initialize_worker(
    self,
    config_path: str | None = None,
    config_dict: dict | None = None,
    role: str = "actor",
    world_size: int | None = None,
    zmq_base_port: int | None = None,
    dispatch_mode: str | None = None,
) -> dict
```

Initialize training workers with configuration.

**Parameters:**
- `config_path`: Path to YAML config file
- `config_dict`: Dictionary config (overrides config_path)
- `role`: Worker role ("actor", "critic", "reward")
- `world_size`: Number of workers (required for new groups)
- `zmq_base_port`: Base port for ZMQ communication
- `dispatch_mode`: Dispatch mode ("broadcast", etc.)

#### update_actor()

```python
def update_actor(self, batch: dict) -> dict
```

Execute a complete training step (forward + backward + optimizer step).

**Parameters:**
- `batch`: Data batch in NexTrainer format

**Returns:** Dictionary with metrics and meta_info

#### compute_log_prob()

```python
def compute_log_prob(self, batch: dict) -> dict
```

Compute log probabilities for trajectories (used for old log prob computation).

**Parameters:**
- `batch`: Data batch with input_ids

**Returns:** Dictionary with log probabilities

#### save_checkpoint() / load_checkpoint()

```python
def save_checkpoint(
    self,
    local_path: str,
    hdfs_path: str | None = None,
    global_step: int = 0,
    saved_fully_shared_ckpt: bool = True,
    save_weight_only: bool = False,
    remove_previous_ckpt: bool = True,
) -> dict

def load_checkpoint(
    self, path: str, del_local_after_load: bool = True, load_weight_only: bool = False
) -> dict
```

Save/load model checkpoints.

## HTTPTrainServiceClient

HTTP client for NexTrainer backend. Located in `nexrl/train_service_backend/api/client.py`.

### Constructor

```python
def __init__(self, base_url: str = "http://localhost:8000", identifier: str | None = None)
```

**Parameters:**
- `base_url`: URL of the training service
- `identifier`: Worker group identifier (immutable once set)

### Worker Group Management

Each client instance is bound to a specific worker group:

```python
# Single worker group
client = HTTPTrainServiceClient("http://localhost:8000", identifier="nexrl0")
client.save_checkpoint("/path/to/ckpt")  # Operates on "nexrl0"

# Multiple worker groups - use separate clients
client1 = HTTPTrainServiceClient("http://localhost:8000", identifier="group1")
client2 = HTTPTrainServiceClient("http://localhost:8000", identifier="group2")
```

### Tensor Serialization

The client handles tensor/numpy array serialization for HTTP transport:

```python
# Converts tensors to base64-encoded format
def _tensor_to_data(self, tensor: torch.Tensor) -> dict:
    return tensor_to_data(tensor)

# Converts back from serialized format
def _data_to_tensor(self, data: dict) -> torch.Tensor:
    return data_to_tensor(data)
```

### Training Step Execution

```python
def update_actor(self, batch: dict) -> dict:
    """Execute training step"""
    # Prepare request with tensor serialization
    request_data = self._prepare_data_proto_request(batch)

    # Split large requests if needed
    chunks = split_for_requests(request_data, max_size=100*1024*1024)

    # Send to training service
    for chunk in chunks:
        response = self.session.post(
            f"{self.base_url}/update_actor",
            json=chunk,
            params={"identifier": self.identifier},
            timeout=self.request_timeout
        )

    # Process response
    result = self._process_data_proto_response(response.json())
    return result
```

### Health Monitoring

```python
def health_check() -> dict:
    """Check service health"""
    response = self.session.get(f"{self.base_url}/health")
    return response.json()

def worker_info() -> dict:
    """Get worker information for this client's worker group"""
    response = self.session.get(
        f"{self.base_url}/worker_info",
        params={"identifier": self.identifier}
    )
    return response.json()
```

## NexTrainer Batch Format

The training service expects batches in a specific format:

```python
batch = {
    # Tensor values (serialized)
    "tensors": {
        "input_ids": [...],           # [batch_size, seq_len]
        "attention_mask": [...],      # [batch_size, seq_len]
        "position_ids": [...],        # [batch_size, seq_len]
        "loss_mask": [...],           # [batch_size, seq_len]
        "token_level_scores": [...],  # [batch_size, seq_len]
        "advantages": [...],          # [batch_size, seq_len] or [batch_size]
        "old_log_probs": [...],       # [batch_size, seq_len] (optional)
    },

    # Non-tensor values
    "values": {
        "model_tag": "default",
        # ... other non-tensor data
    },

    # Metadata
    "metadata": {
        "batch_size": 32,
        "global_token_num": [128, 256, ...],
        # ... other metadata
    }
}
```

This format is created by `Batch.to_nextrainer_batch()` in self-hosted trainers.

## Configuration

### Single Service (Standard Training)

```yaml
service:
  train_service:
    main_actor:
      identifier: "default"
      role: "actor"
      backend: http
      url: "http://localhost:8000"
      resource:
        world_size: 8
      actor:
        # Training hyperparameters
        model:
          path: "/path/to/model"
        optim:
          lr: 2e-6
```

### Multiple Services (OPD)

```yaml
service:
  train_service:
    student_service:
      identifier: "student"
      role: "actor"
      backend: http
      url: "http://localhost:8000"
      resource:
        world_size: 8
      actor:
        model:
          path: "/path/to/student"

    teacher_service:
      identifier: "teacher"
      role: "teacher"
      backend: http
      url: "http://localhost:8000"
      resource:
        world_size: 8
      actor:
        model:
          path: "/path/to/teacher"
```

## Client Creation

Trainers create clients using the factory function:

```python
from nexrl.utils.init_utils import create_train_service_client

client = create_train_service_client(
    backend="self-hosted",
    url="http://localhost:5000",
    identifier="nexrl0"
)
```

The factory selects the appropriate client implementation based on backend.

## Usage in Self-Hosted Trainers

Self-hosted trainers use the client for all training operations:

```python
class SelfHostedTrainer(BaseTrainer):
    def __init__(self, config):
        # Create train service client
        self._train_service_client = create_train_service_client(
            config.train_service.backend,
            config.train_service.url,
            config.train_service.get("identifier", None),
        )

    def initialize_workers(self):
        """Initialize training workers"""
        self._train_service_client.initialize_worker(
            config_dict=worker_config,
            world_size=self.world_size,
            role="actor"
        )

        # Wait for model initialization
        self._train_service_client.init_model()

    def train(self, trajectories):
        # Process trajectories and prepare batch
        batch = self._prepare_batch(batch)

        # Convert to NexTrainer format
        nextrainer_batch = batch.to_nextrainer_batch()

        # Execute training step
        train_result = self._train_service_client.update_actor(nextrainer_batch)

        return train_result
```

## Checkpoint Management

### Saving Checkpoints

```python
def _save_checkpoint(self, step: int):
    """Save checkpoint to local and HDFS"""
    local_path = f"/tmp/checkpoints/step_{step}"
    hdfs_path = f"hdfs://path/to/checkpoints/step_{step}"

    result = self._train_service_client.save_checkpoint(
        local_path=local_path,
        hdfs_path=hdfs_path,
        global_step=step,
        save_weight_only=False,  # Save full checkpoint
        remove_previous_ckpt=True  # Clean up old checkpoints
    )

    return result
```

### Loading Checkpoints

```python
def _load_checkpoint(self, path: str):
    """Load checkpoint from path"""
    result = self._train_service_client.load_checkpoint(
        path=path,
        load_weight_only=False,  # Load full checkpoint
        del_local_after_load=True  # Clean up after loading
    )

    return result
```

## Error Handling

The client includes retry logic and error handling:

```python
# Automatic retries for network errors
try:
    result = client.update_actor(batch)
except requests.exceptions.RequestException as e:
    logger.error(f"Training service request failed: {e}")
    # Retry or handle error

# Timeout handling
client.request_timeout = 200  # seconds
```

## Performance Considerations

### 1. Batch Splitting

Large batches are automatically split to avoid HTTP payload limits:

```python
# Automatically splits batches > 100MB
chunks = split_for_requests(request_data, max_size=100*1024*1024)
```

### 2. Tensor Serialization

Tensors are serialized using base64 encoding for HTTP transport. This adds overhead but ensures compatibility.

### 3. Worker Group Isolation

Use separate client instances for different worker groups to avoid contention:

```python
# Good: Separate clients
client_policy = HTTPTrainServiceClient(url, identifier="policy")
client_value = HTTPTrainServiceClient(url, identifier="value")

# Bad: Single client for multiple groups
client = HTTPTrainServiceClient(url)  # Undefined behavior with multiple groups
```

## MockTrainServiceClient

For testing without a training service backend:

```python
from nexrl.mock import MockTrainServiceClient

client = MockTrainServiceClient(url="mock://localhost")

# Returns dummy responses
result = client.update_actor(batch)
# result = {"metrics": {"loss": 0.5}, "meta_info": {}}
```

Located in `nexrl/mock/mock_train_service_client.py`.

## Related Documentation

- [Self-Hosted Trainers](../06-trainers/self-hosted-trainers.md) - Trainers using training service
- [Weight Synchronization](../08-features/weight-synchronization.md) - Weight sync with training service
- [Checkpointing](../08-features/checkpointing.md) - Checkpoint management
- [Configuration Reference](../11-configuration-reference/service-config.md) - Service configuration
