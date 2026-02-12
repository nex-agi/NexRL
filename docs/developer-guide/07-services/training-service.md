# Training Service

The training service provides distributed training for self-hosted trainers using a rank-0-centric architecture where the driver sends data to rank 0, which distributes it to all workers via fast NCCL communication.

## Architecture Overview

```
Driver (DirectZMQTrainServiceClient)
   │
   │ 1. Send data to rank 0
   ├─── ZMQ PUSH (full batch) ──▶ Rank 0 (PULL :6000)
   │                                  │
   │ 2. Wait for ACK               │
   │◀─── ZMQ PULL (ACK) ─────────────┤
   │                                  │
   │ 3. Broadcast execute (after ACK) │
   └─── HTTP POST /execute ──▶ API Server (PUB :5555)
                                     │
                                     ├──▶ All Workers (SUB)
                                     │         │
                                     │    4. NCCL Scatter ──▶ All Workers
                                     │    (respects DP/SP topology)
                                     │         │
                                     │    5. Workers Execute Training
                                     │         │
                                     │    6. NCCL Gather ──▶ Rank 0
                                     │         │
   ◀─── ZMQ PULL (result) ──────────┴─────────┘
   7. Return gathered result
```

**Key Benefits:**
- **No API Server OOM**: Training data never touches API server
- **ACK-Based Ordering**: Execute signal never arrives before data (prevents race conditions)
- **NCCL Performance**: Fast InfiniBand/NVLink between workers >> network ZMQ
- **SP/DP Aware**: Rank 0 handles device_mesh topology correctly
- **Simple Client**: No need for SP awareness in driver code

## Multi-Service Support

NexRL supports multiple train services for advanced training methods:
- **On-Policy Distillation (OPD)**: Student + Teacher services running simultaneously
- **Multi-model training**: Multiple student models with different configurations

Each service is identified by:
- **Name**: Service name in configuration (e.g., `student_service`, `teacher_service`)
- **Role**: Service role (`actor` for student, `teacher` for reference model)
- **Identifier**: Unique identifier for the worker group (e.g., `student`, `teacher`)

Multiple `DirectZMQTrainServiceClient` instances can connect to the same API server with different identifiers. Each identifier gets its own:
- Worker group with independent ranks
- ZMQ coordinator with dedicated ports
- Independent command and result queues

**Example OPD Setup:**
```yaml
service:
  train_service:
    student_service:
      identifier: "student"
      role: "actor"
      backend: "direct-zmq"
      url: "http://localhost:8000"
      resource:
        world_size: 8

    teacher_service:
      identifier: "teacher"
      role: "teacher"
      backend: "direct-zmq"
      url: "http://localhost:8000"
      resource:
        world_size: 8
```

## Complete Workflow

### 1. Launch API Server

```bash
python -m nexrl.train_service_backend.api.api_server --host 0.0.0.0 --port 8000
```

The API server manages worker groups, broadcasts commands, and forwards results.

### 2. Launch Workers

```bash
torchrun --nproc_per_node=8 \
    -m nexrl.train_service_backend.distributed.worker_process \
    --api-server-url http://localhost:8000 \
    --identifier actor-group \
    --backend fsdp
```

Each worker:
1. Registers with API server (rank 0 registers the group)
2. Binds ZMQ PULL socket on port `6000 + rank`
3. Reports its endpoint to API server via `/report_worker_endpoint`
4. Connects SUB socket to API server for commands
5. Connects PUSH socket to API server for results
6. Sends heartbeat every 5 seconds

### 3. Create Training Client

```python
from nexrl.utils.init_utils import create_train_service_client

client = create_train_service_client(
    backend="direct-zmq",
    url="http://localhost:8000",
    identifier="actor-group",
    config={
        "zmq_recv_timeout_ms": 600000,    # 10 min per worker
        "zmq_total_timeout_ms": 3600000,  # 60 min total
    }
)
```

### 4. Initialize Workers

```python
# Send worker configuration (HTTP to API server)
client.initialize_worker(
    config_dict=training_config,
    role="actor",
    world_size=8
)

# Load model to GPU (HTTP to API server, broadcast to workers)
client.init_model()
```

### 5. Training Loop

```python
# Prepare batch
batch = {
    "batch": {
        "input_ids": torch.tensor(...),      # [batch_size, seq_len]
        "attention_mask": torch.tensor(...),
        "advantages": torch.tensor(...),
    },
    "meta_info": {"global_token_num": [...]}
}

# Execute training step (Direct ZMQ to workers)
result = client.update_actor(batch)
metrics = result["meta_info"]["metrics"]
# Returns: {"actor/loss": 0.5, "actor/lr": 2e-6, "mfu/actor": 0.35, ...}

# Compute old log probs for GRPO (Direct ZMQ to workers)
result = client.compute_log_prob(batch)
old_log_probs = result["batch"]["old_log_probs"]

# Save checkpoint (HTTP to API server, broadcast to workers)
client.save_checkpoint(
    local_path="/path/to/checkpoint",
    global_step=100,
    save_weight_only=False
)
```

## ACK-Based Two-Phase Protocol

Training operations (`update_actor`, `compute_log_prob`) use an ACK-based two-phase protocol that prevents race conditions:

```
Phase 1: Send data to rank 0
─────────────────────────────
Client → Rank 0:
  message = {
      "op_id": "compute_log_prob_123",
      "phase": "data",
      "operation": "compute_log_prob",
      "data": full_batch,  # Not chunked!
      "use_nccl_scatter": True
  }
  worker_pushers[0].send_pyobj(message)  # Direct to rank 0 only

Rank 0:
  1. Receive data on PULL socket
  2. Store in pending_ops[op_id]
  3. Send ACK immediately → {"phase": "data_ack", "op_id": op_id}

Phase 2: Wait for ACK, then broadcast execute
──────────────────────────────────────────────
Client:
  1. Wait for ACK from rank 0 (confirms data received)
  2. POST /execute {"op_id": op_id}

API Server:
  PUB → {"phase": "execute", "op_id": op_id}

All Workers:
  Receive execute signal via SUB socket

Phase 3: NCCL scatter/gather and execution
───────────────────────────────────────────
Rank 0:
  1. Broadcast operation name via NCCL
  2. NCCL scatter: full_batch → data_chunks (respects DP/SP topology)
  3. Execute operation on data_chunks[0]
  4. NCCL gather: collect results from all workers
  5. Send gathered result to client via ZMQ

Other Ranks (1-N):
  1. Receive operation name from rank 0 (NCCL broadcast)
  2. Receive data_chunks[rank] from rank 0 (NCCL scatter)
  3. Execute operation on their chunk
  4. Send result to rank 0 (NCCL gather)
  5. No ZMQ communication needed
```

**Why ACK-Based?**
- **Prevents race conditions**: Execute signal uses PUB/SUB (different channel than data). ACK ensures data arrives before execute signal is sent.
- **Simple**: One small ACK message adds negligible overhead
- **Reliable**: Works across all network topologies and latencies

**Why Rank-0-Centric?**
- **NCCL is faster**: InfiniBand/NVLink >> network ZMQ for inter-worker communication
- **SP/DP aware**: Rank 0 knows device_mesh topology and distributes data correctly
- **Simple client**: Driver doesn't need to know about SP size or DP groups
- **Single network hop**: Only rank 0's network is used, others communicate via NCCL

## Configuration

### Training Service Config (recipe YAML)

```yaml
trainer:
  train_service:
    backend: "direct-zmq"
    url: "http://api-server:8000"
    identifier: "actor-group"
    world_size: 8

    config:
      zmq_recv_timeout_ms: 600000    # 10 min per worker result
      zmq_total_timeout_ms: 3600000  # 60 min total timeout

    training_config_path: "configs/train_config.yaml"
```

### Worker Training Config (train_config.yaml)

```yaml
actor:
  model:
    path: /path/to/model
    use_liger: false
    enable_gradient_checkpointing: false

  fsdp_config:
    fsdp_size: -1  # Full shard across all GPUs
    model_dtype: bf16
    mixed_precision:
      param_dtype: bf16
      reduce_dtype: fp32

  optim:
    lr: 2e-6
    betas: [0.9, 0.95]
    weight_decay: 0.01
    total_training_steps: 1000
    lr_warmup_steps_ratio: 0.1

  ppo_epochs: 1
  ppo_mini_batch_size: 128
  ppo_micro_batch_size: 16
```

## Core Components

### DirectZMQTrainServiceClient

Located in `nexrl/train_service_backend/api/direct_zmq_client.py`.

**Methods:**
- `initialize_worker()` - Initialize workers with config (HTTP)
- `init_model()` - Load model to GPU (HTTP)
- `update_actor()` - Training step (Direct ZMQ + two-phase)
- `compute_log_prob()` - Log probability computation (Direct ZMQ + two-phase)
- `save_checkpoint()` - Save model checkpoint (HTTP)
- `load_checkpoint()` - Load model checkpoint (HTTP)

### API Server

Located in `nexrl/train_service_backend/api/api_server.py`.

**Key Endpoints:**
- `POST /register_worker_group` - Register new worker group
- `POST /report_worker_endpoint` - Workers report their ZMQ endpoints
- `GET /get_worker_group_info` - Get worker endpoints for client connection
- `POST /execute` - Broadcast execute signal (Phase 2)
- `POST /initialize` - Initialize workers with config
- `POST /init_model` - Load model to GPU
- `POST /save_checkpoint` - Save checkpoint
- `POST /heartbeat` - Worker heartbeat (every 5 seconds)

**ZMQ Sockets per Worker Group:**
```
base_port     : PUB socket - broadcast commands and execute signals to workers
base_port + 1 : PULL socket - collect results from workers
base_port + 2 : PUSH socket - forward results to clients
```

### Worker Processes

Located in:
- `nexrl/train_service_backend/distributed/worker_process.py` - Coordination
- `nexrl/train_service_backend/fsdp_worker/fsdp_workers.py` - FSDP training logic

**Worker Commands:**
- `initialize` - Load config, create model wrapper
- `init_model` - Load model to GPU, create optimizer
- `update_actor` - Training step (from pending_ops after execute signal)
- `compute_log_prob` - Log prob computation
- `save_checkpoint` - Save checkpoint
- `load_checkpoint` - Load checkpoint
- `barrier` - Synchronization barrier
- `destroy` - Cleanup and exit

## Environment Variables

### For Workers

```bash
# Worker data port (port = base + rank)
export NEXRL_WORKER_DATA_BASE_PORT=6000

# Worker IP (Kubernetes: use pod IP, not hostname)
export NEXRL_WORKER_IP=10.0.1.5

# API server URL
export API_SERVER_URL=http://api-server:8000
```

### For Training (NCCL)

```bash
# NCCL timeout (increase for slow operations)
export NCCL_TIMEOUT_MS=600000  # 10 minutes

# NCCL debug logging
export NCCL_DEBUG=INFO
```

## Kubernetes Deployment

```yaml
# Worker StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: nexrl-workers
spec:
  replicas: 8
  template:
    spec:
      containers:
      - name: worker
        image: nexrl:latest
        command: ["torchrun", "--nproc_per_node=1", "--standalone"]
        args:
        - "-m"
        - "nexrl.train_service_backend.distributed.worker_process"
        - "--api-server-url=http://nexrl-api-server:8000"
        - "--identifier=actor-group"
        - "--backend=fsdp"
        env:
        - name: NEXRL_WORKER_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP  # Use pod IP, not hostname
        - name: NEXRL_WORKER_DATA_BASE_PORT
          value: "6000"
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 6000
          name: zmq-data
```

## Common Issues

### Workers Not Reporting Endpoints

**Symptom:** `RuntimeError: Only 0/8 workers have reported endpoints`

**Fix:**
```bash
# Check worker logs
kubectl logs nexrl-workers-0

# Verify API server connectivity
kubectl exec nexrl-workers-0 -- curl http://nexrl-api-server:8000/health

# Verify NEXRL_WORKER_IP is set correctly (must be pod IP, not hostname)
kubectl exec nexrl-workers-0 -- env | grep NEXRL_WORKER_IP
```

### NCCL Timeout During Training

**Symptom:** `NCCL error: timeout` during `update_actor`

**Fix:**
```bash
# Increase NCCL timeout
export NCCL_TIMEOUT_MS=600000

# Enable debug logging to find slow worker
export NCCL_DEBUG=INFO
```

### Client Timeout Collecting Results

**Symptom:** `RuntimeError: Timeout collecting results from rank 0`

**Causes:**
- Rank 0 crashed (check logs)
- CUDA OOM on rank 0
- NCCL hang (any worker failure causes NCCL collective to hang)
- Network issue to rank 0

**Fix:** Restart all workers (NCCL requires full restart after any worker failure)

### Only Rank 0 Receiving Data via ZMQ

**Symptom:** Logs show only rank 0 receives data via ZMQ, not all workers

**Cause:** This is **expected behavior**! In the ACK-based rank-0-centric architecture:
- Only rank 0 receives data via ZMQ PULL socket
- Rank 0 sends ACK to confirm receipt
- Other workers receive data via NCCL scatter (much faster)

**Verify:**
```bash
# Check rank 0 logs
grep "Stored data for op_id" worker-0.log  # Should see data reception
grep "Sent data ACK" worker-0.log          # Should see ACK sent
grep "NCCL scatter" worker-0.log           # Should see NCCL scatter

# Check other ranks
grep "participating in NCCL scatter" worker-1.log  # Should see NCCL participation
```

## Related Documentation

- [Self-Hosted Trainers](../06-trainers/self-hosted-trainers.md)
- [FSDP Configuration](../08-features/fsdp.md)
- [Checkpointing](../08-features/checkpointing.md)
