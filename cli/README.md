# NexRL CLI Reference

> **Quick Start:** See [main README](../README.md) for installation and first run.

---

## Usage

```bash
nexrl -m MODE -c CONFIG [OPTIONS]
```

**Options:**
- `-m, --mode` - `self-hosted` or `training-service` (required)
- `-c, --train-config` - Path to training YAML (required)
- `-r, --run-nexrl` - Auto-start training
- `-t, --tag` - Custom job tag
- `--serving-only` - [self-hosted] Only launch inference
- `--no-serving` - [self-hosted] Skip inference

---

## Configuration

Three methods with automatic fallback:

| Method | When to Use | Setup |
|--------|-------------|-------|
| **Zero-Setup** | Testing, demos | None - just run! |
| **Environment Variables** | Development | `source cli/setup_env.sh` |
| **ConfigMaps** | Production, teams | `kubectl apply -f cli/setup/` |

### Environment Variables

**Quick Setup (Recommended for Development):**

```bash
# Source the setup script
source cli/setup_env.sh

# Or customize it first
cp cli/setup_env.sh my_env.sh
# Edit my_env.sh with your values
source my_env.sh
```

**Manual Setup:**

```bash
export NEXRL_STORAGE_PATH="/your/path"
export NEXRL_WORKER_IMAGE="your-image:tag"
export NEXRL_CONTROLLER_IMAGE="your-image:tag"
export WANDB_KEY="your-key"
```

**Full list:**

| Variable | Default |
|----------|---------|
| `NEXRL_STORAGE_PATH` | `/tmp/nexrl` |
| `NEXRL_WORKER_IMAGE` | `nexagi/nexrl:latest` |
| `NEXRL_CONTROLLER_IMAGE` | `nexagi/nexrl:latest` |
| `NEXRL_INFERENCE_IMAGE` | `lmsysorg/sglang:v0.5.4.post2` |
| `NEXRL_QUEUE` | `default` |
| `NEXRL_USER_ID` | `$USER` |
| `WANDB_KEY` | `` |

### Production Setup

For persistent storage and custom images:

```bash
cd cli/setup
kubectl apply -f *.yaml  # Edit files first!
```

See [setup/README.md](./setup/README.md) for details.

---

## Monitoring

```bash
# List all resources
kubectl get all -n nexrl

# View logs
kubectl logs -f -l app=JOB_NAME-driver -n nexrl

# Exec into pod
kubectl exec -it POD_NAME -n nexrl -- bash
```

---

## Architecture

**Self-Hosted:**
1. API Server → Training backend
2. GPU Workers → FSDP training
3. Inference → SGLang serving
4. Driver → NexRL + Ray agents

**Training-Service:**
1. Driver → Connects to external services

---

---

## Troubleshooting

**Using defaults?** Fine for testing! For production, set `NEXRL_STORAGE_PATH` or use ConfigMaps.

**Image errors?** Set `NEXRL_WORKER_IMAGE` to your registry.

**Storage errors?** Set `NEXRL_STORAGE_PATH` to accessible path.

**WandB disabled?** Set `WANDB_KEY` environment variable.

---

**More:** [setup/README.md](./setup/README.md) | [Main README](../README.md)
