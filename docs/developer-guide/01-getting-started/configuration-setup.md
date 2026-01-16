# Configuration Setup

NexRL supports three configuration methods with automatic fallback, allowing you to get started quickly and scale to production deployments.

## Configuration Methods Overview

| Method | When to Use | Setup Effort | Best For |
|--------|-------------|--------------|----------|
| **Built-in Defaults** | Testing, demos | None | Quick experimentation |
| **Environment Variables** | Development | Low | Individual developers |
| **Kubernetes ConfigMaps** | Production | Medium | Teams, production |

NexRL attempts configuration methods in order: ConfigMaps → Environment Variables → Built-in Defaults.

## Built-in Defaults (Zero Configuration)

### What's Included

NexRL ships with sensible defaults that let you run immediately:
- Public Docker images from the registry
- `/tmp` storage for temporary files
- Default resource allocations
- Sample configurations

### Usage

Just run without any setup:

```bash
nexrl -m self-hosted -c recipe/my_task/my_task.yaml --run-nexrl
```

### Limitations

- ❌ Storage is not persistent (uses `/tmp`)
- ❌ Cannot customize Docker images
- ❌ No W&B logging (no API key)
- ❌ Not suitable for production

### When to Use

- Quick testing and experimentation
- Learning NexRL basics
- Demo environments
- CI/CD pipeline testing

## Environment Variables (Development)

### Quick Setup

Use the provided setup script:

```bash
source cli/setup_env.sh
```

This script prompts you for key values and sets up your environment.

### Manual Setup

Alternatively, set variables manually in your shell:

```bash
# Storage configuration
export NEXRL_STORAGE_PATH="/your/persistent/storage"

# Docker images
export NEXRL_WORKER_IMAGE="your-registry/nexrl:tag"
export NEXRL_CONTROLLER_IMAGE="your-registry/nexrl:tag"

# Logging
export WANDB_KEY="your-wandb-key"

# Optional: Kubernetes namespace
export NEXRL_NAMESPACE="nexrl"

# Optional: Resource limits
export NEXRL_DRIVER_CPU="8"
export NEXRL_DRIVER_MEMORY="32Gi"
export NEXRL_ROLLOUT_CPU="4"
export NEXRL_ROLLOUT_MEMORY="16Gi"
```

### Persisting Configuration

Add to your shell profile (`~/.bashrc`, `~/.zshrc`):

```bash
# NexRL Configuration
export NEXRL_STORAGE_PATH="/gpfs/users/myname/nexrl_storage"
export NEXRL_WORKER_IMAGE="myregistry/nexrl:latest"
export NEXRL_CONTROLLER_IMAGE="myregistry/nexrl:latest"
export WANDB_KEY="abc123..."
```

### Verifying Setup

Check that variables are set:

```bash
env | grep NEXRL
env | grep WANDB
```

### Benefits

✅ Quick to set up
✅ Easy to modify
✅ Works for individual developers
✅ No cluster-wide configuration needed
✅ Can override on a per-session basis

### Limitations

- ⚠️ Not shared across team members
- ⚠️ Can be accidentally overwritten
- ⚠️ Requires setting up for each developer
- ⚠️ Not version controlled

### When to Use

- Individual development
- Personal experiments
- Quick iteration cycles
- Testing different configurations
- Development environments

## Kubernetes ConfigMaps (Production)

### One-Time Cluster Setup

Set up ConfigMaps once for the entire cluster:

```bash
cd cli/setup

# 1. Create namespace
kubectl apply -f 01-namespace.yaml

# 2. Edit and apply admin configuration
# Edit 02-admin-config.yaml with your values
kubectl apply -f 02-admin-config.yaml

# 3. Edit and apply user configuration
# Edit 03-user-config.yaml with your values
kubectl apply -f 03-user-config.yaml

# 4. Create Volcano queue
kubectl apply -f 04-volcano-queue.yaml

# 5. Set up RBAC
kubectl apply -f 09-nexrl-job-rbac.yaml
```

### Admin Configuration

Edit `02-admin-config.yaml` before applying:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nexrl-admin-config
  namespace: nexrl
data:
  # Docker registry configuration
  REGISTRY_URL: "your-registry.company.com"
  REGISTRY_PROJECT: "nexrl"

  # Default images
  NEXRL_WORKER_IMAGE: "your-registry.company.com/nexrl/worker:v1.0.0"
  NEXRL_CONTROLLER_IMAGE: "your-registry.company.com/nexrl/controller:v1.0.0"

  # Storage paths
  NEXRL_STORAGE_ROOT: "/gpfs/shared/nexrl"

  # Resource defaults
  NEXRL_DEFAULT_GPU_TYPE: "nvidia.com/gpu"
  NEXRL_DEFAULT_QUEUE: "nexrl-queue"

  # Namespace
  NEXRL_NAMESPACE: "nexrl"
```

### User Configuration

Edit `03-user-config.yaml` before applying:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nexrl-user-config
  namespace: nexrl
data:
  # Logging configuration
  WANDB_PROJECT: "nexrl-experiments"
  WANDB_ENTITY: "your-team"

  # User-specific paths (optional)
  # NEXRL_USER_STORAGE: "/gpfs/users/${USER}/nexrl"
```

### Secrets for Sensitive Data

Create a Secret for API keys:

```bash
kubectl create secret generic nexrl-secrets \
  --from-literal=WANDB_KEY=your-wandb-key \
  --from-literal=TINKER_API_KEY=your-tinker-key \
  -n nexrl
```

### Verifying Setup

Check that ConfigMaps are created:

```bash
kubectl get configmap -n nexrl
kubectl describe configmap nexrl-admin-config -n nexrl
kubectl describe configmap nexrl-user-config -n nexrl
```

Check secrets:

```bash
kubectl get secret nexrl-secrets -n nexrl
```

### Benefits

✅ Shared across entire team
✅ Version controlled (YAML files)
✅ Cluster-wide consistency
✅ Separation of admin vs user config
✅ Secrets properly managed
✅ Production-ready

### Limitations

- ⚠️ Requires cluster admin access for initial setup
- ⚠️ Changes require kubectl apply
- ⚠️ Less flexible for individual experiments

### When to Use

- Production deployments
- Team environments
- Shared clusters
- Consistent configuration across users
- Long-term projects

## Configuration Priority

NexRL checks configuration sources in this order:

```
1. Kubernetes ConfigMaps (if available)
   ↓
2. Environment Variables (if set)
   ↓
3. Built-in Defaults (always available)
```

Each level overrides the previous one. This allows you to:
- Use team-wide ConfigMaps by default
- Override with environment variables for experiments
- Fall back to built-in defaults for quick testing

## Common Configuration Variables

### Required for Production

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXRL_STORAGE_PATH` | Persistent storage path | `/gpfs/shared/nexrl` |
| `NEXRL_WORKER_IMAGE` | Docker image for workers | `registry/nexrl:v1.0` |
| `NEXRL_CONTROLLER_IMAGE` | Docker image for controller | `registry/nexrl:v1.0` |
| `WANDB_KEY` | Weights & Biases API key | `abc123...` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXRL_NAMESPACE` | Kubernetes namespace | `nexrl` |
| `NEXRL_DRIVER_CPU` | Driver CPU request | `8` |
| `NEXRL_DRIVER_MEMORY` | Driver memory request | `32Gi` |
| `NEXRL_ROLLOUT_CPU` | Rollout worker CPU | `4` |
| `NEXRL_ROLLOUT_MEMORY` | Rollout worker memory | `16Gi` |
| `NEXRL_INFERENCE_GPU` | Inference GPU count | `2` |
| `NEXRL_TRAINING_GPU` | Training GPU count | `4` |

## Storage Configuration

### Storage Requirements

NexRL needs high-performance shared storage for:
- Model checkpoints
- Weight synchronization buffers
- Training logs
- Configuration files
- Temporary data

### Supported Storage Types

**Recommended:**
- GPFS (IBM Spectrum Scale)
- Lustre
- High-performance NFS with local caching
- Parallel file systems

**Not Recommended:**
- Standard NFS (too slow for large models)
- Local storage (not shared across pods)
- Network file systems without caching

### Setting Up Storage

**Using Environment Variables:**

```bash
export NEXRL_STORAGE_PATH="/gpfs/users/myname/nexrl_storage"
```

**Using ConfigMap:**

```yaml
data:
  NEXRL_STORAGE_ROOT: "/gpfs/shared/nexrl"
```

**In Recipe Configuration:**

```yaml
trainer:
  checkpoint_path: "${env:NEXRL_STORAGE_PATH}/checkpoints/${experiment_name}"
  sync_weight_path: "${env:NEXRL_STORAGE_PATH}/sync_weights/${experiment_name}"
```

### Directory Structure

NexRL creates this structure in your storage path:

```
${NEXRL_STORAGE_PATH}/
├── checkpoints/
│   └── ${experiment_name}/
│       ├── global_step_100/
│       ├── global_step_200/
│       └── ...
├── sync_weights/
│   └── ${experiment_name}/
│       └── latest/
├── logs/
│   └── ${experiment_name}/
└── data/
    └── ${experiment_name}/
```

## Docker Image Configuration

### Using Custom Images

If you've built custom NexRL images:

```bash
# Build images
docker build -t myregistry/nexrl:v1.0 .
docker push myregistry/nexrl:v1.0

# Configure NexRL to use them
export NEXRL_WORKER_IMAGE="myregistry/nexrl:v1.0"
export NEXRL_CONTROLLER_IMAGE="myregistry/nexrl:v1.0"
```

### Image Pull Secrets

If your registry requires authentication:

```bash
kubectl create secret docker-registry nexrl-registry-secret \
  --docker-server=myregistry.com \
  --docker-username=myuser \
  --docker-password=mypass \
  -n nexrl
```

Reference in your pod specs (automatically handled by CLI).

## Troubleshooting

### Issue: ConfigMap not found

**Symptoms:**
```
Error: configmap "nexrl-admin-config" not found
```

**Solution:**
```bash
kubectl apply -f cli/setup/02-admin-config.yaml
```

### Issue: Storage path not accessible

**Symptoms:**
```
Error: Permission denied accessing /gpfs/...
```

**Solutions:**
1. Check path exists: `ls -ld $NEXRL_STORAGE_PATH`
2. Check permissions: `chmod 755 $NEXRL_STORAGE_PATH`
3. Verify mount in cluster: `kubectl exec -it POD -- ls -ld /path`

### Issue: Image pull failures

**Symptoms:**
```
Error: ImagePullBackOff
```

**Solutions:**
1. Verify image exists: `docker pull $NEXRL_WORKER_IMAGE`
2. Check image pull secrets are configured
3. Verify registry URL is correct

### Issue: Environment variables not loaded

**Symptoms:**
Configuration uses defaults instead of your values.

**Solutions:**
1. Verify variables are set: `env | grep NEXRL`
2. Restart your shell after modifying `.bashrc`
3. Check variables are exported: `export VARIABLE=value`

## Best Practices

### For Development

1. **Use environment variables** for quick iteration
2. **Set in shell profile** for consistency
3. **Use personal storage paths** to avoid conflicts
4. **Keep secrets out of scripts** (use separate secret management)

### For Production

1. **Use ConfigMaps** for all configuration
2. **Use Secrets** for sensitive data (API keys, passwords)
3. **Version control** ConfigMap YAML files
4. **Document changes** when updating ConfigMaps
5. **Use separate namespaces** for dev/staging/prod
6. **Implement backup strategy** for shared storage
7. **Set resource limits** to prevent resource exhaustion
8. **Monitor storage usage** regularly

### For Teams

1. **Standardize on ConfigMaps** for shared configuration
2. **Document storage paths** and conventions
3. **Use consistent image tags** (not `latest`)
4. **Create onboarding guide** for new team members
5. **Set up CI/CD** for image building
6. **Use RBAC** to control access

## Next Steps

- Learn about [Deployment Modes](./deployment-modes.md) to choose your setup
- Explore [Recipe Configuration](../10-recipes/recipe-configuration.md) for task-specific settings
- Review [Complete Configuration Reference](../12-configuration-reference/complete-config.md)
- Set up [Distributed Execution](../11-distributed-execution/ray-integration.md) for scaling
