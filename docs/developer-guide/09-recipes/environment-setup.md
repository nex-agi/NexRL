# Environment Setup

This document explains how to configure environment variables and setup scripts for NexRL recipes.

## Overview

Environment setup scripts (`.env.sh` files) configure the runtime environment before training starts. They export necessary environment variables, set up services, and prepare the workspace.

## Environment Setup Scripts

### File Naming Convention

Environment scripts are named by deployment mode:

- `self_hosted.env.sh` - Self-hosted mode setup
- `tinker.env.sh` - Tinker service mode setup
- `weaver.env.sh` - Weaver service mode setup

### Script Location

Place scripts in the recipe root directory:

```
recipe/my_task/
├── common.yaml
├── self_hosted.yaml
├── self_hosted.env.sh       # Environment setup script
└── agent_workspace/
```

### Configuration Reference

Reference the script in mode-specific YAML:

```yaml
# self_hosted.yaml
environment:
  setup_script: "self_hosted.env.sh"
  require_setup_script: true    # Fail if script errors
```

## Basic Environment Script

### Minimal Example

```bash
#!/bin/bash
set -e  # Exit on error

echo "Setting up environment..."

# Export required variables
export NEXRL_DATA_PATH="/path/to/data"
export NEXRL_MODEL_PATH="/path/to/models"
export EXPERIMENT_PATH="/path/to/output"

echo "Environment setup complete"
```

### Real Example (News Task)

From `recipe/nexau_news/self_hosted.env.sh`:

```bash
set -e

echo "=========================================="
echo "Setting up environment for News (Self-hosted)"
echo "=========================================="

# LangFuse Monitoring (Optional)
export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"

echo "=========================================="
echo "✓ News (Self-hosted) environment setup complete"
echo "=========================================="
```

## Common Environment Variables

### Data Paths

**`NEXRL_DATA_PATH`** (required)
- Base path for training data files
- Used in configs: `${oc.env:NEXRL_DATA_PATH}/task/train.parquet`

```bash
export NEXRL_DATA_PATH="/gpfs/data/nexrl"
```

**`NEXRL_MODEL_PATH`** (required)
- Base path for model files (HuggingFace models)
- Used for tokenizers and model weights

```bash
export NEXRL_MODEL_PATH="/gpfs/models/huggingface.co"
```

### Output Paths

**`EXPERIMENT_PATH`** (self-hosted)
- Output directory for checkpoints and weights
- Used for `checkpoint_path` and `sync_weight_path`

```bash
export EXPERIMENT_PATH="/gpfs/experiments/my_task"
```

### Service URLs

**`INFERENCE_BASE_URL`** (self-hosted)
- Inference service endpoint (SGLang/vLLM)
- Format: `http://hostname:port`

```bash
export INFERENCE_BASE_URL="http://10.0.0.1:8000"
```

**`API_SERVER_URL`** (self-hosted)
- Training service API server address
- Used to construct training service URL

```bash
export API_SERVER_URL="10.0.0.2"
```

### API Keys

**`WEAVER_API_KEY`** (weaver mode)
- Weaver service authentication key

```bash
export WEAVER_API_KEY="sk-your-weaver-api-key"
```

**`TINKER_API_KEY`** (tinker mode)
- Tinker service authentication key

```bash
export TINKER_API_KEY="tml-your-tinker-api-key"
```

### Optional Services

**`FEISHU_WEBHOOK_URL`** (optional)
- Feishu/Lark webhook for notifications

```bash
export FEISHU_WEBHOOK_URL="https://open.feishu.cn/..."
```

**LangFuse Variables** (optional)
- For agent tracing and monitoring

```bash
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

## Environment Script Patterns

### Error Handling

Use `set -e` to exit on any command failure:

```bash
#!/bin/bash
set -e  # Exit immediately if command fails

# Commands here will fail the script if they error
export REQUIRED_VAR="value"
```

### Conditional Variables

Set variables only if not already defined:

```bash
# Use existing value or default
export NEXRL_DATA_PATH="${NEXRL_DATA_PATH:-/default/path/to/data}"
```

### Derived Variables

Compute variables from other variables:

```bash
export BASE_PATH="/gpfs/nexrl"
export NEXRL_DATA_PATH="${BASE_PATH}/data"
export EXPERIMENT_PATH="${BASE_PATH}/experiments/${USER}"
```

### Service Checks

Verify services are available:

```bash
# Check if inference service is reachable
if ! curl -s "${INFERENCE_BASE_URL}/health" > /dev/null; then
    echo "Warning: Inference service not reachable"
fi
```

### Creating Directories

Ensure output directories exist:

```bash
export EXPERIMENT_PATH="/gpfs/experiments/my_task"

# Create directory if it doesn't exist
mkdir -p "${EXPERIMENT_PATH}"
mkdir -p "${EXPERIMENT_PATH}/ckpt"
mkdir -p "${EXPERIMENT_PATH}/sync_weight"
```

## Mode-Specific Setup

### Self-Hosted Mode

Self-hosted mode requires paths and service URLs:

```bash
#!/bin/bash
set -e

echo "Setting up self-hosted environment..."

# Data and model paths
export NEXRL_DATA_PATH="/gpfs/data/nexrl"
export NEXRL_MODEL_PATH="/gpfs/models/huggingface.co"

# Output paths
export EXPERIMENT_PATH="/gpfs/experiments/my_task"

# Service endpoints
export INFERENCE_BASE_URL="http://inference-service:8000"
export API_SERVER_URL="training-api-server"

# Optional: Notifications
export FEISHU_WEBHOOK_URL="https://open.feishu.cn/..."

echo "✓ Self-hosted environment ready"
```

### Tinker Mode

Tinker mode primarily needs API key and data paths:

```bash
#!/bin/bash
set -e

echo "Setting up Tinker environment..."

# Tinker API key (required)
export TINKER_API_KEY="tml-your-api-key"

# Data paths
export NEXRL_DATA_PATH="/gpfs/data/nexrl"

# Optional: Agent tracing
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"

echo "✓ Tinker environment ready"
```

### Weaver Mode

Weaver mode needs API key and data paths:

```bash
#!/bin/bash
set -e

echo "Setting up Weaver environment..."

# Weaver API key (required)
export WEAVER_API_KEY="sk-your-api-key"

# Data paths
export NEXRL_DATA_PATH="/gpfs/data/nexrl"

# Optional: Notifications
export FEISHU_WEBHOOK_URL="https://open.feishu.cn/..."

echo "✓ Weaver environment ready"
```

## Workspace Setup

### Agent Workspace

For tasks with custom tools or complex setups:

```bash
#!/bin/bash
set -e

# Paths
export RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AGENT_WORKSPACE="${RECIPE_DIR}/agent_workspace"

# Add agent workspace to Python path
export PYTHONPATH="${AGENT_WORKSPACE}:${PYTHONPATH}"

# Ensure tools are accessible
if [ -d "${AGENT_WORKSPACE}/tools" ]; then
    echo "✓ Agent tools found"
fi
```

### Custom Dependencies

Install or activate task-specific dependencies:

```bash
#!/bin/bash
set -e

# Activate virtual environment if needed
if [ -f "${RECIPE_DIR}/venv/bin/activate" ]; then
    source "${RECIPE_DIR}/venv/bin/activate"
fi

# Install task-specific packages
pip install -q -r "${RECIPE_DIR}/requirements.txt"
```

## Integration with Configuration

### Environment Script Execution

The controller loads and executes the setup script before initialization:

1. Recipe config specifies: `environment.setup_script: "self_hosted.env.sh"`
2. Controller resolves path relative to recipe directory
3. Script is executed in a shell
4. Environment variables are available to all modules
5. If `require_setup_script: true`, errors fail the training

### Variable Usage in Config

Environment variables are interpolated in YAML:

```yaml
# Recipe config uses variables exported by setup script
data:
  data_files:
    - "${oc.env:NEXRL_DATA_PATH}/task/train.parquet"
  tokenizer_path: "${oc.env:NEXRL_MODEL_PATH}/model"

service:
  inference_service:
    base_url: "${oc.env:INFERENCE_BASE_URL}"

trainer:
  checkpoint_path: "${oc.env:EXPERIMENT_PATH}/ckpt"
```

### Optional vs Required

**Optional Setup** (`require_setup_script: false`)
- Script errors are logged but don't fail training
- Useful for optional services (monitoring, notifications)

```yaml
environment:
  setup_script: "self_hosted.env.sh"
  require_setup_script: false
```

**Required Setup** (`require_setup_script: true`)
- Script errors fail the training immediately
- Use for essential variables and services

```yaml
environment:
  setup_script: "self_hosted.env.sh"
  require_setup_script: true
```

## Best Practices

### Organization

1. **One script per mode** - Keep mode-specific setup separate
2. **Use clear sections** - Group related variables with comments
3. **Include echo statements** - Show progress during setup
4. **Error on missing required vars** - Validate critical variables

### Error Handling

1. **Always use `set -e`** - Fail fast on errors
2. **Check critical services** - Verify availability when possible
3. **Validate paths exist** - Check important directories
4. **Provide helpful messages** - Clear error messages

### Security

1. **Don't commit secrets** - Use placeholder values in repo
2. **Document required secrets** - List in README
3. **Use secure storage** - Load from credential managers when possible
4. **Rotate keys regularly** - Follow security best practices

### Portability

1. **Use environment variables** - Don't hard-code paths
2. **Provide defaults** - Use `${VAR:-default}` pattern
3. **Make paths relative when possible** - To recipe directory
4. **Document assumptions** - Storage requirements, services needed

## Troubleshooting

### Script Not Found

**Problem:** `Environment script not found`

**Solution:**
- Check script path in YAML config
- Verify file exists at specified location
- Ensure path is relative to recipe directory
- Check file permissions (should be readable)

### Variables Not Set

**Problem:** Config shows `${oc.env:VAR_NAME}` instead of value

**Solution:**
- Verify script is executed (check logs)
- Ensure variable is exported (not just set)
- Check spelling of variable name
- Set `require_setup_script: true` to catch errors

### Script Fails Silently

**Problem:** Script errors but training continues

**Solution:**
- Set `require_setup_script: true` in config
- Add `set -e` at script start
- Check script exit code manually
- Review error logs

### Permission Denied

**Problem:** Can't execute setup script

**Solution:**
- Make script executable: `chmod +x *.env.sh`
- Or ensure bash can read it (doesn't need +x)
- Check file ownership
- Verify no SELinux/AppArmor issues

## Advanced Topics

### Dynamic Configuration

Generate configuration at runtime:

```bash
#!/bin/bash
set -e

# Detect GPU count and set workers
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
export NUM_WORKERS=$((GPU_COUNT * 16))

echo "Detected ${GPU_COUNT} GPUs, setting NUM_WORKERS=${NUM_WORKERS}"
```

### Multi-User Setup

Handle shared resources:

```bash
#!/bin/bash
set -e

# User-specific experiment path
export EXPERIMENT_PATH="/gpfs/experiments/${USER}/my_task"

# Create user directories
mkdir -p "${EXPERIMENT_PATH}"
```

### Service Discovery

Dynamically find services:

```bash
#!/bin/bash
set -e

# Discover inference service via Kubernetes
INFERENCE_SERVICE=$(kubectl get svc inference-service -o jsonpath='{.spec.clusterIP}')
export INFERENCE_BASE_URL="http://${INFERENCE_SERVICE}:8000"

echo "Found inference service at ${INFERENCE_BASE_URL}"
```

## Examples from Real Recipes

### Math Task (Minimal)

Math task has no environment setup script as it doesn't require special services:

```yaml
# math/self_hosted.yaml
environment:
  setup_script: ""
  require_setup_script: false
```

All necessary variables are expected to be set externally.

### News Task (Standard)

News task has basic setup for optional monitoring:

```bash
# nexau_news/self_hosted.env.sh
set -e

echo "Setting up environment for News (Self-hosted)"

# LangFuse Monitoring (Optional)
export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"

echo "✓ News environment setup complete"
```

### Deep Search Task (With Tools)

Deep search may set up additional tool dependencies or API keys for custom tools.

## Related Documentation

- [Recipe Structure](./recipe-structure.md) - Recipe organization
- [Recipe Configuration](./recipe-configuration.md) - YAML configuration
- [Configuration Setup](../01-getting-started/configuration-setup.md) - Initial environment setup
- [Deployment Modes](../01-getting-started/deployment-modes.md) - Mode-specific requirements

---

**Complete**: Recipe documentation finished. See [README](../README.md) for next sections.
