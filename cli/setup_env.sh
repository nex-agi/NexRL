# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export NEXRL_SYSTEM_NAMESPACE="nexrl"

# Volcano scheduler queue
export NEXRL_QUEUE="default"

# Priority class (optional - leave empty for default)
export NEXRL_PRIORITY_CLASS=""

# Service account for pod execution
export NEXRL_SERVICE_ACCOUNT="default"

# ============================================================
# Docker Images
# ============================================================

# Controller/Worker image (NexRL driver and agents)
export NEXRL_CONTROLLER_IMAGE="nexagi/nexrl:v1.0.0"
export NEXRL_WORKER_IMAGE="nexagi/nexrl:v1.0.0"

# Inference image (SGLang)
export NEXRL_INFERENCE_IMAGE="lmsysorg/sglang:v0.5.4.post2"

# ============================================================
# Storage Configuration
# ============================================================

# Storage path - change to your persistent storage location
# Default: /tmp/nexrl (fine for testing, not for production!)
export NEXRL_STORAGE_PATH="/tmp/nexrl"

# Storage type: hostPath or nfs
export NEXRL_STORAGE_TYPE="hostPath"

# NFS server (only if NEXRL_STORAGE_TYPE="nfs")
# export NEXRL_NFS_SERVER="your-nfs-server.example.com"
# export NEXRL_NFS_PATH="/path/to/storage"

# ============================================================
# User Configuration
# ============================================================

# User ID (defaults to $USER if not set)
export NEXRL_USER_ID="${USER}"
export NEXRL_USER_NAME="${USER}"

# NexRL path (auto-detected if not set)
if [ -z "$NEXRL_PATH" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export NEXRL_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# ============================================================
# Optional: WandB Configuration
# ============================================================

# WandB API key for experiment tracking (leave empty to disable)
export WANDB_KEY="${WANDB_KEY:-}"

# WandB host (optional - for self-hosted WandB)
# export WANDB_HOST="https://api.wandb.ai"

# ============================================================
# Optional: Training Service API Keys
# ============================================================

# Only needed for training-service mode with external services
# Leave empty if using self-hosted mode only

# Tinker configuration (optional)
export TINKER_API_KEY=""
export TINKER_BASE_URL=""

# Weaver configuration (optional)
export WEAVER_API_KEY=""
export WEAVER_BASE_URL=""

# ============================================================
# Optional: Advanced Configuration
# ============================================================

# Registry (optional - for private registries)
# export NEXRL_REGISTRY="your-registry.example.com"

# SGLang router service account (optional)
# export SGLANG_ROUTER_ACCOUNT_NAME="sglang-router"

# ============================================================
# Validation
# ============================================================

echo "=================================================="
echo "✓ NexRL CLI Environment Configured"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  NEXRL_SYSTEM_NAMESPACE: ${NEXRL_SYSTEM_NAMESPACE}"
echo "  NEXRL_QUEUE: ${NEXRL_QUEUE}"
echo "  NEXRL_SERVICE_ACCOUNT: ${NEXRL_SERVICE_ACCOUNT}"
echo "  NEXRL_CONTROLLER_IMAGE: ${NEXRL_CONTROLLER_IMAGE}"
echo "  NEXRL_WORKER_IMAGE: ${NEXRL_WORKER_IMAGE}"
echo "  NEXRL_INFERENCE_IMAGE: ${NEXRL_INFERENCE_IMAGE}"
echo "  NEXRL_STORAGE_PATH: ${NEXRL_STORAGE_PATH}"
echo "  NEXRL_STORAGE_TYPE: ${NEXRL_STORAGE_TYPE}"
echo "  NEXRL_PATH: ${NEXRL_PATH}"
echo "  NEXRL_USER_ID: ${NEXRL_USER_ID}"
echo ""

# Check kubectl configuration
if command -v kubectl &> /dev/null; then
    echo "✓ kubectl is available"

    # Check if kubectl is configured
    if kubectl cluster-info &> /dev/null; then
        echo "✓ kubectl is configured and connected to cluster"
        CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "unknown")
        echo "  Current context: ${CURRENT_CONTEXT}"
    else
        echo "⚠️  WARNING: kubectl not configured or cannot connect to cluster"
        echo "   Configure kubectl: export KUBECONFIG=/path/to/kubeconfig.yaml"
        echo "   Or place config at: ~/.kube/config"
    fi

    # Check namespace
    if kubectl get namespace "${NEXRL_SYSTEM_NAMESPACE}" &> /dev/null; then
        echo "✓ Namespace '${NEXRL_SYSTEM_NAMESPACE}' exists"
    else
        echo "⚠️  WARNING: Namespace '${NEXRL_SYSTEM_NAMESPACE}' does not exist"
        echo "   Create it: kubectl create namespace ${NEXRL_SYSTEM_NAMESPACE}"
        echo "   Or use setup: kubectl apply -f cli/setup/01-namespace.yaml"
    fi

    # Check Volcano scheduler
    if kubectl get crd volcanojobs.batch.volcano.sh &> /dev/null; then
        echo "✓ Volcano scheduler is installed"
    else
        echo "⚠️  WARNING: Volcano scheduler not found"
        echo "   Install: https://github.com/volcano-sh/volcano"
    fi
else
    echo "⚠️  WARNING: kubectl not found in PATH"
    echo "   Please install kubectl"
fi

echo ""
echo "Optional Services:"
if [ -n "$WANDB_KEY" ]; then
    echo "  ✓ WandB logging enabled"
else
    echo "  ○ WandB logging disabled (set WANDB_KEY to enable)"
fi

if [ -n "$TINKER_API_KEY" ] || [ -n "$WEAVER_API_KEY" ]; then
    echo "  ✓ Training service APIs configured"
else
    echo "  ○ Training service APIs not configured (only needed for training-service mode)"
fi

echo ""
echo "Storage Configuration:"
if [ "$NEXRL_STORAGE_PATH" = "/tmp/nexrl" ]; then
    echo "  ⚠️  Using /tmp/nexrl (fine for testing, not for production!)"
    echo "     Set NEXRL_STORAGE_PATH to persistent storage for production"
else
    echo "  ✓ Storage path: ${NEXRL_STORAGE_PATH}"
fi

echo ""
echo "=================================================="
echo "✓ Ready to use NexRL CLI!"
echo ""
echo "Usage:"
echo "  # Self-hosted mode"
echo "  nexrl -m self-hosted -c recipe/your_recipe.yaml --run-nexrl"
echo ""
echo "  # Training-service mode"
echo "  nexrl -m training-service -c recipe/your_recipe.yaml --run-nexrl"
echo ""
echo "Alternative: Use Kubernetes ConfigMaps (recommended for production)"
echo "  kubectl apply -f cli/setup/01-namespace.yaml"
echo "  kubectl apply -f cli/setup/02-admin-config.yaml  # Edit first!"
echo "  kubectl apply -f cli/setup/03-user-config.yaml   # Edit first!"
echo ""
echo "=================================================="
