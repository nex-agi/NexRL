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

set -x
# ============================================================================
# STEP 1: Launch API Server
# ============================================================================
if [[ -z "${EXPERIMENT_PATH}" ]]; then
    echo "ERROR: EXPERIMENT_PATH is not set."
    exit 1
fi

nexrl_path=${NEXRL_PATH}
if [[ -z "${nexrl_path}" ]]; then
    nexrl_path=`pwd`
fi

api_server_log_path=${EXPERIMENT_PATH}/api_server.log
api_server_host="0.0.0.0"
api_server_port="8000"

unset https_proxy
unset http_proxy
unset all_proxy

api_server_url="http://${api_server_host}:${api_server_port}"
echo "API server URL: ${api_server_url}"
echo "Step 1: Launching API Server..."
echo "  Host: ${api_server_host}"
echo "  Port: ${api_server_port}"
echo "  URL: ${api_server_url}"
echo ""

# Check if API server is already running
if curl -s -f "${api_server_url}/health" > /dev/null 2>&1; then
    echo "API Server is already running at ${api_server_url}"
    echo "Skipping launch. To restart, kill the existing server first."
else
    echo "Launching API server..."
    # Launch API server in background
    {
        echo "=============================================="
        echo "API Server Launch Information"
        echo "=============================================="
        echo "API URL: ${api_server_url}"
        echo "Host: ${api_server_host}"
        echo "Port: ${api_server_port}"
        echo "Time: $(date)"
        echo "=============================================="
        echo ""
        python -m nexrl.train_service_backend.api.api_server \
            --host "$api_server_host" \
            --port "$api_server_port"
    } > "${api_server_log_path}" 2>&1 &

    api_server_pid=$!
    echo "API Server started (PID: ${api_server_pid})"
    echo "Log: ${api_server_log_path}"

    # Wait for API server to be ready
    echo "Waiting for API server to be ready at ${api_server_url}..."
    for i in {1..30}; do
        if curl -s -f "${api_server_url}/health" > /dev/null 2>&1; then
            echo "✓ API Server is ready at ${api_server_url}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "✗ API Server failed to start. Check logs at ${api_server_log_path}"
            exit 1
        fi
        sleep 1
    done
fi
echo ""
sleep inf
