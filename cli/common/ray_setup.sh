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

#!/bin/bash
export PYTHONNOUSERSITE=1

# Function to wait for all ray nodes to be ready
wait_for_ray_nodes() {
    local nexrl_path=$1
    local expected_nodes=$2
    local role=$3
    local master_ip_file=$4
    local num_agent_per_worker=$5

    # Check if ray is already running on this node
    if ps -ef | grep raylet | grep -v grep > /dev/null; then
        echo "Ray is already running on this node, skipping ray start..."
        return 0
    fi

    cd $nexrl_path

    set -e
    # Set timeout limit (in seconds)
    timeout=600  # 10 minutes
    start_time=$(date +%s)

    if [ "$role" == "driver" ]; then
        echo `hostname -i` > ${master_ip_file}
        ray start --head --port=6379 --num-cpus 9 --ray-debugger-external --resources="{\"controller\": 1}" --dashboard-host 0.0.0.0 --dashboard-port 8266
    else
        while [ ! -f ${master_ip_file} ]; do
            current_time=$(date +%s)
            if [ $((current_time - start_time)) -gt $timeout ]; then
                echo "Timeout waiting for master_ip file after $timeout seconds"
                exit 1
            fi
            echo "Waiting for master_ip file to be created..."
            sleep 5
        done

        while ! nc -z `cat ${master_ip_file}` 6379; do
            current_time=$(date +%s)
            if [ $((current_time - start_time)) -gt $timeout ]; then
                echo "Timeout waiting for ray head node after $timeout seconds"
                exit 1
            fi
            echo "Waiting for RANK 0 to start ray head..."
            sleep 5
        done
        ray start --num-cpus $num_agent_per_worker --ray-debugger-external --resources="{\"agent\": $num_agent_per_worker}" --address `cat ${master_ip_file}`:6379 --block

    fi

    # only head node will enter here.
    while [ $(ray status | grep -c "node_") -lt $expected_nodes ]; do
        current_time=$(date +%s)
        if [ $((current_time - start_time)) -gt $timeout ]; then
            echo "Timeout waiting for all nodes after $timeout seconds"
            exit 1
        fi
        echo "Waiting for all nodes to be ready..."
        sleep 5
    done

    echo "all ray node[$expected_nodes] is ready"
}

run_environment_setup_script() {
    # ============================================================================
    # Source user-defined environment setup script (if provided)
    # ============================================================================
    if [[ -n "${ENVIRONMENT_SETUP_SCRIPT}" ]]; then
        echo "=========================================="
        echo "Running environment setup script"
        echo "Script: ${ENVIRONMENT_SETUP_SCRIPT}"
        echo "=========================================="

        # Convert to absolute path if relative
        if [[ "${ENVIRONMENT_SETUP_SCRIPT}" != /* ]]; then
            # Relative paths must be resolved relative to the config file's directory
            if [[ -z "${TRAIN_CONFIG}" ]]; then
                echo "ERROR: TRAIN_CONFIG environment variable must be set to resolve relative paths"
                echo "Relative path: ${ENVIRONMENT_SETUP_SCRIPT}"
                exit 1
            fi
            CONFIG_DIR="$(dirname "${TRAIN_CONFIG}")"
            ENVIRONMENT_SETUP_SCRIPT="${CONFIG_DIR}/${ENVIRONMENT_SETUP_SCRIPT}"
            echo "Resolving relative path from config directory: ${CONFIG_DIR}"
        fi

        if [[ ! -f "${ENVIRONMENT_SETUP_SCRIPT}" ]]; then
            echo "ERROR: Environment setup script not found: ${ENVIRONMENT_SETUP_SCRIPT}"
            echo "Looked at: ${ENVIRONMENT_SETUP_SCRIPT}"
            exit 1
        fi

        # Source the script to inherit all environment variables
        echo "Sourcing: ${ENVIRONMENT_SETUP_SCRIPT}"
        source "${ENVIRONMENT_SETUP_SCRIPT}"

        if [[ $? -ne 0 ]]; then
            echo "ERROR: Environment setup script failed with exit code $?"
            exit 1
        fi

        echo "✓ Environment setup completed successfully"
        echo "=========================================="
        echo ""
    else
        echo "No environment setup script specified (ENVIRONMENT_SETUP_SCRIPT not set)"
        echo "Skipping environment setup..."
        echo ""
    fi
}

# Only execute main code if script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ -z "$NEXRL_PATH" ]; then
        echo "ERROR: NEXRL_PATH environment variable is not set!"
        echo "Usage: NEXRL_PATH=/path/to/nexrl $0"
        exit 1
    fi

    if [ -z "$EXPERIMENT_PATH" ]; then
        echo "ERROR: EXPERIMENT_PATH environment variable is not set!"
        echo "Ensure the launcher passes EXPERIMENT_PATH (see nexrl_driver_cpu_job.yaml.jinja)"
        exit 1
    fi

    nexrl_path=${NEXRL_PATH}
    experiment_path=${EXPERIMENT_PATH}
    export PYTHONPATH=`pwd`:$PYTHONPATH

    master_ip_file=${experiment_path}/ray_master_ip_file

    wait_for_ray_nodes $nexrl_path $WORLD_SIZE $ROLE $master_ip_file $NUM_AGENTS_PER_WORKER

    echo ""
    sleep inf
fi
