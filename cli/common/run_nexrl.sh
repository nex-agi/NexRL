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

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If this pod is marked as agent worker, just bring up ray and wait.
if [[ "${IS_AGENT_WORKER}" == "true" ]]; then
    echo "Agent worker pod detected; starting ray standby via ray_setup.sh"
    exec "${SCRIPT_DIR}/ray_setup.sh"
fi

if [[ -z "${TRAIN_CONFIG}" ]]; then
    echo "ERROR: TRAIN_CONFIG is not set. Please provide path to a config file."
    exit 1
fi

if [[ ! -f "${TRAIN_CONFIG}" ]]; then
    echo "ERROR: TRAIN_CONFIG file not found: ${TRAIN_CONFIG}"
    exit 1
fi

# Source the wait_for_ray_nodes function from ray_setup.sh
source "${SCRIPT_DIR}/ray_setup.sh"

# Driver pod needs to set up Ray cluster before running NexRL
echo "Driver pod detected; setting up Ray cluster before running NexRL"
if [ -z "$NEXRL_PATH" ]; then
    echo "ERROR: NEXRL_PATH environment variable is not set!"
    exit 1
fi

if [ -z "$EXPERIMENT_PATH" ]; then
    echo "ERROR: EXPERIMENT_PATH environment variable is not set!"
    exit 1
fi

nexrl_path=${NEXRL_PATH}
experiment_path=${EXPERIMENT_PATH}
master_ip_file=${experiment_path}/ray_master_ip_file

wait_for_ray_nodes $nexrl_path $WORLD_SIZE $ROLE $master_ip_file $NUM_AGENTS_PER_WORKER
run_environment_setup_script

CONFIG_DIR="$(dirname "${TRAIN_CONFIG}")"
CONFIG_BASENAME="$(basename "${TRAIN_CONFIG}")"
CONFIG_NAME="${CONFIG_BASENAME%.*}"

log_path=${EXPERIMENT_PATH}/nexrl.log


export PYTHONPATH=`pwd`:$PYTHONPATH
echo "Launching NexRL with config: ${TRAIN_CONFIG}"

if [[ -n "${DEBUG_HYDRA_OVERRIDES}" ]]; then
    echo "Applying debug overrides: ${DEBUG_HYDRA_OVERRIDES}"
    python -m nexrl.main --config-path="${CONFIG_DIR}" --config-name="${CONFIG_NAME}" ${DEBUG_HYDRA_OVERRIDES} 2>&1 | tee ${log_path}
else
    python -m nexrl.main --config-path="${CONFIG_DIR}" --config-name="${CONFIG_NAME}" 2>&1 | tee ${log_path}
fi
