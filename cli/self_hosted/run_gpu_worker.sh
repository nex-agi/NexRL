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

if [ -z "$EXPERIMENT_PATH" ]; then
    echo "ERROR: EXPERIMENT_PATH environment variable is not set!"
    exit 1
fi

if [ -z "$NEXRL_PATH" ]; then
    echo "ERROR: NEXRL_PATH environment variable is not set!"
    echo "Usage: NEXRL_PATH=/path/to/nexrl $0"
    exit 1
fi

echo "NEXRL_PATH: $NEXRL_PATH"

if [ -z "$RANK" ]; then
    echo "ERROR: RANK environment variable is not set!"
    echo "Usage: RANK=0 $0"
    exit 1
fi

echo "RANK: $RANK"


cd $NEXRL_PATH
export PYTHONPATH=`pwd`:$PYTHONPATH
unset https_proxy
unset http_proxy
unset all_proxy;

identifier_label=${IDENTIFIER:-default}
nrank_label=${RANK:-0}
log_path=${EXPERIMENT_PATH}/workers-${identifier_label}-nrank${nrank_label}.log

cd $NEXRL_PATH && \
torchrun --nproc-per-node 8  --node_rank=$RANK --nnodes=${WORLD_SIZE} --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT}  \
    -m nexrl.train_service_backend.distributed.worker_process \
    --api-server-url http://${API_SERVER_URL}:8000 \
    --backend fsdp \
    --identifier $IDENTIFIER \
    --dispatch-mode scatter \
    > "${log_path}" 2>&1 &

echo ""
sleep inf
