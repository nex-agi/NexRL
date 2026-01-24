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

import time
from contextlib import contextmanager
from typing import Any

import torch

from ..train_service_client import TrainServiceClient


class MockTrainServiceClient(TrainServiceClient):
    def initialize_worker(
        self,
        config_path: str | None = None,
        config_dict: dict[str, Any] | None = None,
        role: str = "actor",
        world_size: int | None = None,
        zmq_base_port: int | None = None,
        dispatch_mode: str | None = None,
    ) -> dict[str, Any]:
        """Mock implementation of initialize_worker"""
        time.sleep(0.1)
        return {"status": "success", "message": "Mock worker initialized"}

    def init_model(self) -> dict[str, Any]:
        """Mock implementation of init_model"""
        time.sleep(0.1)
        return {"status": "success", "message": "Mock model initialized"}

    def update_actor(self, batch: dict):
        time.sleep(1)
        return {"meta_info": {"metrics": {"success": True}}}

    def update_actor_with_distillation(self, batch: dict):
        time.sleep(1)
        return {"meta_info": {"metrics": {"success": True}}}

    def compute_log_prob(self, batch: dict) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "batch": {
                "old_log_probs": torch.tensor(
                    [[0.0]] * batch["meta_info"]["batch_size"], dtype=torch.float32
                )
            }
        }

    @contextmanager
    def actor_context(self):
        """Context manager for actor operations."""
        yield

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        saved_fully_shared_ckpt: bool = True,
        save_weight_only: bool = False,
        remove_previous_ckpt: bool = True,
    ) -> dict[str, Any]:
        time.sleep(1)
        return {"success_mock": True}

    def load_checkpoint(
        self, path: str, del_local_after_load: bool = True, load_weight_only: bool = False
    ) -> dict[str, Any]:
        time.sleep(1)
        return {"success_mock": True}
