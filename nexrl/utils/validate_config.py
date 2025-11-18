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

from omegaconf import DictConfig


def validate_config(config: DictConfig) -> None:
    if config.trajectory_pool.check_batch_ready_function in [
        "loaded_batch_finished",
        "batch_size_reached_and_loaded_batch_finished",
    ]:
        assert (
            config.data.keep_batch_order
        ), "keep_batch_order must be true when check_batch_ready_function needs loaded_batch_finished"

    if config.data.keep_batch_order:
        assert config.trajectory_pool.check_batch_ready_function in [
            "loaded_batch_finished",
            "batch_size_reached_and_loaded_batch_finished",
        ], "check_batch_ready_function must contain loaded_batch_finished when keep_batch_order is true"

    assert (
        config.data.batch_size <= config.trajectory_pool.batch_size
    ), "batch_size in dataloader must be less than or equal to trajectory_pool.batch_size"

    assert config.data.keep_batch_order == (
        config.weight.sync_mode in ["sync", "batch-async"]
    ), "keep_batch_order must be true when sync_mode is sync or batch-async"

    assert (
        config.rollout_worker.max_response_length == config.service.inference_service.max_tokens
    ), "max_response_length must be equal to max_tokens"

    assert (
        config.rollout_worker.max_prompt_length == config.data.max_prompt_length
    ), "max_prompt_length must be equal to max_prompt_length"

    assert (
        config.service.inference_service.model_tag == config.service.train_service.model_tag
    ), "model_tag must be the same for inference and train services"

    assert (
        config.train_worker.sync_weight_path == config.weight.sync_weight_path
    ), "sync_weight_path must be the same for train and weight sync"

    assert config.train_worker.sync_weight_path != "", "sync_weight_path must not be empty"
