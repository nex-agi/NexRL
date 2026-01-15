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

# Check if weight provider is available (internal-only feature)
try:
    import north_checkpoint  # pylint: disable=unused-import

    HAS_WEIGHT_PROVIDER = True
except ImportError:
    HAS_WEIGHT_PROVIDER = False


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
        config.data.max_response_length == config.service.inference_service.max_tokens
    ), "max_response_length must be equal to max_tokens"

    assert (
        config.service.inference_service.model_tag == config.service.train_service.model_tag
    ), "model_tag must be the same for inference and train services"

    if config.trainer.get("sync_weight_path", None) is not None:
        assert (
            config.trainer.sync_weight_path == config.weight.sync_weight_path
        ), "sync_weight_path must be the same for trainer and weight sync"
        assert config.trainer.sync_weight_path != "", "sync_weight_path must not be empty"

    if "tinker" in [
        config.service.train_service.backend,
        config.service.inference_service.backend,
        config.weight.sync_method,
    ]:
        assert config.service.tinker_service.api_key != "", "api_key must not be empty"
        assert (
            config.service.train_service.backend == "tinker"
        ), "train_service.backend must be tinker"
        assert (
            config.service.inference_service.backend == "tinker"
        ), "inference_service.backend must be tinker"
        assert config.weight.sync_method == "tinker", "weight.sync_method must be tinker"

    if "weaver" in [
        config.service.train_service.backend,
        config.service.inference_service.backend,
        config.weight.sync_method,
    ]:
        weaver_service = config.service.get("weaver_service", {})
        assert weaver_service.get("api_key", "") != "", "api_key must not be empty"
        assert (
            config.service.train_service.backend == "weaver"
        ), "train_service.backend must be weaver"
        assert (
            config.service.inference_service.backend == "weaver"
        ), "inference_service.backend must be weaver"
        assert config.weight.sync_method == "weaver", "weight.sync_method must be weaver"

    # Validate weight provider availability for network sync method (self-hosted mode)
    if config.weight.sync_method == "network":
        assert HAS_WEIGHT_PROVIDER, (
            "Weight sync_method 'network' requires 'north_checkpoint' package to be installed. "
            "This is an internal-only feature. For open-source usage, please use sync_method='disk' instead."
        )

        # Check that rollout service has use_weight_provider enabled
        rollout_config = config.service.train_service.actor.get("rollout", {})
        use_weight_provider = rollout_config.get("use_weight_provider", False)
        assert use_weight_provider, (
            "When weight sync_method is 'network', rollout.use_weight_provider must be set to true. "
            "Please set service.train_service.rollout.use_weight_provider=true in your recipe."
        )
