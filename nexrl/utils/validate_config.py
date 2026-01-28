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

from nexrl.utils.config_utils import get_actor_train_service_config

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

    # Validate inference_service has identifier
    assert config.service.inference_service.get(
        "identifier"
    ), "service.inference_service.identifier must be set"

    # Validate service.actor_train_service is specified and points to a valid service
    train_service = config.service.train_service
    actor_service_name = config.service.get("actor_train_service")
    assert actor_service_name, "service.actor_train_service must be specified"

    # Collect valid service names (keys that map to dict configs)
    # Note: With OmegaConf, nested configs are DictConfig objects, not regular dicts
    valid_service_names = [
        k for k in train_service.keys() if isinstance(train_service[k], (dict, DictConfig))
    ]
    assert (
        actor_service_name in valid_service_names
    ), f"service.actor_train_service='{actor_service_name}' must be one of: {valid_service_names}"

    # Validate each train_service group has an identifier
    train_identifiers = []
    for service_name in valid_service_names:
        service_config = train_service[service_name]
        if service_config.get("identifier"):
            train_identifiers.append(service_config.get("identifier"))
    assert (
        len(train_identifiers) > 0
    ), "At least one train_service group with identifier is required"

    if config.trainer.get("sync_weight_path", None) is not None:
        assert (
            config.trainer.sync_weight_path == config.weight.sync_weight_path
        ), "sync_weight_path must be the same for trainer and weight sync"
        assert config.trainer.sync_weight_path != "", "sync_weight_path must not be empty"

    # Get actor train service for backend checks
    actor_train_service = get_actor_train_service_config(config)

    # Check if any train service uses tinker backend
    train_backends = []
    for service_name in valid_service_names:
        service_config = train_service[service_name]
        if service_config.get("backend"):
            train_backends.append(service_config.get("backend"))

    if "tinker" in train_backends or "tinker" in [
        config.service.inference_service.backend,
        config.weight.sync_method,
    ]:
        assert config.service.tinker_service.api_key != "", "api_key must not be empty"
        assert (
            config.service.inference_service.backend == "tinker"
        ), "inference_service.backend must be tinker"
        assert config.weight.sync_method == "tinker", "weight.sync_method must be tinker"

    if "weaver" in train_backends or "weaver" in [
        config.service.inference_service.backend,
        config.weight.sync_method,
    ]:
        weaver_service = config.service.get("weaver_service", {})
        assert weaver_service.get("api_key", "") != "", "api_key must not be empty"
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

        # Check that actor train service has use_weight_provider enabled
        rollout_config = actor_train_service.actor.get("rollout", {})
        use_weight_provider = rollout_config.get("use_weight_provider", False)
        assert use_weight_provider, (
            "When weight sync_method is 'network', rollout.use_weight_provider must be set to true. "
            "Please set service.train_service.<actor>.actor.rollout.use_weight_provider=true in your recipe."
        )
