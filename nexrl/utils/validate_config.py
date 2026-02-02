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

    # Validate inference_service has identifier (with backward compatibility for model_tag)
    import warnings

    identifier = config.service.inference_service.get("identifier")
    model_tag = config.service.inference_service.get("model_tag")

    if not identifier and not model_tag:
        raise ValueError(
            "service.inference_service must have either 'identifier' or 'model_tag' field"
        )

    if not identifier and model_tag:
        warnings.warn(
            "Using deprecated 'model_tag' field. Please rename to 'identifier'. "
            "See migration guide in docs/developer-guide/09-recipes/.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Validate train_service structure (with backward compatibility)
    train_service = config.service.train_service

    # Check if this is old flat structure
    is_flat_structure = "backend" in train_service or "url" in train_service

    if is_flat_structure:
        warnings.warn(
            "Using deprecated flat train_service structure. "
            "Please migrate to nested structure with explicit service names and roles. "
            "See migration guide in docs/developer-guide/09-recipes/.",
            DeprecationWarning,
            stacklevel=2,
        )
        # In old structure, validation passes (single implicit actor service)
        valid_service_names = []
        train_identifiers = [train_service.get("identifier", "default")]
    else:
        # New nested structure validation
        actor_services = [
            k
            for k, v in train_service.items()
            if isinstance(v, (dict, DictConfig)) and v.get("role") == "actor"
        ]

        # Collect valid service names (keys that map to dict configs)
        valid_service_names = [
            k for k in train_service.keys() if isinstance(train_service[k], (dict, DictConfig))
        ]

        # Count services without role field
        services_without_role = [k for k in valid_service_names if not train_service[k].get("role")]

        # If no actor role found, but only one service exists, assume it's actor
        if len(actor_services) == 0 and len(valid_service_names) == 1:
            warnings.warn(
                f"Train service '{valid_service_names[0]}' is missing 'role' field. "
                "Assuming role='actor'. Please add explicit role='actor' field. "
                "See migration guide in docs/developer-guide/09-recipes/.",
                DeprecationWarning,
                stacklevel=2,
            )
        elif len(actor_services) == 0 and len(services_without_role) > 0:
            # Multiple services without role - cannot determine which is actor
            raise ValueError(
                f"Multiple train services found without 'role' field: {services_without_role}. "
                "Cannot determine which is 'actor'. Please add explicit 'role' field to each service."
            )
        elif len(actor_services) != 1:
            raise ValueError(
                f"Exactly one train_service must have role='actor', found {len(actor_services)}"
            )

        # Validate each train_service group has an identifier (with backward compatibility for model_tag)
        train_identifiers = []
        for service_name in valid_service_names:
            service_config = train_service[service_name]
            svc_identifier = service_config.get("identifier")
            svc_model_tag = service_config.get("model_tag")

            if not svc_identifier and svc_model_tag:
                warnings.warn(
                    f"Train service '{service_name}' uses deprecated 'model_tag' field. "
                    "Please rename to 'identifier'. "
                    "See migration guide in docs/developer-guide/09-recipes/.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                train_identifiers.append(svc_model_tag)
            elif svc_identifier:
                train_identifiers.append(svc_identifier)

        assert (
            len(train_identifiers) > 0
        ), "At least one train_service group with identifier (or model_tag) is required"

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
