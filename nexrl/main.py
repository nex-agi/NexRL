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

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from .controller import NexRLController
from .utils.config_utils import dump_resolved_config, migrate_legacy_config
from .utils.logging_utils import set_logging_basic_config
from .utils.validate_config import validate_config

logger = logging.getLogger(__name__)


def main_task(config: DictConfig):
    """Main task that can run as Ray remote function or locally"""
    # Print initial config for debugging
    from pprint import pprint

    OmegaConf.resolve(config)
    migrate_legacy_config(config)

    logger.info("Starting NexRL with configuration:")
    pprint(OmegaConf.to_container(config, resolve=True))

    # Dump resolved config to experiment path for reproducibility
    dump_resolved_config(config)

    controller = NexRLController(config)
    controller.run()


@hydra.main(config_path="config", config_name="rl_train", version_base=None)
def main(config: DictConfig):
    """Main entry point supporting both local and Ray launch modes"""
    set_logging_basic_config()

    # 1. Try TRAIN_CONFIG environment variable (most reliable, set by launch scripts)
    try:
        config_file_path = os.environ["TRAIN_CONFIG"]
        logger.info(f"Using config file path from TRAIN_CONFIG: {config_file_path}")
    # Fail hard if we couldn't determine the config file path
    except Exception as exc:
        raise RuntimeError(
            "Cannot determine config file path. "
            "Ensure TRAIN_CONFIG environment variable is set or run via Hydra."
        ) from exc

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    # Store in config for downstream use
    # Temporarily disable struct mode to allow adding new attribute
    OmegaConf.set_struct(config, False)
    config._config_file_path = config_file_path  # pylint: disable=protected-access
    OmegaConf.set_struct(config, True)
    logger.info(f"Starting NexRL in {config.launch_mode} mode")

    validate_config(config)

    if config.launch_mode == "local":
        # Run directly in local mode
        main_task(config)
    elif config.launch_mode == "ray":
        import ray

        main_task(config)
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown completed")
    else:
        raise ValueError(
            f"Unsupported launch mode: {config.launch_mode}. Supported modes: 'local', 'ray'"
        )
    logger.info("NexRL training completed")
    logger.info(f"Experiment log path: {os.environ.get('EXPERIMENT_PATH', 'not found')}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
