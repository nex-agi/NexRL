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

"""
Configuration management for NexRL CLI
"""

import base64
import logging
from typing import Any, Dict, Optional

from . import k8s_utils

logger = logging.getLogger(__name__)


def load_admin_config() -> Optional[Dict[str, Any]]:
    """Load admin configuration from K8s

    All configuration is read from the fixed nexrl system namespace
    """
    system_namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE

    configmap = k8s_utils.get_configmap("nexrl-admin-settings", system_namespace)
    if not configmap:
        logger.error(
            f"Failed to load admin configuration from ConfigMap in namespace '{system_namespace}'"
        )
        return None

    secret = k8s_utils.get_secret("nexrl-redis-secret", system_namespace)
    if not secret:
        logger.error(
            f"Failed to load Redis configuration from Secret in namespace '{system_namespace}'"
        )
        return None

    config = {}

    if "data" in configmap:
        config.update(configmap["data"])

    if "data" in secret:
        for key, value in secret["data"].items():
            try:
                config[key.lower()] = base64.b64decode(value).decode("utf-8")
            except Exception as e:
                logger.error(f"Failed to decode secret key {key}: {e}")

    return config


def save_admin_config(config: Dict[str, Any]) -> bool:
    """Save admin configuration to K8s

    All configuration is saved to the fixed nexrl system namespace
    """
    system_namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE

    secret_keys = ["redis_host", "redis_port", "redis_username", "redis_password"]
    secret_data = {}
    configmap_data = {}

    for key, value in config.items():
        if key.lower() in secret_keys:
            secret_data[key.upper()] = str(value)
        else:
            configmap_data[key] = str(value)

    if not k8s_utils.create_or_update_secret("nexrl-redis-secret", system_namespace, secret_data):
        logger.error(f"Failed to save Redis secret to namespace '{system_namespace}'")
        return False

    if not k8s_utils.create_or_update_configmap(
        "nexrl-admin-settings", system_namespace, configmap_data
    ):
        logger.error(f"Failed to save admin settings to namespace '{system_namespace}'")
        return False

    logger.info(f"Admin configuration saved successfully to namespace '{system_namespace}'")
    return True


def get_router_urls() -> Optional[Dict[str, str]]:
    """Get router URLs from ConfigMap

    All configuration is read from the fixed nexrl system namespace
    """
    system_namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE

    configmap = k8s_utils.get_configmap("nexrl-routers-config", system_namespace)
    if not configmap:
        logger.error(f"Failed to load router configuration from namespace '{system_namespace}'")
        return None

    if "data" not in configmap:
        logger.error("Router ConfigMap has no data")
        return None

    return configmap["data"]


def get_docker_images() -> Optional[Dict[str, str]]:
    """Get Docker image configuration"""
    config = load_admin_config()
    if not config:
        return None

    images = {
        "train_router": config.get("train_router_image", ""),
        "worker": config.get("worker_image", ""),
        "controller": config.get("controller_image", ""),
        "inference": config.get("inference_image", ""),
    }

    return images


def get_default_job_config() -> Optional[Dict[str, Any]]:
    """Get default job configuration"""
    config = load_admin_config()
    if not config:
        return None

    return {
        "namespace": k8s_utils.NEXRL_SYSTEM_NAMESPACE,
        "storage_path": config.get("storage_path", "/gpfs/data/nexrl"),
        "storage_type": config.get("storage_type", "hostPath"),
        "default_queue": config.get("default_queue", "default"),
        "default_priority_class": config.get("default_priority_class", "high-priority-job"),
        "registry": config.get("registry", "docker.io"),
    }


def update_router_urls(train_url: str, rollout_url: str) -> bool:
    """Update router URLs

    All configuration is saved to the fixed nexrl system namespace
    """
    system_namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE

    data = {
        "train_router_url": train_url,
        "rollout_router_url": rollout_url,
        "namespace": system_namespace,
    }

    return k8s_utils.create_or_update_configmap("nexrl-routers-config", system_namespace, data)


def get_storage_config() -> Optional[Dict[str, str]]:
    """Get storage configuration"""
    config = load_admin_config()
    if not config:
        return None

    return {
        "storage_type": config.get("storage_type", "hostPath"),
        "storage_path": config.get("storage_path", "/gpfs/data/nexrl"),
    }
