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

Supports multiple configuration sources with fallback chain:
1. Kubernetes ConfigMap (production, multi-user)
2. Environment variables (development, testing)
3. Sensible defaults (quick start, demos)
"""

import base64
import logging
import os
from typing import Any, Dict, Optional

from . import k8s_utils

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    # Docker images - users should override these!
    "worker_image": "ghcr.io/nex-agi/nexrl-open-source:v1.0.1",
    "controller_image": "ghcr.io/nex-agi/nexrl-open-source:v1.0.1",
    "inference_image": "lmsysorg/sglang:v0.5.4.post2",
    # Storage - defaults to /tmp for quick testing
    "storage_path": "/tmp/nexrl",
    "storage_type": "hostPath",
    # Kubernetes settings
    "default_queue": "default",
    "default_priority_class": "",
    "registry": "ghcr.io",
}


def load_admin_config() -> Optional[Dict[str, Any]]:
    """Load admin configuration with fallback chain

    Priority (highest to lowest):
    1. Kubernetes ConfigMap (production)
    2. Environment variables (dev/testing)
    3. Built-in defaults (quick start)

    Returns:
        Configuration dictionary, or None if critical config is missing
    """
    system_namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE

    # Try to load from Kubernetes ConfigMap first
    configmap = k8s_utils.get_configmap("nexrl-admin-settings", system_namespace)
    if configmap and "data" in configmap:
        logger.info(
            f"📦 Using admin config from Kubernetes ConfigMap in namespace '{system_namespace}'"
        )
        return configmap["data"]

    # Fall back to environment variables + defaults
    logger.info(f"ℹ️  ConfigMap 'nexrl-admin-settings' not found in namespace '{system_namespace}'")
    logger.info("✓ Using environment variables and defaults for configuration")
    logger.info("   For production: kubectl apply -f cli/setup/02-admin-config.yaml")

    config = {
        "storage_path": os.getenv("NEXRL_STORAGE_PATH", DEFAULT_CONFIG["storage_path"]),
        "storage_type": os.getenv("NEXRL_STORAGE_TYPE", DEFAULT_CONFIG["storage_type"]),
        "worker_image": os.getenv("NEXRL_WORKER_IMAGE", DEFAULT_CONFIG["worker_image"]),
        "controller_image": os.getenv("NEXRL_CONTROLLER_IMAGE", DEFAULT_CONFIG["controller_image"]),
        "inference_image": os.getenv("NEXRL_INFERENCE_IMAGE", DEFAULT_CONFIG["inference_image"]),
        "default_queue": os.getenv("NEXRL_QUEUE", DEFAULT_CONFIG["default_queue"]),
        "default_priority_class": os.getenv(
            "NEXRL_PRIORITY_CLASS", DEFAULT_CONFIG["default_priority_class"]
        ),
        "registry": os.getenv("NEXRL_REGISTRY", DEFAULT_CONFIG["registry"]),
    }

    # Warn about using defaults
    if config["storage_path"] == DEFAULT_CONFIG["storage_path"]:
        logger.warning(
            f"⚠️  Using default storage path '{DEFAULT_CONFIG['storage_path']}' (not suitable for production!)\n"
            "   → Set NEXRL_STORAGE_PATH env var for testing\n"
            "   → Or apply ConfigMap for production: kubectl apply -f cli/setup/02-admin-config.yaml"
        )

    return config


def save_admin_config(config: Dict[str, Any]) -> bool:
    """Save admin configuration to K8s ConfigMap

    All configuration is saved to the fixed nexrl system namespace
    """
    system_namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE

    if not k8s_utils.create_or_update_configmap("nexrl-admin-settings", system_namespace, config):
        logger.error(f"Failed to save admin settings to namespace '{system_namespace}'")
        return False

    logger.info(f"Admin configuration saved successfully to namespace '{system_namespace}'")
    return True


def get_docker_images() -> Optional[Dict[str, str]]:
    """Get Docker image configuration"""
    config = load_admin_config()
    if not config:
        return None

    images = {
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


def get_storage_config() -> Optional[Dict[str, str]]:
    """Get storage configuration"""
    config = load_admin_config()
    if not config:
        return None

    return {
        "storage_type": config.get("storage_type", "hostPath"),
        "storage_path": config.get("storage_path", "/gpfs/data/nexrl"),
    }


def load_user_secrets(namespace: str, user_id: str) -> Dict[str, str]:
    """Load user secrets with fallback chain

    Priority (highest to lowest):
    1. Environment variables (dev/testing - highest priority)
    2. Kubernetes Secret (production - fallback)

    Returns a dictionary with WandB and optional service API keys
    """
    secrets = {
        "WANDB_KEY": "",
        "WANDB_HOST": "",
        "TINKER_API_KEY": "",
        "TINKER_BASE_URL": "",
        "WEAVER_API_KEY": "",
        "WEAVER_BASE_URL": "",
    }

    # First, try to load from Kubernetes Secret (lower priority)
    wandb_secret_name = f"wandb-secret"
    secret = k8s_utils.get_secret(wandb_secret_name, namespace)

    if secret and "data" in secret:
        logger.info(f"📦 Using WandB config from Kubernetes Secret")
        for key, value in secret["data"].items():
            try:
                decoded = base64.b64decode(value).decode("utf-8")
                # Map secret keys to env var names
                if key == "api_key":
                    secrets["WANDB_KEY"] = decoded
                elif key == "host":
                    secrets["WANDB_HOST"] = decoded
            except Exception as e:
                logger.debug(f"Failed to decode wandb secret key {key}: {e}")

    # Then, override with environment variables if set (higher priority)
    wandb_key_env = os.getenv("WANDB_KEY", "")
    wandb_host_env = os.getenv("WANDB_HOST", "")

    if wandb_key_env:
        logger.info("✓ Using WANDB_KEY from environment variable")
        secrets["WANDB_KEY"] = wandb_key_env

    if wandb_host_env:
        secrets["WANDB_HOST"] = wandb_host_env

    # Summary message
    if not secrets["WANDB_KEY"]:
        logger.info("ℹ️  WandB: Not configured (set WANDB_KEY env var to enable)")
    else:
        logger.info("✓ WandB: Configured and enabled")

    # Optional: Load training service secrets (for training-service mode)
    service_secret_name = f"nexrl-service-secrets-{user_id}"
    service_secret = k8s_utils.get_secret(service_secret_name, namespace)

    if service_secret and "data" in service_secret:
        logger.info(f"📦 Using training service config from Kubernetes Secret")
        for key, value in service_secret["data"].items():
            try:
                decoded = base64.b64decode(value).decode("utf-8")
                # Map secret keys to env var names
                key_upper = key.upper()
                if key_upper in secrets:
                    secrets[key_upper] = decoded
            except Exception as e:
                logger.debug(f"Failed to decode service secret key {key}: {e}")

    # Override with environment variables if set (higher priority)
    tinker_api_key_env = os.getenv("TINKER_API_KEY", "")
    tinker_base_url_env = os.getenv("TINKER_BASE_URL", "")
    weaver_api_key_env = os.getenv("WEAVER_API_KEY", "")
    weaver_base_url_env = os.getenv("WEAVER_BASE_URL", "")

    if tinker_api_key_env:
        logger.info("✓ Using TINKER_API_KEY from environment variable")
        secrets["TINKER_API_KEY"] = tinker_api_key_env
    if tinker_base_url_env:
        secrets["TINKER_BASE_URL"] = tinker_base_url_env
    if weaver_api_key_env:
        logger.info("✓ Using WEAVER_API_KEY from environment variable")
        secrets["WEAVER_API_KEY"] = weaver_api_key_env
    if weaver_base_url_env:
        secrets["WEAVER_BASE_URL"] = weaver_base_url_env

    # Summary for training service APIs
    configured_services = []
    if secrets.get("TINKER_API_KEY"):
        configured_services.append("Tinker")
    if secrets.get("WEAVER_API_KEY"):
        configured_services.append("Weaver")

    if configured_services:
        logger.info(f"✓ Training Services: {', '.join(configured_services)} configured")
    else:
        logger.info("ℹ️  Training Services: Not configured (optional)")

    return secrets
