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
Validation utilities for NexRL CLI
"""
import logging
from typing import Tuple

from . import k8s_utils

logger = logging.getLogger(__name__)


def validate_admin_setup() -> Tuple[bool, str]:
    """Validate admin environment setup

    All configuration is read from the fixed nexrl system namespace
    """
    if not k8s_utils.check_kubectl_available():
        return False, "kubectl is not available. Please install kubectl first."

    system_namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE

    if not k8s_utils.check_namespace_exists(system_namespace):
        return (
            False,
            f"NexRL system namespace '{system_namespace}' does not exist. Please run 'nexrl admin-setup' first.",
        )

    admin_config = k8s_utils.get_configmap("nexrl-admin-settings", system_namespace)
    if not admin_config:
        return (
            False,
            f"Admin configuration not found in system namespace '{system_namespace}'. Please run 'nexrl admin-setup' first.",
        )

    redis_secret = k8s_utils.get_secret("nexrl-redis-secret", system_namespace)
    if not redis_secret:
        return (
            False,
            f"Redis secret not found in system namespace '{system_namespace}'. Please run 'nexrl admin-setup' first.",
        )

    routers_config = k8s_utils.get_configmap("nexrl-routers-config", system_namespace)
    if not routers_config:
        return (
            False,
            f"Routers configuration not found in system namespace '{system_namespace}'. Please run 'nexrl admin-setup' first.",
        )

    return True, "Admin environment is properly configured."


def validate_user_dependencies() -> Tuple[bool, list]:
    """Validate user dependencies"""
    issues = []

    if not k8s_utils.check_kubectl_available():
        issues.append("kubectl is not available")

    is_valid, message = validate_admin_setup()
    if not is_valid:
        issues.append(f"Admin setup incomplete: {message}")

    return len(issues) == 0, issues


def check_redis_connectivity(
    host: str, port: int, password: str | None = None, username: str | None = None
) -> Tuple[bool, str]:
    """Test Redis connectivity"""
    try:
        import redis

        client = redis.Redis(
            host=host, port=port, password=password, username=username, socket_connect_timeout=5
        )

        client.ping()
        client.close()

        return True, "Redis connection successful"
    except ImportError:
        return False, "redis-py package not installed. Run: pip install redis"
    except Exception as e:
        return False, f"Redis connection failed: {str(e)}"


def validate_storage_path(storage_path: str) -> Tuple[bool, str]:
    """Validate storage path"""
    import os

    if not storage_path:
        return False, "Storage path is empty"

    if not os.path.isabs(storage_path):
        return False, f"Storage path must be absolute: {storage_path}"

    return True, "Storage path is valid"


def validate_docker_images(images: dict) -> Tuple[bool, list]:
    """Validate Docker image configuration"""
    issues = []

    required_images = ["train_router_image", "worker_image", "controller_image", "inference_image"]

    for key in required_images:
        if key not in images or not images[key]:
            issues.append(f"Missing required image: {key}")
        elif not ":" in images[key]:
            logger.warning(f"Image {key} does not have a tag specified: {images[key]}")

    return len(issues) == 0, issues


def check_volcanojob_support() -> Tuple[bool, str]:
    """Check if VolcanoJob CRD exists"""
    if k8s_utils.check_crd_exists("jobs.batch.volcano.sh"):
        return True, "VolcanoJob CRD is available"
    else:
        return False, "VolcanoJob CRD not found. Please install Volcano scheduler first."


def check_kubectl_permissions(namespace: str) -> Tuple[bool, list]:
    """Check required kubectl permissions"""
    issues = []

    required_permissions = [
        ("deployments", "create"),
        ("services", "create"),
        ("configmaps", "create"),
        ("secrets", "create"),
        ("volcanojobs", "create"),
        ("serviceaccounts", "create"),
        ("roles", "create"),
        ("rolebindings", "create"),
    ]

    for resource, verb in required_permissions:
        if not k8s_utils.check_kubectl_permission(resource, verb, namespace):
            issues.append(f"Missing permission: {verb} {resource} in namespace {namespace}")

    return len(issues) == 0, issues
