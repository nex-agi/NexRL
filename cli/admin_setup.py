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
Admin setup command for NexRL CLI

Initializes the cluster environment for NexRL training.
"""

import logging
import sys
from typing import Any, Dict

import click

from .utils import config_manager, k8s_utils, validation, yaml_builder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Check all dependencies

    All resources are deployed to the fixed nexrl namespace
    """
    logger.info("Checking dependencies...")

    if not k8s_utils.check_kubectl_available():
        logger.error("kubectl is not available. Please install kubectl first.")
        return False
    logger.info("✓ kubectl available")

    is_valid, message = validation.check_volcanojob_support()
    if not is_valid:
        logger.error(message)
        return False
    logger.info(f"✓ {message}")

    namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE
    if not k8s_utils.check_namespace_exists(namespace):
        logger.warning(f"Namespace '{namespace}' does not exist")
        if click.confirm(f"Create namespace '{namespace}'?", default=True):
            if k8s_utils.create_namespace(namespace):
                logger.info(f"✓ Created namespace '{namespace}'")
            else:
                logger.error(f"Failed to create namespace '{namespace}'")
                return False
        else:
            logger.error("Namespace is required to continue")
            return False
    else:
        logger.info(f"✓ Namespace '{namespace}' exists")

    is_valid, issues = validation.check_kubectl_permissions(namespace)
    if not is_valid:
        logger.error(f"Insufficient kubectl permissions for namespace '{namespace}':")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    logger.info(f"✓ Namespace permissions verified")

    return True


def collect_configuration() -> Dict[str, Any]:
    """Interactively collect configuration"""
    logger.info("\n=== Configuration Collection ===")

    config = {}

    logger.info("\n--- Redis Configuration ---")
    config["redis_host"] = click.prompt("Redis Host", type=str)
    config["redis_port"] = click.prompt("Redis Port", default="6379", type=str)
    config["redis_username"] = click.prompt("Redis Username", default="", type=str)
    config["redis_password"] = click.prompt("Redis Password", default="", hide_input=True, type=str)

    logger.info("Testing Redis connection...")
    is_connected, message = validation.check_redis_connectivity(
        config["redis_host"],
        int(config["redis_port"]),
        config["redis_password"],
        config["redis_username"],
    )
    if is_connected:
        logger.info(f"✓ {message}")
    else:
        logger.warning(f"⚠ {message}")
        if not click.confirm("Redis connection failed. Continue anyway?", default=False):
            sys.exit(1)

    logger.info("\n--- Storage Configuration ---")
    config["storage_type"] = "hostPath"
    config["storage_path"] = click.prompt("Storage path (hostPath)", default="/nfs", type=str)

    logger.info("\n--- Docker Image Configuration ---")
    config["registry"] = click.prompt("Image registry", default="docker.io", type=str)

    config["train_router_image"] = click.prompt(
        "Train Router image", default=f"{config['registry']}/nexagi/nexrl:latest", type=str
    )

    config["worker_image"] = click.prompt(
        "Training Workers image", default=f"{config['registry']}/nexagi/nexrl:latest", type=str
    )

    config["controller_image"] = click.prompt(
        "NexRL Controller image", default=f"{config['registry']}/nexagi/nexrl:latest", type=str
    )

    config["inference_image"] = click.prompt(
        "Inference service image", default=f"{config['registry']}/lmsysorg/sglang:latest", type=str
    )

    config["rollout_router_image"] = click.prompt(
        "Rollout Router image", default=f"{config['registry']}/lmsysorg/sglang:latest", type=str
    )

    logger.info("\n--- Default Resource Configuration ---")
    config["default_queue"] = click.prompt("Default queue", default="default", type=str)

    config["default_priority_class"] = click.prompt(
        "Default priority class", default="high-priority-job", type=str
    )

    return config


def deploy_infrastructure(config: Dict[str, Any]) -> bool:
    """Deploy infrastructure components

    All resources are deployed to the fixed nexrl namespace
    """
    logger.info("\n=== Deploying Infrastructure ===")

    namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE

    try:
        logger.info("1. Creating Volcano Queue...")
        queue_name = config.get("default_queue", "default")

        if k8s_utils.check_queue_exists(queue_name):
            logger.info(f"✓ Queue '{queue_name}' already exists")
        else:
            queue_yaml = yaml_builder.render_volcano_queue(
                queue_name=queue_name, weight=1, reclaimable=True
            )
            # Queue is cluster-scoped, so no namespace is needed
            if not k8s_utils.apply_yaml(queue_yaml):
                logger.error(f"Failed to create Queue '{queue_name}'")
                return False
            logger.info(f"✓ Queue '{queue_name}' created")

        logger.info(f"2. Creating Redis Secret (namespace: {namespace})...")
        redis_secret_yaml = yaml_builder.render_redis_secret(
            namespace=namespace,
            redis_config={
                "host": config["redis_host"],
                "port": config["redis_port"],
                "username": config["redis_username"],
                "password": config["redis_password"],
            },
        )
        if not k8s_utils.apply_yaml(redis_secret_yaml, namespace):
            logger.error(f"Failed to create Redis Secret (namespace: {namespace})")
            return False
        logger.info(f"✓ Redis Secret created (namespace: {namespace})")

        logger.info(f"3. Creating ServiceAccount and RBAC (namespace: {namespace})...")
        rbac_yaml = yaml_builder.render_sglang_router_rbac(namespace)
        if not k8s_utils.apply_yaml(rbac_yaml, namespace):
            logger.error(f"Failed to create RBAC (namespace: {namespace})")
            return False
        logger.info(f"✓ RBAC created (namespace: {namespace})")

        logger.info(f"4. Deploying Train Router (namespace: {namespace})...")
        train_router_deployment_yaml = yaml_builder.render_train_router_deployment(
            namespace=namespace, image=config["train_router_image"]
        )
        if not k8s_utils.apply_yaml(train_router_deployment_yaml, namespace):
            logger.error(f"Failed to deploy Train Router (namespace: {namespace})")
            return False

        train_router_service_yaml = yaml_builder.render_train_router_service(namespace)
        if not k8s_utils.apply_yaml(train_router_service_yaml, namespace):
            logger.error(f"Failed to create Train Router Service (namespace: {namespace})")
            return False

        logger.info("Waiting for Train Router to be ready...")
        if not k8s_utils.wait_for_deployment("train-router", namespace, timeout=1800):
            logger.error("Train Router failed to become ready")
            return False
        logger.info(f"✓ Train Router deployed (namespace: {namespace})")

        logger.info(f"5. Deploying Rollout Router (namespace: {namespace})...")
        rollout_router_deployment_yaml = yaml_builder.render_rollout_router_deployment(
            namespace=namespace,
            served_model_name="nexrl",
            image=config.get("rollout_router_image", "lmsysorg/sglang:latest"),
        )
        if not k8s_utils.apply_yaml(rollout_router_deployment_yaml, namespace):
            logger.error(f"Failed to deploy Rollout Router (namespace: {namespace})")
            return False

        rollout_router_service_yaml = yaml_builder.render_rollout_router_service(namespace)
        if not k8s_utils.apply_yaml(rollout_router_service_yaml, namespace):
            logger.error(f"Failed to create Rollout Router Service (namespace: {namespace})")
            return False

        logger.info("Waiting for Rollout Router to be ready...")
        if not k8s_utils.wait_for_deployment("rollout-router", namespace, timeout=1800):
            logger.error("Rollout Router failed to become ready")
            return False
        logger.info(f"✓ Rollout Router deployed (namespace: {namespace})")

        logger.info(f"6. Creating Routers ConfigMap (namespace: {namespace})...")
        routers_configmap_yaml = yaml_builder.render_routers_configmap(namespace)
        if not k8s_utils.apply_yaml(routers_configmap_yaml, namespace):
            logger.error(f"Failed to create Routers ConfigMap (namespace: {namespace})")
            return False
        logger.info(f"✓ Routers ConfigMap created (namespace: {namespace})")

        logger.info(f"7. Saving admin configuration (namespace: {namespace})...")
        if not config_manager.save_admin_config(config):
            logger.error(f"Failed to save admin configuration (namespace: {namespace})")
            return False
        logger.info(f"✓ Admin configuration saved (namespace: {namespace})")

        return True

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


def verify_deployment() -> bool:
    """Verify deployment

    All resources are deployed to the fixed nexrl namespace
    """
    logger.info("\n=== Verifying Deployment ===")

    namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE

    # Check Volcano Queue (cluster-scoped)
    logger.info("\nChecking Volcano Queue...")
    admin_config = config_manager.load_admin_config()
    if admin_config:
        queue_name = admin_config.get("default_queue", "default")
        if k8s_utils.check_queue_exists(queue_name):
            logger.info(f"✓ queue/{queue_name} (cluster-scoped)")
        else:
            logger.error(f"✗ queue/{queue_name} not found")
            return False

    components = [
        ("configmap", "nexrl-routers-config"),
        ("configmap", "nexrl-admin-settings"),
        ("secret", "nexrl-redis-secret"),
        ("deployment", "train-router"),
        ("deployment", "rollout-router"),
        ("service", "train-router-svc"),
        ("service", "rollout-router-svc"),
    ]

    all_ok = True

    logger.info(f"\nChecking components (namespace: {namespace})...")
    for resource_type, name in components:
        try:
            logger.info(f"✓ {resource_type}/{name} (namespace: {namespace})")
        except Exception as e:
            logger.error(f"Failed to check {resource_type}/{name}: {e}")
            all_ok = False

    if all_ok:
        logger.info("\n✓ All components deployed successfully!")
        logger.info(f"\nNexRL namespace: {namespace}")

        router_urls = config_manager.get_router_urls()
        if router_urls:
            logger.info("\n=== Access Information ===")
            logger.info(f"Train Router URL: {router_urls.get('train_router_url')}")
            logger.info(f"Rollout Router URL: {router_urls.get('rollout_router_url')}")
    else:
        logger.error("\nSome components failed to deploy, please check logs")

    return all_ok


@click.command()
@click.option("--non-interactive", is_flag=True, help="Non-interactive mode (use defaults)")
def admin_setup(non_interactive: bool):
    """Initialize NexRL cluster environment

    All components will be deployed to the fixed 'nexrl' namespace
    """

    logger.info("=" * 60)
    logger.info("NexRL Admin Setup")
    logger.info("=" * 60)

    if not check_dependencies():
        logger.error("\nDependency check failed. Please resolve the issues and try again")
        sys.exit(1)

    logger.info("\n✓ All dependency checks passed")

    is_valid, message = validation.validate_admin_setup()
    if is_valid:
        logger.info(f"\n✓ {message}")
        if not click.confirm("Existing configuration detected. Reconfigure?", default=False):
            logger.info("Keeping existing configuration, exiting")
            sys.exit(0)

    if non_interactive:
        logger.error("Non-interactive mode not yet implemented")
        sys.exit(1)
    else:
        config = collect_configuration()

    logger.info("\n=== Configuration Summary ===")
    for key, value in config.items():
        if "password" in key.lower():
            logger.info(f"{key}: {'*' * len(str(value))}")
        else:
            logger.info(f"{key}: {value}")

    if not click.confirm("\nConfirm configuration and start deployment?", default=True):
        logger.info("Deployment cancelled")
        sys.exit(0)

    if not deploy_infrastructure(config):
        logger.error("\nDeployment failed")
        sys.exit(1)

    if not verify_deployment():
        logger.error("\nVerification failed")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("✓ NexRL cluster environment initialized successfully!")
    logger.info("=" * 60)
    logger.info("\nYou can now use 'nexrl launch' to start training jobs")


if __name__ == "__main__":
    admin_setup()
