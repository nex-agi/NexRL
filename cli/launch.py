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
Launch command for NexRL CLI

Launches NexRL training jobs on Kubernetes.
"""

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import yaml  # type: ignore

from .utils import config_manager, k8s_utils, validation, yaml_builder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Local configuration file paths
NEXRL_CONFIG_DIR = Path.home() / ".nexrl"
NEXRL_CONFIG_FILE = NEXRL_CONFIG_DIR / "config.yaml"


def load_local_config() -> dict[str, Any]:
    """Load local configuration file"""
    if not NEXRL_CONFIG_FILE.exists():
        return {}

    try:
        with open(NEXRL_CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f) or {}
        return config
    except Exception as e:
        logger.debug(f"Failed to load local config: {e}")
        return {}


def save_local_config(config: dict[str, Any]) -> bool:
    """Save local configuration file"""
    try:
        NEXRL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(NEXRL_CONFIG_FILE, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.debug(f"Config saved to {NEXRL_CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False


def ensure_user_config() -> dict[str, Any]:
    """Ensure user configuration exists, create if missing"""
    config = load_local_config()

    if config.get("user_id") and config.get("user_name"):
        return config

    # First-time setup
    click.echo("\n" + "=" * 60)
    click.echo("Welcome to NexRL!")
    click.echo("=" * 60)
    click.echo("\nFirst-time setup: Please provide user information")

    default_name = os.getenv("USER", "user")
    user_name = click.prompt("Enter your username", default=default_name)
    user_id = str(uuid.uuid4())

    config.update(
        {
            "user_id": user_id,
            "user_name": user_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    if save_local_config(config):
        click.echo(f"\n✓ User configuration created")
        click.echo(f"  Username: {user_name}")
        click.echo(f"  User ID: {user_id}")
        click.echo(f"\nTip: Run 'nexrl init' to configure WandB")
    else:
        logger.error("Failed to save configuration")
        sys.exit(1)

    return config


def load_job_config(job_path: str) -> dict[str, Any]:
    """Read job.yaml configuration file and resolve all interpolation expressions

    Args:
        job_path: Job directory path containing job.yaml

    Returns:
        Configuration dictionary with all interpolations resolved

    Raises:
        FileNotFoundError: If job.yaml not found
        Exception: If configuration loading fails
    """
    from omegaconf import OmegaConf

    # Convert to absolute path
    job_path = os.path.abspath(job_path)

    # job.yaml is in job_path directory
    job_yaml_path = os.path.join(job_path, "rl_train.yaml")
    if not os.path.exists(job_yaml_path):
        raise FileNotFoundError(f"rl_train.yaml not found in {job_path}")

    logger.debug(f"Loading job config from {job_yaml_path}")

    # Read configuration using yaml
    with open(job_yaml_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Remove defaults section as it references paths inside the container
    # defaults will be handled by Hydra inside the container
    if "defaults" in raw_config:
        del raw_config["defaults"]

    # Use OmegaConf to parse interpolation expressions
    # Including ${oc.env:USER}, ${shared.model_path}, etc.
    conf = OmegaConf.create(raw_config)

    # Resolve all interpolation expressions and convert to plain dictionary
    # resolve=True will recursively resolve all ${...} references
    config = OmegaConf.to_container(conf, resolve=True)

    logger.debug("Job configuration loaded and resolved successfully")
    return config


def check_launch_dependencies() -> bool:
    """Check dependencies for launching jobs"""
    logger.info("Checking dependencies...")

    if not k8s_utils.check_kubectl_available():
        logger.error("kubectl is not available")
        return False

    is_valid, message = validation.validate_admin_setup()
    if not is_valid:
        logger.error(message)
        return False

    is_valid, issues = validation.validate_user_dependencies()
    if not is_valid:
        logger.error("User dependency validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False

    logger.info("✓ Dependencies validated")
    return True


def generate_identifier(job_name: str) -> str:
    """Generate job identifier - directly use job_name"""
    return job_name


def load_user_secrets(namespace: str, user_id: str) -> dict[str, Any]:
    """Load user configuration from Kubernetes Secrets"""
    import base64
    import json

    user_config = {
        "wandb_enabled": False,
        "wandb_api_key": "",
        "wandb_host": "https://api.wandb.ai",
    }

    # Load WandB Secret
    wandb_secret_name = f"nexrl-wandb-{user_id}"
    try:
        result = k8s_utils.run_kubectl_command(
            ["get", "secret", wandb_secret_name, "-n", namespace, "-o", "jsonpath={.data}"],
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            if "enabled" in data:
                enabled = base64.b64decode(data["enabled"]).decode("utf-8")
                user_config["wandb_enabled"] = enabled == "true"
            if "api_key" in data:
                user_config["wandb_api_key"] = base64.b64decode(data["api_key"]).decode("utf-8")
            if "host" in data:
                user_config["wandb_host"] = base64.b64decode(data["host"]).decode("utf-8")
            logger.debug(
                f"WandB config: {'enabled' if user_config['wandb_enabled'] else 'disabled'}"
            )
    except Exception as e:
        logger.debug(f"WandB secret not found: {e}")

    return user_config


def launch_workers(
    job_name: str,
    namespace: str,
    identifier: str,
    job_path: str,
    composed_config: dict[str, Any],
    admin_config: dict[str, Any],
) -> bool:
    """Launch training workers"""
    logger.info("Launching training workers...")

    try:
        # Extract parallel strategy from composed configuration
        train_parallel = composed_config.get("train_parallel", {})
        world_size = train_parallel.get("world_size", 1)
        gpus_per_pod = train_parallel.get("gpus_per_pod", 8)
        memory_per_gpu = train_parallel.get("memory_per_gpu", 200)

        # Calculate worker memory
        worker_memory = gpus_per_pod * memory_per_gpu

        logger.info(f"  World Size: {world_size}")
        logger.info(f"  GPUs per Pod: {gpus_per_pod}")
        logger.info(f"  Memory per Pod: {worker_memory}Gi")

        router_urls = config_manager.get_router_urls()
        if not router_urls:
            logger.error("Failed to retrieve router URLs")
            return False

        train_router_url = router_urls["train_router_url"]

        # Use job.yaml under job_path as configuration file path
        train_worker_config_file = os.path.join(job_path, "job.yaml")

        worker_image = admin_config.get("worker_image")
        storage_path = admin_config.get("storage_path")
        assert isinstance(worker_image, str), "worker_image must be configured in admin settings"
        assert isinstance(storage_path, str), "storage_path must be configured in admin settings"

        worker_yaml = yaml_builder.render_worker_volcanojob(
            job_name=job_name,
            namespace=namespace,
            identifier=identifier,
            worker_image=worker_image,
            train_router_url=train_router_url,
            worker_config_path=train_worker_config_file,
            storage_path=storage_path,
            world_size=world_size,
            gpus_per_pod=gpus_per_pod,
            WORKER_MEMORY=f"{worker_memory}Gi",
            QUEUE=admin_config.get("default_queue", "default"),
            PRIORITY_CLASS_NAME=admin_config.get("default_priority_class", "high-priority-job"),
        )

        if not k8s_utils.apply_yaml(worker_yaml, namespace):
            logger.error("Failed to create worker VolcanoJob")
            return False

        logger.info(f"✓ Created worker VolcanoJob: {job_name}-workers")
        return True

    except Exception as e:
        logger.error(f"Failed to launch workers: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


def launch_inference(
    job_name: str, namespace: str, composed_config: dict[str, Any], admin_config: dict[str, Any]
) -> bool:
    """Launch inference service"""
    logger.info("Launching inference service...")

    try:
        # Extract parallel strategy from composed configuration
        inference_parallel = composed_config.get("inference_parallel", {})
        replicas = inference_parallel.get("replicas", 1)
        tensor_parallel_size = inference_parallel.get("tensor_parallel_size", 1)
        inference_gpus = inference_parallel.get("gpus_per_replica", tensor_parallel_size)

        # Get model information from train_service.actor.model.path
        service_config = composed_config.get("service", {})
        train_service_config = service_config.get("train_service", {})
        actor_config = train_service_config.get("actor", {})
        model_config = actor_config.get("model", {})
        model_path = model_config.get("path")
        model_tag = composed_config.get("shared", {}).get("model_tag", job_name)

        logger.info(f"  Replicas: {replicas}")
        logger.info(f"  Tensor Parallel Size: {tensor_parallel_size}")
        logger.info(f"  GPUs per Replica: {inference_gpus}")
        logger.info(f"  Model Path: {model_path}")
        logger.info(f"  Model Tag: {model_tag}")

        if not model_path:
            logger.error("Missing 'path' in train_service.actor.model config")
            return False

        inference_image = admin_config.get("inference_image")
        storage_path = admin_config.get("storage_path")
        assert isinstance(
            inference_image, str
        ), "inference_image must be configured in admin settings"
        assert isinstance(storage_path, str), "storage_path must be configured in admin settings"

        inference_yaml = yaml_builder.render_rollout_workers_deployment(
            job_name=job_name,
            namespace=namespace,
            served_model_name=model_tag,
            inference_image=inference_image,
            model_path=model_path,
            storage_path=storage_path,
            inference_gpus=inference_gpus,
            INFERENCE_REPLICAS=replicas,
            TP_SIZE=tensor_parallel_size,
            DP_SIZE=replicas,
        )

        if not k8s_utils.apply_yaml(inference_yaml, namespace):
            logger.error("Failed to create inference Deployment")
            return False

        logger.info(f"✓ Created inference Deployment: {job_name}-rollout")
        return True

    except Exception as e:
        logger.error(f"Failed to launch inference service: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


def launch_controller(
    job_name: str,
    namespace: str,
    identifier: str,
    composed_config: dict[str, Any],
    admin_config: dict[str, Any],
    user_id: str,
    job_path: str,
) -> bool:
    """Launch NexRL controller"""
    logger.info("Launching controller...")

    try:
        # Use composed configuration directly
        nexrl_config = composed_config.copy()

        # Load user secrets
        user_secrets = load_user_secrets(namespace, user_id)

        router_urls = config_manager.get_router_urls()
        if not router_urls:
            logger.error("Failed to retrieve router URLs")
            return False

        train_router_url = router_urls["train_router_url"]
        rollout_router_url = router_urls["rollout_router_url"]

        # Build command-line parameters
        def build_params_from_dict(config_dict, prefix=""):
            """Recursively build parameter list from config dict"""
            params = []
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    nested_params = build_params_from_dict(
                        value, f"{prefix}.{key}" if prefix else key
                    )
                    params.extend(nested_params)
                elif value is not None and value != "":
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, bool):
                        params.append(f"{full_key}={str(value).lower()}")
                    elif isinstance(value, (int, float)):
                        params.append(f"{full_key}={value}")
                    elif isinstance(value, list):
                        # Lists are passed to Hydra in JSON format
                        list_str = json.dumps(value)
                        # JSON string is wrapped in single quotes when passed to bash
                        # Internal double quotes will be preserved
                        params.append(f"{full_key}='{list_str}'")
                    else:
                        # String values need to be wrapped in single quotes to avoid special character issues
                        # Escape internal single quotes
                        escaped_value = str(value).replace("'", "'\\''")
                        params.append(f"{full_key}='{escaped_value}'")
            return params

        nexrl_params = []

        # Top-level parameters
        for key in ["project_name", "experiment_name", "launch_mode"]:
            if key in nexrl_config and nexrl_config[key]:
                value = nexrl_config[key]
                # Wrap string values in single quotes
                if isinstance(value, str):
                    escaped_value = value.replace("'", "'\\''")
                    nexrl_params.append(f"{key}='{escaped_value}'")
                else:
                    nexrl_params.append(f"{key}={value}")

        # Nested configuration sections
        # Note: skip train_parallel and inference_parallel as they're only for K8s deployment
        for section in [
            "shared",
            "data",
            "rollout_worker",
            "service",
            "trajectory_pool",
            "algorithm",
            "train_worker",
            "weight",
            "validate",
            "logger",
        ]:
            if section in nexrl_config:
                section_params = build_params_from_dict(nexrl_config[section], section)
                nexrl_params.extend(section_params)

        # Inject runtime parameters
        updated_params = []
        for param in nexrl_params:
            if param.startswith("service.train_service.url="):
                updated_params.append(f"service.train_service.url='{train_router_url}'")
            elif param.startswith("service.inference_service.base_url="):
                updated_params.append(f"service.inference_service.base_url='{rollout_router_url}'")
            elif param.startswith("service.train_service.identifier="):
                updated_params.append(f"service.train_service.identifier='{identifier}'")
            else:
                updated_params.append(param)

        # Ensure critical parameters exist
        if not any(p.startswith("service.train_service.url=") for p in updated_params):
            updated_params.append(f"service.train_service.url='{train_router_url}'")
        if not any(p.startswith("service.inference_service.base_url=") for p in updated_params):
            updated_params.append(f"service.inference_service.base_url='{rollout_router_url}'")
        if not any(p.startswith("service.train_service.identifier=") for p in updated_params):
            updated_params.append(f"+service.train_service.identifier='{identifier}'")

        controller_config = {
            "NEXRL_PARAMS": " \\\n            ".join(updated_params),
            "WANDB_ENABLED": user_secrets["wandb_enabled"],
            "WANDB_HOST": user_secrets["wandb_host"],
        }

        controller_image = admin_config.get("controller_image")
        storage_path = admin_config.get("storage_path")
        assert isinstance(
            controller_image, str
        ), "controller_image must be configured in admin settings"
        assert isinstance(storage_path, str), "storage_path must be configured in admin settings"

        controller_yaml = yaml_builder.render_nexrl_controller_volcanojob(
            job_name=job_name,
            namespace=namespace,
            identifier=identifier,
            controller_image=controller_image,
            train_router_url=train_router_url,
            rollout_router_url=rollout_router_url,
            storage_path=storage_path,
            job_config_path=job_path,
            QUEUE=admin_config.get("default_queue", "default"),
            PRIORITY_CLASS_NAME=admin_config.get("default_priority_class", "high-priority-job"),
            USER_ID=user_id,
            **controller_config,
        )

        if not k8s_utils.apply_yaml(controller_yaml, namespace):
            logger.error("Failed to create controller VolcanoJob")
            return False

        logger.info(f"✓ Controller ready: {job_name}-controller")
        return True

    except Exception as e:
        logger.error(f"Failed to launch controller: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


def monitor_job(job_name: str, namespace: str, watch: bool = False):
    """Monitor job status"""
    click.echo("\n" + "=" * 60)
    click.echo(f"Job: {job_name}")
    click.echo(f"Namespace: {namespace}")
    click.echo("=" * 60)

    click.echo("\nComponents:")
    click.echo(f"  - Workers: {job_name}-workers (VolcanoJob)")
    click.echo(f"  - Rollout: {job_name}-rollout (Deployment)")
    click.echo(f"  - Controller: {job_name}-controller (VolcanoJob)")

    pods = k8s_utils.get_pods_by_label(f"app={job_name}-workers", namespace)
    if pods:
        click.echo(f"\n✓ {len(pods)} worker pod(s) found")

    click.echo("\nUseful commands:")
    click.echo(f"  View workers logs:")
    click.echo(f"    kubectl logs -f -l app={job_name}-workers -n {namespace}")
    click.echo(f"  View controller logs:")
    click.echo(f"    kubectl logs -f -l app={job_name}-controller -n {namespace}")
    click.echo(f"  View rollout logs:")
    click.echo(f"    kubectl logs -f -l app={job_name}-rollout -n {namespace}")
    click.echo(f"  Cleanup:")
    click.echo(f"    kubectl delete vj {job_name}-workers {job_name}-controller -n {namespace}")
    click.echo(f"    kubectl delete deployment {job_name}-rollout -n {namespace}")

    if watch:
        click.echo("\nMonitoring job status (Ctrl+C to exit)...")
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            click.echo("\nStopped monitoring")


@click.command()
@click.option("--job-path", required=True, help="Job configuration directory path")
@click.option("--watch", is_flag=True, help="Monitor job status")
def launch(job_path: str, watch: bool):
    """Launch NexRL training job"""

    click.echo("=" * 60)
    click.echo("NexRL Launch")
    click.echo("=" * 60)

    # Ensure user configuration exists
    local_config = ensure_user_config()

    # Use fixed namespace
    namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE

    click.echo(f"\nUser: {local_config['user_name']}")
    click.echo(f"Namespace: {namespace}")

    # Check dependencies
    if not check_launch_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)

    # Load admin configuration
    admin_config = config_manager.load_admin_config()
    if not admin_config:
        logger.error("Failed to load admin configuration")
        sys.exit(1)

    # Load job configuration
    logger.info("Loading job configuration...")
    try:
        composed_config = load_job_config(job_path)
        logger.info("✓ Job configuration loaded successfully")

        # Extract job_name from configuration
        job_name = composed_config.get("job_name")
        if not job_name:
            logger.error("job_name not found in rl_train.yaml")
            sys.exit(1)

        logger.info(f"Job name from config: {job_name}")

        # Log key configuration parameters
        train_parallel = composed_config.get("train_parallel", {})
        inference_parallel = composed_config.get("inference_parallel", {})

        click.echo(
            f"  Train Workers: {train_parallel.get('world_size', 'N/A')} GPUs ({train_parallel.get('gpus_per_pod', 'N/A')} per pod)"
        )
        click.echo(
            f"  Inference Workers: {inference_parallel.get('replicas', 'N/A')} replicas (TP={inference_parallel.get('tensor_parallel_size', 'N/A')})"
        )
    except Exception as e:
        logger.error(f"Failed to load job configuration: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        sys.exit(1)

    # Generate identifier
    identifier = generate_identifier(job_name)

    # Confirm launch
    click.echo(f"\nJob: {job_name}")
    click.echo(f"Identifier: {identifier}")
    click.echo(f"Config path: {job_path}")

    if not click.confirm("\nLaunch job?", default=True):
        click.echo("Cancelled")
        sys.exit(0)

    # Launch workers
    if not launch_workers(job_name, namespace, identifier, job_path, composed_config, admin_config):
        logger.error("Failed to launch workers")
        sys.exit(1)

    # Launch inference service
    if not launch_inference(job_name, namespace, composed_config, admin_config):
        logger.error("Failed to launch inference service")
        k8s_utils.delete_resource("volcanojob", f"{job_name}-workers", namespace)
        sys.exit(1)

    # Wait for worker pods to be ready
    train_parallel = composed_config.get("train_parallel", {})
    world_size = train_parallel.get("world_size", 1)
    gpus_per_pod = train_parallel.get("gpus_per_pod", 8)
    num_worker_pods = (world_size + gpus_per_pod - 1) // gpus_per_pod  # Ceiling division

    logger.info(f"Waiting for {num_worker_pods} worker pod(s) to be ready...")
    if not k8s_utils.wait_for_pods_ready(
        f"app={job_name}-workers", namespace, expected_count=num_worker_pods, timeout=1800
    ):
        logger.error("Worker pods failed to become ready")
        k8s_utils.delete_resource("volcanojob", f"{job_name}-workers", namespace)
        sys.exit(1)

    # Wait for inference pods to be ready
    inference_parallel = composed_config.get("inference_parallel", {})
    num_inference_replicas = inference_parallel.get("replicas", 1)

    logger.info(f"Waiting for {num_inference_replicas} inference pod(s) to be ready...")
    if not k8s_utils.wait_for_pods_ready(
        f"job-name={job_name}", namespace, expected_count=num_inference_replicas, timeout=1800
    ):
        logger.error("Inference pods failed to become ready")
        k8s_utils.delete_resource("volcanojob", f"{job_name}-workers", namespace)
        k8s_utils.delete_resource("deployment", f"{job_name}-rollout", namespace)
        sys.exit(1)

    # Launch controller (only after workers and inference are ready)
    logger.info("All dependencies ready, launching controller...")
    if not launch_controller(
        job_name,
        namespace,
        identifier,
        composed_config,
        admin_config,
        local_config["user_id"],
        job_path,
    ):
        logger.error("Failed to launch controller")
        k8s_utils.delete_resource("volcanojob", f"{job_name}-workers", namespace)
        k8s_utils.delete_resource("deployment", f"{job_name}-rollout", namespace)
        sys.exit(1)

    click.echo("\n" + "=" * 60)
    click.echo("✓ NexRL job launched successfully!")
    click.echo("=" * 60)

    # Monitor job
    monitor_job(job_name, namespace, watch)


if __name__ == "__main__":
    launch()
