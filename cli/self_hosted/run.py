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
Unified launcher for NexRL on Kubernetes (Self-Hosted Mode - Open Source)

What it does:
1) Launch API server (training backend)
2) Launch GPU workers (FSDP training)
3) Launch SGLang inference service
4) Launch NexRL driver with Ray agents

Usage:
  python cli/run.py self-hosted --train-config <train-config-path> --run-nexrl
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add CLI directory to path (must be before local imports)
CLI_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CLI_DIR))

# pylint: disable=wrong-import-position
import yaml  # type: ignore

from ..common.config_utils import (
    load_agent_settings,
    load_config,
    load_identifier_world_sizes,
    load_inference_resource,
)
from ..utils import config_manager, k8s_utils
from . import yaml_builder as builder

# pylint: enable=wrong-import-position

USER = os.getenv("USER", "nexrl")
NEXRL_PATH = Path(os.getenv("NEXRL_PATH") or Path(__file__).resolve().parents[2]).resolve()

# Training driver job
DRIVER_CMD = "cli/common/run_nexrl.sh"
SLEEP_CMD = "cli/common/ray_setup.sh"

# Common settings
NAMESPACE = k8s_utils.NEXRL_SYSTEM_NAMESPACE
SERVICE_ACCOUNT_NAME = os.getenv("NEXRL_SERVICE_ACCOUNT", "nexrl-job-manager")
SGLANG_ROUTER_ACCOUNT_NAME = os.getenv("SGLANG_ROUTER_ACCOUNT_NAME", "sglang-router")

# Timeouts
POD_READY_TIMEOUT_SEC = 600
POD_POLL_INTERVAL_SEC = 5


# ============================================================================
# Helpers
# ============================================================================
def sh(cmd: str, *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command."""
    return subprocess.run(
        cmd,
        shell=True,
        check=check,
        text=True,
        capture_output=capture,
    )


def wait_for_pod(namespace: str, pod_name: str) -> None:
    """Wait for pod to become Running."""
    waited = 0
    while waited < POD_READY_TIMEOUT_SEC:
        proc = sh(
            f"kubectl get pod -n {namespace} {pod_name} -o jsonpath='{{.status.phase}}'",
            capture=True,
            check=False,
        )
        status = (proc.stdout or "").strip()
        if status == "Running":
            print(f"[INFO] Pod {pod_name} is Running")
            return
        print(f"[INFO] Waiting for pod {pod_name}, current status: {status or 'Unknown'}")
        waited += POD_POLL_INTERVAL_SEC
        time.sleep(POD_POLL_INTERVAL_SEC)
    print(f"[WARNING] Pod {pod_name} not Running within {POD_READY_TIMEOUT_SEC}s")


def get_pod_ip(namespace: str, pod_name: str) -> str:
    """Get pod IP address."""
    proc = sh(
        f"kubectl get pod -n {namespace} {pod_name} -o jsonpath='{{.status.podIP}}'",
        capture=True,
    )
    ip = (proc.stdout or "").strip()
    if not ip:
        raise RuntimeError(f"Failed to get pod IP for {pod_name}")
    return ip


def get_service_cluster_ip(namespace: str, service_name: str) -> str:
    """Get service cluster IP."""
    proc = sh(
        f"kubectl get svc -n {namespace} {service_name} -o jsonpath='{{.spec.clusterIP}}'",
        capture=True,
    )
    ip = (proc.stdout or "").strip()
    if not ip:
        raise RuntimeError(f"Failed to get service IP for {service_name}")
    return ip


def apply_yaml_and_get_pod_for_deployment(
    yaml_content: str, namespace: str, pod_selector: str
) -> str:
    """Apply YAML and extract pod name for Deployments."""
    # Write YAML to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_file = f.name

    try:
        # Apply with kubectl
        result = sh(f"kubectl apply -f {yaml_file} -n {namespace}", capture=True, check=False)
        if result.returncode != 0:
            print(f"[ERROR] kubectl apply failed!")
            print(f"[ERROR] stdout: {result.stdout}")
            print(f"[ERROR] stderr: {result.stderr}")
            print(f"[ERROR] YAML file kept at: {yaml_file}")
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )
        print(f"[INFO] Applied YAML: {result.stdout.strip()}")

        # Wait a moment for pod to be created
        time.sleep(2)

        # Get pod name using label selector
        proc = sh(
            f"kubectl get pods -n {namespace} -l app={pod_selector} "
            f"-o jsonpath='{{.items[0].metadata.name}}'",
            capture=True,
            check=False,
        )
        pod_name = (proc.stdout or "").strip()
        if not pod_name:
            # Try without label selector (for deployments)
            proc = sh(
                f"kubectl get pods -n {namespace} --selector=app={pod_selector} "
                f"-o jsonpath='{{.items[0].metadata.name}}'",
                capture=True,
                check=False,
            )
            pod_name = (proc.stdout or "").strip()

        return pod_name if pod_name else pod_selector
    finally:
        Path(yaml_file).unlink(missing_ok=True)


def apply_yaml_and_get_pod_for_volcanojob(yaml_content: str, namespace: str, job_name: str) -> str:
    """Apply YAML and get the master pod name for Volcano jobs.

    Args:
        yaml_content: The Volcano job YAML to apply
        namespace: Kubernetes namespace
        job_name: The Volcano job name (from metadata.name in YAML)

    Returns:
        The master pod name (e.g., job-name-master-0)
    """
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_file = f.name

    try:
        # Apply with kubectl
        result = sh(f"kubectl apply -f {yaml_file} -n {namespace}", capture=True, check=False)
        if result.returncode != 0:
            print(f"[ERROR] kubectl apply failed!")
            print(f"[ERROR] stdout: {result.stdout}")
            print(f"[ERROR] stderr: {result.stderr}")
            print(f"[ERROR] YAML file kept at: {yaml_file}")
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )
        print(f"[INFO] Applied Volcano job: {result.stdout.strip()}")

        # Volcano jobs create pods with the pattern: {job-name}-{task-name}-{index}
        # For driver jobs, the master pod is: {job-name}-master-0
        master_pod_name = f"{job_name}-master-0"
        print(f"[INFO] Expected master pod name: {master_pod_name}")

        # Wait for the pod to be created and verify it exists
        max_wait = 30  # seconds
        interval = 2  # seconds
        waited = 0

        while waited < max_wait:
            proc = sh(
                f"kubectl get pod {master_pod_name} -n {namespace} --ignore-not-found -o name",
                capture=True,
                check=False,
            )
            if proc.stdout.strip():
                print(f"[INFO] Confirmed pod exists: {master_pod_name}")
                return master_pod_name

            print(
                f"[INFO] Waiting for pod {master_pod_name} to be created... ({waited}s/{max_wait}s)"
            )
            time.sleep(interval)
            waited += interval

        # If we get here, pod wasn't found, but return the name anyway
        # The wait_for_pod function will handle the timeout if it doesn't exist
        print(f"[WARNING] Pod {master_pod_name} not found after {max_wait}s, but continuing...")
        return master_pod_name

    finally:
        Path(yaml_file).unlink(missing_ok=True)


def compute_experiment_path(run_id: str, config_path: str | Path) -> Path:
    """Compute experiment path."""
    with open(str(config_path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    experiment_name = cfg.get("experiment_name", "default")
    return NEXRL_PATH / "logs" / experiment_name / run_id


def load_environment_script(config: dict) -> str | None:
    """Extract environment setup script path from config."""
    env_config = config.get("environment", {})
    script_path = env_config.get("setup_script")

    if env_config.get("require_setup_script", False) and not script_path:
        raise ValueError("environment.setup_script is required but not specified in config")

    if script_path:
        print(f"[INFO] Environment setup script: {script_path}")

    return script_path


# ============================================================================
# Orchestration
# ============================================================================
def launch_api_server(
    run_id: str,
    output_dir: Path,
    experiment_path: Path,
    admin_config: dict,
    tag: str = "",
) -> str:
    """Launch API server as VolcanoJob."""
    print("[INFO] Launching API server...")

    # Extract HMS (HHMMSS) from run_id (YYYYMMDD-HHMMSS)
    hms = run_id.split("-")[-1] if "-" in run_id else run_id
    suffix = f"{tag}-{hms}" if tag else run_id
    api_name = f"{USER}-api-server-{suffix}"

    yaml_content = builder.render_api_server_volcanojob(
        name=api_name,
        namespace=NAMESPACE,
        image=admin_config["controller_image"],  # Same image as controller
        nexrl_path=str(NEXRL_PATH),
        storage_path=admin_config["storage_path"],
        experiment_path=str(experiment_path),
        queue=admin_config.get("default_queue", "default"),
        priority_class_name=admin_config.get("default_priority_class", "high-priority-job"),
    )

    # Save YAML
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_file = output_dir / f"{api_name}.yaml"
    yaml_file.write_text(yaml_content)

    # Apply
    pod_name = apply_yaml_and_get_pod_for_volcanojob(yaml_content, NAMESPACE, api_name)
    print(f"[INFO] ✓ API server created: {api_name}")

    wait_for_pod(NAMESPACE, pod_name)

    # Get pod IP (with hostNetwork, pod IP is the host IP)
    pod_ip = get_pod_ip(NAMESPACE, pod_name)
    print(f"[INFO] ✓ API server ready at: {pod_ip}")

    return pod_ip


def launch_gpu_workers(
    run_id: str,
    api_server_ip: str,
    output_dir: Path,
    experiment_path: Path,
    identifier_world_sizes: dict[str, int],
    admin_config: dict,
    tag: str = "",
):
    """Launch GPU workers."""
    print("[INFO] Launching GPU workers...")

    # Extract HMS (HHMMSS) from run_id (YYYYMMDD-HHMMSS)
    hms = run_id.split("-")[-1] if "-" in run_id else run_id
    suffix = f"{tag}-{hms}" if tag else run_id
    for identifier, world_size in identifier_world_sizes.items():
        gpu_name = f"{USER}-gpu-worker-{identifier}-{suffix}"

        print(f"[INFO] Launching GPU worker group: {identifier} (world_size={world_size})")

        yaml_content = builder.render_gpu_worker_volcanojob(
            name=gpu_name,
            namespace=NAMESPACE,
            identifier=identifier,
            image=admin_config["worker_image"],
            api_server_url=api_server_ip,
            nexrl_path=str(NEXRL_PATH),
            storage_path=admin_config["storage_path"],
            experiment_path=str(experiment_path),
            world_size=world_size,
            queue=admin_config.get("default_queue", "default"),
            priority_class_name=admin_config.get("default_priority_class", "high-priority-job"),
        )

        # Save YAML
        yaml_file = output_dir / f"{gpu_name}.yaml"
        yaml_file.write_text(yaml_content)

        # Apply
        pod_name = apply_yaml_and_get_pod_for_volcanojob(yaml_content, NAMESPACE, gpu_name)
        print(f"[INFO] ✓ GPU worker created: {gpu_name}")
        wait_for_pod(NAMESPACE, pod_name)


def launch_rollout_router(
    run_id: str,
    output_dir: Path,
    admin_config: dict,
    served_model_name: str,
    tag: str = "",
) -> str:
    """Launch dynamic rollout router and return its pod IP."""
    print("[INFO] Launching dynamic rollout router...")

    # Extract HMS (HHMMSS) from run_id (YYYYMMDD-HHMMSS)
    hms = run_id.split("-")[-1] if "-" in run_id else run_id
    suffix = f"{tag}-{hms}" if tag else run_id
    router_name = f"{USER}-rollout-router-{suffix}"

    yaml_content = builder.render_rollout_router_deployment(
        name=router_name,
        namespace=NAMESPACE,
        image=admin_config.get("inference_image", "lmsysorg/sglang:latest"),
        service_account_name=SGLANG_ROUTER_ACCOUNT_NAME,  # Use service account with pod list permissions
        redis_host=None,  # No Redis for dynamic router
        served_model_name=served_model_name,  # Pass model name for service discovery
    )

    # Save YAML
    yaml_file = output_dir / f"{router_name}.yaml"
    yaml_file.write_text(yaml_content)

    # Apply
    apply_yaml_and_get_pod_for_deployment(yaml_content, NAMESPACE, router_name)
    print(f"[INFO] ✓ Rollout router created: {router_name}")

    # Wait for pod to be ready
    if not k8s_utils.wait_for_pods_ready(
        f"app={router_name}", NAMESPACE, expected_count=1, timeout=600
    ):
        print("[WARNING] Router pod not ready within timeout")
        return ""

    # Get pod IP
    proc = sh(
        f"kubectl get pod -n {NAMESPACE} -l app={router_name} "
        f"-o jsonpath='{{.items[0].status.podIP}}'",
        capture=True,
        check=False,
    )
    pod_ip = (proc.stdout or "").strip()

    if pod_ip:
        router_url = f"{pod_ip}:12345"
        print(f"[INFO] ✓ Router ready at: http://{router_url}")
        return router_url
    else:
        print("[WARNING] Failed to get router pod IP")
        return ""


def launch_inference(
    served_model_name: str,
    model_path: Path,
    replicas: int,
    gpus_per_pod: int,
    run_id: str,
    output_dir: Path,
    admin_config: dict,
    tag: str = "",
) -> bool:
    """Launch SGLang inference service."""
    print("[INFO] Launching inference service...")

    # Extract HMS (HHMMSS) from run_id (YYYYMMDD-HHMMSS)
    hms = run_id.split("-")[-1] if "-" in run_id else run_id
    suffix = f"{tag}-{hms}" if tag else run_id
    inference_name = f"{USER}-inference-{suffix}"

    print(f"[INFO] Inference config:")
    print(f"  Model: {served_model_name}")
    print(f"  Model path: {model_path}")
    print(f"  Replicas: {replicas}")
    print(f"  GPUs per replica: {gpus_per_pod}")

    yaml_content = builder.render_inference_deployment(
        name=inference_name,
        namespace=NAMESPACE,
        served_model_name=served_model_name,
        image=admin_config["inference_image"],
        model_path=str(model_path),
        storage_path=admin_config["storage_path"],
        replicas=replicas,
        gpus_per_replica=gpus_per_pod,
        tensor_parallel_size=gpus_per_pod,
    )

    # Save YAML
    yaml_file = output_dir / f"{inference_name}.yaml"
    yaml_file.write_text(yaml_content)

    # Apply
    apply_yaml_and_get_pod_for_deployment(yaml_content, NAMESPACE, inference_name)
    print(f"[INFO] ✓ Inference deployment created: {inference_name}")

    # Quick check that pods are being created
    time.sleep(2)
    result = sh(
        f"kubectl get pods -n {NAMESPACE} -l deployment={inference_name} --no-headers 2>/dev/null | wc -l",
        capture=True,
        check=False,
    )
    pod_count = result.stdout.strip() if result.returncode == 0 else "0"
    print(f"[INFO] {pod_count}/{replicas} pods created, waiting for them to become ready...")
    print(f"[INFO] (Model loading may take 2-5 minutes per pod)")

    # Wait for pods to be ready - use unique deployment label
    success = k8s_utils.wait_for_pods_ready(
        f"deployment={inference_name}", NAMESPACE, expected_count=replicas, timeout=1800
    )

    if not success:
        print(f"\n[WARNING] Pods did not become ready within timeout.")
        print(f"[INFO] Debug commands:")
        print(f"  kubectl get pods -n {NAMESPACE} -l deployment={inference_name}")
        print(f"  kubectl logs -n {NAMESPACE} -l deployment={inference_name} --tail=50 -f")
    else:
        print(f"[INFO] ✓ All inference pods ready!")

    return success


def launch_nexrl_driver(
    run_id: str,
    api_server_ip: str,
    output_dir: Path,
    experiment_path: Path,
    run_driver: bool,
    identifier_world_sizes: dict[str, int],
    num_agent_workers: int,
    num_agents_per_worker: int,
    train_config_path: Path,
    served_model_name: str,
    environment_setup_script: str | None,
    admin_config: dict,
    user_config: dict,
    user_secrets: dict,
    tag: str = "",
    inference_base_url: str = "",
) -> str:
    """Launch NexRL driver."""
    print("[INFO] Launching NexRL driver...")

    driver_cmd = DRIVER_CMD if run_driver else SLEEP_CMD
    # Extract HMS (HHMMSS) from run_id (YYYYMMDD-HHMMSS)
    hms = run_id.split("-")[-1] if "-" in run_id else run_id
    suffix = f"{tag}-{hms}" if tag else run_id
    job_name = f"{USER}-nexrl-driver-{suffix}"

    # Extract optional service configurations from user_secrets (env vars)
    wandb_key = user_secrets.get("WANDB_KEY") or None
    wandb_host = user_secrets.get("WANDB_HOST") or None
    tinker_api_key = user_secrets.get("TINKER_API_KEY") or None
    tinker_base_url = user_secrets.get("TINKER_BASE_URL") or None
    weaver_api_key = user_secrets.get("WEAVER_API_KEY") or None
    weaver_base_url = user_secrets.get("WEAVER_BASE_URL") or None

    yaml_content = builder.render_driver_volcanojob(
        name=job_name,
        namespace=NAMESPACE,
        identifier=",".join(identifier_world_sizes.keys()),
        image=admin_config["controller_image"],
        api_server_url=api_server_ip,
        nexrl_path=str(NEXRL_PATH),
        storage_path=admin_config["storage_path"],
        experiment_path=str(experiment_path),
        train_config=str(train_config_path),
        served_model_name=served_model_name,
        num_agent_workers=num_agent_workers,
        num_agents_per_worker=num_agents_per_worker,
        user_id=user_config["user_id"],
        queue=admin_config.get("default_queue", "default"),
        priority_class_name=admin_config.get("default_priority_class", "high-priority-job"),
        cmd=driver_cmd,
        service_account_name=SERVICE_ACCOUNT_NAME,
        environment_setup_script=environment_setup_script,
        inference_base_url=inference_base_url,
        wandb_key=wandb_key,
        wandb_host=wandb_host,
        tinker_api_key=tinker_api_key,
        tinker_base_url=tinker_base_url,
        weaver_api_key=weaver_api_key,
        weaver_base_url=weaver_base_url,
    )

    # Save YAML
    yaml_file = output_dir / f"{job_name}.yaml"
    yaml_file.write_text(yaml_content)

    # Apply
    pod_name = apply_yaml_and_get_pod_for_volcanojob(yaml_content, NAMESPACE, job_name)
    print(f"[INFO] ✓ NexRL driver created: {job_name}")

    wait_for_pod(NAMESPACE, pod_name)
    print(f"[INFO] ✓ Driver pod ready: {pod_name}")
    return pod_name


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="NexRL Self-Hosted Launcher (Open Source)")
    parser.add_argument("--run-nexrl", "-r", action="store_true", help="Run NexRL automatically")
    parser.add_argument(
        "--train-config",
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument("--tag", "-t", type=str, default="", help="Custom tag for job names")
    parser.add_argument(
        "--inference-url",
        type=str,
        default="",
        help="Use existing inference service URL (skips launching inference). Do not include http:// prefix.",
    )
    args = parser.parse_args()

    train_config_path = Path(args.train_config).resolve()

    # Load configurations
    print("[INFO] Loading configurations...")
    config = load_config(train_config_path)
    admin_config = config_manager.load_admin_config()
    if not admin_config:
        print("[ERROR] Failed to load configuration.")
        print("[ERROR] This should not happen - please report this bug.")
        sys.exit(1)

    # Load user config (with fallback to env vars)
    user_config_file = Path.home() / ".nexrl" / "config.yaml"
    if user_config_file.exists():
        with open(user_config_file, encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
        print(f"[INFO] Loaded user config from {user_config_file}")
    else:
        # Fall back to environment variables
        user_id = os.getenv("NEXRL_USER_ID", os.getenv("USER", "default-user"))
        user_config = {
            "user_id": user_id,
            "user_name": os.getenv("NEXRL_USER_NAME", user_id),
        }
        print(f"[INFO] Using user_id from environment: {user_id}")
        print(f"[INFO] (Optional) Create ~/.nexrl/config.yaml for persistent config")

    # Load user secrets
    user_secrets = config_manager.load_user_secrets(NAMESPACE, user_config["user_id"])

    identifier_world_sizes = load_identifier_world_sizes(config)
    inference_resource = load_inference_resource(config)
    agent_resource = load_agent_settings(config)
    environment_setup_script = load_environment_script(config)

    served_model_name = inference_resource["served_model_name"]
    model_path = Path(inference_resource["model_path"])
    replicas = inference_resource["replicas"]
    gpus_per_pod = inference_resource["gpus_per_replica"]

    num_agent_workers = agent_resource["num_workers"]
    num_agents_per_worker = agent_resource["agents_per_worker"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment_path = compute_experiment_path(run_id, train_config_path)
    experiment_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using experiment path: {experiment_path}")
    output_dir = experiment_path / "k8s_jobs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy train config
    import shutil as _shutil

    config_copy_path = experiment_path / "train_config.yaml"
    _shutil.copy2(train_config_path, config_copy_path)
    print(f"[INFO] Copied train config to: {config_copy_path}")

    # Launch components
    api_server_ip = launch_api_server(run_id, output_dir, experiment_path, admin_config, args.tag)
    print(f"[INFO] API server IP: {api_server_ip}")

    launch_gpu_workers(
        run_id,
        api_server_ip,
        output_dir,
        experiment_path,
        identifier_world_sizes,
        admin_config,
        args.tag,
    )

    # Determine inference router URL
    inference_base_url = ""
    if args.inference_url:
        # Use provided inference URL
        inference_base_url = args.inference_url
        # Strip http:// prefix if present
        if inference_base_url.startswith("http://"):
            inference_base_url = inference_base_url[7:]
        if inference_base_url.startswith("https://"):
            inference_base_url = inference_base_url[8:]
        print(f"[INFO] Using provided inference URL: {inference_base_url}")
        print("[INFO] Skipping inference service launch")
    else:
        # Launch inference service
        if not launch_inference(
            served_model_name,
            model_path,
            replicas,
            gpus_per_pod,
            run_id,
            output_dir,
            admin_config,
            args.tag,
        ):
            print("[ERROR] Inference service failed to become ready")
            sys.exit(1)

        # Launch dynamic router for this model
        print("[INFO] Launching dynamic router...")
        inference_base_url = launch_rollout_router(
            run_id,
            output_dir,
            admin_config,
            served_model_name,
            args.tag,
        )
        if not inference_base_url:
            print("[ERROR] Failed to launch dynamic router")
            sys.exit(1)

        print(f"[INFO] INFERENCE_BASE_URL set to: {inference_base_url}")

    driver_pod_name = launch_nexrl_driver(
        run_id,
        api_server_ip,
        output_dir,
        experiment_path,
        args.run_nexrl,
        identifier_world_sizes,
        num_agent_workers,
        num_agents_per_worker,
        train_config_path,
        served_model_name,
        environment_setup_script,
        admin_config,
        user_config,
        user_secrets,
        args.tag,
        inference_base_url=inference_base_url,
    )

    print("\n" + "=" * 60)
    print("✓ All jobs submitted and ready!")
    print("=" * 60)

    if not args.run_nexrl:
        print("\nTo manually run the experiment:")
        print(f"  kubectl exec -it {driver_pod_name} -n {NAMESPACE} -- bash")
        print("Then inside the pod:")
        print(f"  bash cli/common/run_nexrl.sh")
    else:
        print(f"\nExperiment is running automatically inside {driver_pod_name}")

    print(f"\nExperiment log: {experiment_path}")


if __name__ == "__main__":
    main()
