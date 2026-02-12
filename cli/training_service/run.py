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
Unified launcher for NexRL on Kubernetes (Training-Service Mode - Open Source)

What it does:
1) Launch training driver job (run_nexrl.sh) or a placeholder sleep job

Usage:
  python cli/run.py training-service --train-config <train-config-path> --run-nexrl
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Add CLI directory to path (must be before local imports)
CLI_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CLI_DIR))

# pylint: disable=wrong-import-position
import logging

import yaml  # type: ignore

from ..common.config_utils import load_agent_settings, load_config, load_model_name
from ..common.debug_utils import (
    build_debug_overrides,
    check_trajectory_exists,
    find_most_recent_experiment_with_trajectory,
    prompt_user_confirmation,
)
from ..utils import config_manager, k8s_utils
from . import yaml_builder as builder

# pylint: enable=wrong-import-position

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)

USER = os.getenv("USER", "nexrl")
NEXRL_PATH = Path(os.getenv("NEXRL_PATH") or Path(__file__).resolve().parents[2]).resolve()

# Training driver job
DRIVER_CMD = "cli/common/run_nexrl.sh"
SLEEP_CMD = "cli/common/ray_setup.sh"

# Common settings
NAMESPACE = k8s_utils.NEXRL_SYSTEM_NAMESPACE
SERVICE_ACCOUNT_NAME = os.getenv("NEXRL_SERVICE_ACCOUNT", "nexrl-job-manager")

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
    raise TimeoutError(f"Pod {pod_name} not Running within {POD_READY_TIMEOUT_SEC}s")


def apply_yaml_and_get_pod(yaml_content: str, namespace: str, job_name: str) -> str:
    """Apply YAML and get the master pod name.

    Args:
        yaml_content: The Volcano job YAML to apply
        namespace: Kubernetes namespace
        job_name: The Volcano job name (from metadata.name in YAML)

    Returns:
        The master pod name (e.g., job-name-master-0)
    """
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
def launch_nexrl_driver(
    run_id: str,
    output_dir: Path,
    experiment_path: Path,
    run_driver: bool,
    num_agent_workers: int,
    num_agents_per_worker: int,
    train_config_path: Path,
    served_model_name: str,
    environment_setup_script: str | None,
    admin_config: dict,
    user_config: dict,
    user_secrets: dict,
    tag: str = "",
    debug_hydra_overrides: str = "",
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
        image=admin_config["controller_image"],
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
        wandb_key=wandb_key,
        wandb_host=wandb_host,
        tinker_api_key=tinker_api_key,
        tinker_base_url=tinker_base_url,
        weaver_api_key=weaver_api_key,
        weaver_base_url=weaver_base_url,
        debug_hydra_overrides=debug_hydra_overrides,
    )

    # Save YAML
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_file = output_dir / f"{job_name}.yaml"
    yaml_file.write_text(yaml_content)

    # Apply
    pod_name = apply_yaml_and_get_pod(yaml_content, NAMESPACE, job_name)
    print(f"[INFO] Applied driver job: {job_name}")

    wait_for_pod(NAMESPACE, pod_name)
    return pod_name


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="NexRL Training-Service Launcher (Open Source)")
    parser.add_argument(
        "--run-nexrl", "-r", action="store_true", help="Run NexRL training automatically"
    )
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
        "--debug-mode", action="store_true", help="Enable debug mode for trajectory dump/load"
    )
    parser.add_argument(
        "--debug-baseline-path",
        type=str,
        default="",
        help="Path to baseline experiment for trajectory reuse (optional)",
    )
    args = parser.parse_args()

    train_config_path = Path(args.train_config).resolve()

    # Load configurations
    print("[INFO] Loading configurations...")
    config = load_config(train_config_path)

    # Handle debug mode
    debug_overrides = ""
    if args.debug_mode:
        trajectory_path = None
        experiment_name = config.get("experiment_name", "default")

        if args.debug_baseline_path:
            # Explicit baseline: validate and use
            baseline_path = Path(args.debug_baseline_path)
            trajectory_path = check_trajectory_exists(baseline_path)
            if not trajectory_path:
                print(f"[ERROR] No trajectory found at {baseline_path}/debug_dump/trajectory/")
                sys.exit(1)
            print(f"[INFO] Using baseline trajectory: {trajectory_path}")
        else:
            # Auto-detect: find latest WITH trajectory, then prompt
            result = find_most_recent_experiment_with_trajectory(
                NEXRL_PATH / "logs", experiment_name
            )
            if result:
                run_path, traj_path = result
                print(f"[INFO] Found trajectory from run: {run_path.name}")
                if prompt_user_confirmation(run_path, traj_path):
                    trajectory_path = traj_path
                    print(f"[INFO] Using trajectory: {trajectory_path}")
                else:
                    print("[WARNING] User declined. Normal rollout will execute.")
            else:
                print("[WARNING] No trajectory found in any previous run.")
                print("[WARNING] Normal rollout will execute.")

        # Build Hydra override string
        debug_overrides = build_debug_overrides(trajectory_path)

        if trajectory_path:
            print("[INFO] Debug mode: Mock rollout (reusing trajectory)")
            print(
                "[INFO] Automatically reducing rollout workers to 1 (no parallel benefit in mock mode)"
            )
        else:
            print("[INFO] Debug mode: Normal rollout (will dump trajectory)")

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

    agent_resource = load_agent_settings(config)
    environment_setup_script = load_environment_script(config)
    served_model_name = load_model_name(config)

    num_agent_workers = agent_resource["num_workers"]
    num_agents_per_worker = agent_resource["agents_per_worker"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment_path = compute_experiment_path(run_id, train_config_path)
    experiment_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using experiment path: {experiment_path}")

    # Copy train config
    import shutil as _shutil

    config_copy_path = experiment_path / "train_config.yaml"
    _shutil.copy2(train_config_path, config_copy_path)
    print(f"[INFO] Copied train config to: {config_copy_path}")

    output_dir = experiment_path / "k8s_jobs"

    driver_pod_name = launch_nexrl_driver(
        run_id,
        output_dir,
        experiment_path,
        args.run_nexrl,
        num_agent_workers,
        num_agents_per_worker,
        train_config_path,
        served_model_name,
        environment_setup_script,
        admin_config,
        user_config,
        user_secrets,
        args.tag,
        debug_overrides,
    )

    print("\n" + "=" * 60)
    print("✓ All jobs submitted and ready!")
    print("=" * 60)

    if not args.run_nexrl:
        print("\nTo manually run the experiment:")
        print(f"  kubectl exec -it {driver_pod_name} -n {NAMESPACE} -- bash")
        print("Then inside the pod:")
        print(f"  bash cli/common/run_nexrl.sh")
        print(f"\nFind experiment log at: {experiment_path}")
    else:
        print(f"\nExperiment is running automatically inside {driver_pod_name}")
        print(f"\nFind experiment log at: {experiment_path}")


if __name__ == "__main__":
    main()
