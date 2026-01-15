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
Kubernetes utilities for NexRL CLI
"""

import json
import logging
import os
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

# NexRL system configuration uses a fixed namespace (configurable via env var)
NEXRL_SYSTEM_NAMESPACE = os.getenv("NEXRL_SYSTEM_NAMESPACE", "nexrl")


def check_kubectl_available() -> bool:
    """Check if kubectl is available"""
    try:
        result = subprocess.run(
            ["kubectl", "version", "--client"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_kubectl_permission(resource: str, verb: str, namespace: str) -> bool:
    """Check kubectl permissions"""
    try:
        result = subprocess.run(
            ["kubectl", "auth", "can-i", verb, resource, "-n", namespace],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "yes" in result.stdout.lower()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_crd_exists(crd_name: str) -> bool:
    """Check if CRD exists"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "crd", crd_name], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_queue_exists(queue_name: str) -> bool:
    """Check if Volcano Queue exists"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "queue", queue_name], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_namespace_exists(namespace: str) -> bool:
    """Check if namespace exists"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "namespace", namespace], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def create_namespace(namespace: str) -> bool:
    """Create namespace"""
    try:
        result = subprocess.run(
            ["kubectl", "create", "namespace", namespace],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info(f"Created namespace: {namespace}")
            return True
        else:
            logger.error(f"Failed to create namespace: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Error creating namespace: {e}")
        return False


def apply_yaml(yaml_content: str, namespace: str = NEXRL_SYSTEM_NAMESPACE) -> bool:
    """Apply YAML configuration

    Args:
        yaml_content: YAML content to apply
        namespace: Namespace to apply to (optional, for cluster-scoped resources)
    """
    try:
        cmd = ["kubectl", "apply", "-f", "-"]
        if namespace:
            cmd.extend(["-n", namespace])

        result = subprocess.run(
            cmd,
            input=yaml_content,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            if namespace:
                logger.info(f"Applied YAML successfully in namespace {namespace}")
            else:
                logger.info(f"Applied YAML successfully")
            logger.debug(result.stdout)
            return True
        else:
            logger.error(f"Failed to apply YAML: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Error applying YAML: {e}")
        return False


def wait_for_deployment(name: str, namespace: str, timeout: int = 300) -> bool:
    """Wait for Deployment to be ready"""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "wait",
                "--for=condition=available",
                f"deployment/{name}",
                "-n",
                namespace,
                f"--timeout={timeout}s",
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
        if result.returncode == 0:
            logger.info(f"Deployment {name} is ready")
            return True
        else:
            logger.error(f"Deployment {name} failed to become ready: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Error waiting for deployment: {e}")
        return False


def wait_for_volcanojob(name: str, namespace: str, timeout: int = 300) -> bool:
    """Wait for VolcanoJob to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ["kubectl", "get", "vj", name, "-n", namespace, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                job_status = json.loads(result.stdout)
                status = job_status.get("status", {})
                state = status.get("state", {}).get("phase", "")

                if state == "Running":
                    logger.info(f"VolcanoJob {name} is running")
                    return True
                elif state in ["Failed", "Terminated"]:
                    logger.error(f"VolcanoJob {name} failed with state: {state}")
                    return False

                logger.debug(f"VolcanoJob {name} state: {state}")

            time.sleep(5)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error checking VolcanoJob status: {e}")
            time.sleep(5)

    logger.error(f"Timeout waiting for VolcanoJob {name}")
    return False


def get_configmap(name: str, namespace: str) -> dict[str, Any] | None:
    """Get ConfigMap (returns None if not found - this is normal for dev/testing)"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "configmap", name, "-n", namespace, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            # ConfigMap not found is normal - we fall back to env vars
            logger.debug(
                f"ConfigMap {name} not found in namespace {namespace} (using env vars/defaults)"
            )
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        logger.debug(f"Could not get ConfigMap {name}: {e}")
        return None


def get_secret(name: str, namespace: str) -> dict[str, Any] | None:
    """Get Secret (returns None if not found - this is normal)"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "secret", name, "-n", namespace, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            # Secret not found is normal - we fall back to env vars
            # Use debug level so it doesn't show by default
            logger.debug(
                f"Secret {name} not found in namespace {namespace} (this is expected when using env vars)"
            )
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        logger.debug(f"Could not get Secret {name}: {e}")
        return None


def create_or_update_secret(name: str, namespace: str, data: dict[str, str]) -> bool:
    """Create or update Secret"""
    subprocess.run(
        ["kubectl", "delete", "secret", name, "-n", namespace], capture_output=True, timeout=10
    )
    cmd = ["kubectl", "create", "secret", "generic", name, "-n", namespace]
    for key, value in data.items():
        cmd.extend([f"--from-literal={key}={value}"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"Created/updated Secret {name}")
            return True
        else:
            logger.error(f"Failed to create Secret: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Error creating Secret: {e}")
        return False


def create_or_update_configmap(name: str, namespace: str, data: dict[str, str]) -> bool:
    """Create or update ConfigMap"""
    subprocess.run(
        ["kubectl", "delete", "configmap", name, "-n", namespace], capture_output=True, timeout=10
    )
    cmd = ["kubectl", "create", "configmap", name, "-n", namespace]
    for key, value in data.items():
        cmd.extend([f"--from-literal={key}={value}"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"Created/updated ConfigMap {name}")
            return True
        else:
            logger.error(f"Failed to create ConfigMap: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Error creating ConfigMap: {e}")
        return False


def delete_resource(resource_type: str, name: str, namespace: str) -> bool:
    """Delete K8s resource

    Args:
        resource_type: Resource type, supports both short names (e.g. 'volcanojob') and full format
        name: Resource name
        namespace: Namespace
    """
    resource_type_mapping = {
        "volcanojob": "vj",
        "deployment": "deployment",
        "service": "service",
        "configmap": "configmap",
        "secret": "secret",
    }

    full_resource_type = resource_type_mapping.get(resource_type, resource_type)

    try:
        result = subprocess.run(
            ["kubectl", "delete", full_resource_type, name, "-n", namespace],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            logger.info(f"Deleted {full_resource_type}/{name}")
            return True
        else:
            logger.warning(f"Failed to delete {full_resource_type}/{name}: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Error deleting resource: {e}")
        return False


def get_pod_logs(pod_name: str, namespace: str, tail: int = 100) -> str | None:
    """Get Pod logs"""
    try:
        result = subprocess.run(
            ["kubectl", "logs", pod_name, "-n", namespace, f"--tail={tail}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
        else:
            logger.error(f"Failed to get logs for {pod_name}: {result.stderr}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Error getting pod logs: {e}")
        return None


def get_pods_by_label(label: str, namespace: str) -> list[str]:
    """Get Pod list by label"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", label, "-n", namespace, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            pods_data = json.loads(result.stdout)
            return [pod["metadata"]["name"] for pod in pods_data.get("items", [])]
        else:
            logger.error(f"Failed to get pods: {result.stderr}")
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error getting pods: {e}")
        return []


def wait_for_pods_ready(
    label: str, namespace: str, expected_count: int = 0, timeout: int = 600
) -> bool:
    """Wait for pods with given label to be ready

    Args:
        label: Label selector for pods (e.g. "app=my-app")
        namespace: Kubernetes namespace
        expected_count: Expected number of pods (optional, if None just wait for any pods to be ready)
        timeout: Timeout in seconds

    Returns:
        True if pods are ready, False otherwise
    """
    logger.info(f"Waiting for pods with label '{label}' to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-l", label, "-n", namespace, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                pods = pods_data.get("items", [])

                if not pods:
                    logger.debug(f"No pods found with label '{label}', waiting...")
                    time.sleep(5)
                    continue

                # Check if we have the expected number of pods (if specified)
                if expected_count is not None and len(pods) < expected_count:
                    logger.debug(f"Found {len(pods)}/{expected_count} pods, waiting for more...")
                    time.sleep(5)
                    continue

                # Check if all pods are ready
                all_ready = True
                ready_count = 0
                for pod in pods:
                    pod_name = pod["metadata"]["name"]
                    status = pod.get("status", {})
                    phase = status.get("phase", "")

                    # Check if pod is in Running phase
                    if phase != "Running":
                        logger.debug(f"Pod {pod_name} is in phase: {phase}")
                        all_ready = False
                        continue

                    # Check container readiness
                    conditions = status.get("conditions", [])
                    pod_ready = False
                    for condition in conditions:
                        if condition.get("type") == "Ready":
                            if condition.get("status") == "True":
                                pod_ready = True
                                ready_count += 1
                            break

                    if not pod_ready:
                        logger.debug(f"Pod {pod_name} is not ready yet")
                        all_ready = False

                if all_ready and (expected_count is None or len(pods) == expected_count):
                    logger.info(f"✓ All {len(pods)} pod(s) with label '{label}' are ready")
                    return True
                else:
                    logger.debug(f"Pods status: {ready_count}/{len(pods)} ready")

            time.sleep(5)

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error checking pod status: {e}")
            time.sleep(5)

    logger.error(f"Timeout waiting for pods with label '{label}' to be ready")
    return False


def run_kubectl_command(
    args: list[str], capture_output: bool = False, timeout: int = 30
) -> subprocess.CompletedProcess:
    """Run kubectl command

    Args:
        args: kubectl command arguments (without 'kubectl' prefix)
        capture_output: Whether to capture output
        timeout: Command timeout in seconds

    Returns:
        CompletedProcess object
    """
    cmd = ["kubectl"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"kubectl command timed out: {' '.join(cmd)}")
        raise
    except FileNotFoundError:
        logger.error("kubectl not found")
        raise
