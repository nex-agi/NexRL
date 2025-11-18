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
Init command for NexRL CLI

Initialize user configurations like WandB API key, etc.
"""

import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import yaml  # type: ignore

from .utils import k8s_utils, yaml_builder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Local configuration file path
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
        logger.warning(f"Failed to load local configuration file: {e}")
        return {}


def save_local_config(config: dict[str, Any]) -> bool:
    """Save local configuration file"""
    try:
        # Ensure directory exists
        NEXRL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        with open(NEXRL_CONFIG_FILE, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"✓ Local configuration saved to: {NEXRL_CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save local configuration file: {e}")
        return False


def check_secret_exists(secret_name: str, namespace: str) -> bool:
    """Check if Secret already exists"""
    try:
        result = k8s_utils.run_kubectl_command(
            ["get", "secret", secret_name, "-n", namespace], capture_output=True
        )
        return result.returncode == 0
    except Exception:
        return False


def get_current_secret_value(secret_name: str, key: str, namespace: str) -> str | None:
    """Get current value from Secret"""
    try:
        result = k8s_utils.run_kubectl_command(
            ["get", "secret", secret_name, "-n", namespace, "-o", f"jsonpath={{.data.{key}}}"],
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout:
            # Base64 decode
            import base64

            return base64.b64decode(result.stdout).decode("utf-8")
    except Exception:
        pass
    return None


@click.command()
@click.option("--name", help="User name (for configuration identification)")
@click.option("--wandb-api-key", help="WandB API Key")
@click.option("--wandb-host", help="WandB Host URL (default: https://api.wandb.ai)")
@click.option("--enable-wandb/--disable-wandb", default=None, help="Enable/disable WandB")
@click.option("--interactive/--no-interactive", default=True, help="Interactive input")
def init(
    name: str | None,
    wandb_api_key: str | None,
    wandb_host: str | None,
    enable_wandb: bool | None,
    interactive: bool,
):
    """
    Initialize NexRL user configuration

    Save WandB and other configurations to Kubernetes Secret,
    which will be automatically used in subsequent launches.

    Examples:
        # Interactive initialization
        nexrl init

        # Command-line parameter initialization
        nexrl init --wandb-api-key xxx --enable-wandb
    """
    logger.info("=" * 60)
    logger.info("NexRL User Configuration Initialization")
    logger.info("=" * 60)

    # Use fixed namespace
    namespace = k8s_utils.NEXRL_SYSTEM_NAMESPACE

    # Load local configuration
    local_config = load_local_config()
    current_name = local_config.get("user_name", "")
    current_user_id = local_config.get("user_id", "")

    # Handle username
    if interactive and name is None:
        if current_name:
            logger.info(f"\nCurrent configured username: {current_name}")
            logger.info(f"User ID: {current_user_id}")
            if not click.confirm("Use current username?", default=True):
                name = click.prompt("Please enter your username", default=current_name)
            else:
                name = current_name
        else:
            logger.info("\nFirst-time NexRL configuration")
            name = click.prompt(
                "Please enter your username (for configuration identification)",
                default=os.getenv("USER", "user"),
            )
    elif name is None:
        name = current_name or os.getenv("USER", "user")

    # Ensure user_id exists (first-time configuration or old configuration upgrade)
    user_id = current_user_id
    if not user_id:
        user_id = str(uuid.uuid4())
        logger.info(f"✓ Generated user ID: {user_id}")

    # Save local configuration
    local_config["user_id"] = user_id
    local_config["user_name"] = name
    if "created_at" not in local_config:
        local_config["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    local_config["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_local_config(local_config)

    # Check kubectl
    if not k8s_utils.check_kubectl_available():
        logger.error("❌ kubectl not available")
        sys.exit(1)

    logger.info(f"\n✓ Username: {name}")
    logger.info(f"✓ User ID: {user_id}")
    logger.info(f"✓ Namespace: {namespace}")

    # ====== WandB Configuration ======
    logger.info("\n=== WandB Configuration ===")

    wandb_secret_name = f"nexrl-wandb-{user_id}"
    wandb_exists = check_secret_exists(wandb_secret_name, namespace)

    if wandb_exists:
        current_enabled = get_current_secret_value(wandb_secret_name, "enabled", namespace)
        current_host = get_current_secret_value(wandb_secret_name, "host", namespace)
        logger.info(f"Current configuration:")
        logger.info(f"  Status: {'Enabled' if current_enabled == 'true' else 'Disabled'}")
        logger.info(f"  API Key: ***")
        if current_host:
            logger.info(f"  Host: {current_host}")

    if interactive and wandb_api_key is None and enable_wandb is None:
        if click.confirm("Configure WandB?", default=wandb_exists):
            enable_wandb = True
            if wandb_exists:
                update_key = click.confirm("Update WandB API Key?", default=False)
                if update_key:
                    wandb_api_key = click.prompt("WandB API Key", hide_input=True)
                else:
                    wandb_api_key = get_current_secret_value(
                        wandb_secret_name, "api_key", namespace
                    )

                # Ask for host
                if wandb_host is None:
                    use_custom_host = click.confirm("Use custom WandB Host?", default=False)
                    if use_custom_host:
                        default_host = current_host if current_host else "https://api.wandb.ai"
                        wandb_host = click.prompt("WandB Host URL", default=default_host)
                    else:
                        wandb_host = current_host if current_host else "https://api.wandb.ai"
            else:
                wandb_api_key = click.prompt("WandB API Key", hide_input=True)

                # Ask for host
                if wandb_host is None:
                    use_custom_host = click.confirm("Use custom WandB Host?", default=False)
                    if use_custom_host:
                        wandb_host = click.prompt("WandB Host URL", default="https://api.wandb.ai")
                    else:
                        wandb_host = "https://api.wandb.ai"
        else:
            enable_wandb = False
            wandb_api_key = ""
            wandb_host = "https://api.wandb.ai"

    # Apply WandB configuration
    if enable_wandb is not None:
        wandb_enabled_str = "true" if enable_wandb else "false"
        wandb_api_key = wandb_api_key or ""
        wandb_host = wandb_host or "https://api.wandb.ai"

        wandb_yaml = yaml_builder.render_wandb_secret(
            namespace=namespace,
            user_id=user_id,
            api_key=wandb_api_key,
            enabled=wandb_enabled_str,
            host=wandb_host,
        )

        if k8s_utils.apply_yaml(wandb_yaml, namespace):
            logger.info(f"✓ WandB configuration {'enabled' if enable_wandb else 'disabled'}")
            if enable_wandb and wandb_host != "https://api.wandb.ai":
                logger.info(f"  Custom Host: {wandb_host}")
        else:
            logger.error("❌ Failed to save WandB configuration")

    # Complete
    logger.info("\n" + "=" * 60)
    logger.info("✓ Configuration initialization complete!")
    logger.info("=" * 60)
    logger.info("\nTips:")
    logger.info(f"  - User configuration saved to: {NEXRL_CONFIG_FILE}")
    logger.info(f"  - Secret configuration saved to Kubernetes namespace: {namespace}")
    logger.info("  - These configurations will be automatically used when running 'nexrl launch'")
    logger.info("  - To modify, run 'nexrl init' again")


if __name__ == "__main__":
    init()
