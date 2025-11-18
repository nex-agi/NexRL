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
YAML template rendering for NexRL CLI
"""

import logging
import os
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

# Get templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")

# Create Jinja2 environment
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(template_name: str, **kwargs) -> str:
    """Render Jinja2 template"""
    try:
        template = jinja_env.get_template(template_name)
        rendered = template.render(**kwargs)
        logger.debug(f"Rendered template: {template_name}")
        return rendered
    except Exception as e:
        logger.error(f"Failed to render template {template_name}: {e}")
        raise


def render_redis_secret(namespace: str, redis_config: Dict[str, str]) -> str:
    """Render Redis Secret YAML"""
    return render_template(
        "redis_secret.yaml.jinja",
        NAMESPACE=namespace,
        REDIS_HOST=redis_config.get("host", ""),
        REDIS_PORT=redis_config.get("port", "6379"),
        REDIS_USERNAME=redis_config.get("username", ""),
        REDIS_PASSWORD=redis_config.get("password", ""),
    )


def render_wandb_secret(
    namespace: str, user_id: str, api_key: str, enabled: str, host: str = "https://api.wandb.ai"
) -> str:
    """Render WandB Secret YAML"""
    return render_template(
        "wandb_secret.yaml.jinja",
        NAMESPACE=namespace,
        USER_ID=user_id,
        API_KEY=api_key,
        ENABLED=enabled,
        HOST=host,
    )


def render_sglang_router_rbac(namespace: str) -> str:
    """Render SGLang Router RBAC YAML"""
    return render_template("sglang_router_rbac.yaml.jinja", NAMESPACE=namespace)


def render_train_router_deployment(namespace: str, image: str, **kwargs) -> str:
    """Render Train Router Deployment YAML"""
    return render_template(
        "train_router_deployment.yaml.jinja",
        NAMESPACE=namespace,
        TRAIN_ROUTER_IMAGE=image,
        **kwargs,
    )


def render_train_router_service(namespace: str) -> str:
    """Render Train Router Service YAML"""
    return render_template("train_router_service.yaml.jinja", NAMESPACE=namespace)


def render_rollout_router_deployment(
    namespace: str, served_model_name: str, image: str, **kwargs
) -> str:
    """Render Rollout Router Deployment YAML"""
    return render_template(
        "rollout_router_deployment.yaml.jinja",
        NAMESPACE=namespace,
        SERVED_MODEL_NAME=served_model_name,
        ROLLOUT_ROUTER_IMAGE=image,
        **kwargs,
    )


def render_rollout_router_service(namespace: str) -> str:
    """Render Rollout Router Service YAML"""
    return render_template("rollout_router_service.yaml.jinja", NAMESPACE=namespace)


def render_routers_configmap(namespace: str) -> str:
    """Render Routers ConfigMap YAML"""
    return render_template("routers_configmap.yaml.jinja", NAMESPACE=namespace)


def render_admin_configmap(namespace: str, config: Dict[str, Any]) -> str:
    """Render Admin ConfigMap YAML"""
    return render_template("admin_configmap.yaml.jinja", NAMESPACE=namespace, **config)


def render_worker_volcanojob(
    job_name: str,
    namespace: str,
    identifier: str,
    worker_image: str,
    train_router_url: str,
    worker_config_path: str,
    storage_path: str,
    world_size: int,
    gpus_per_pod: int,
    **kwargs,
) -> str:
    """Render Worker VolcanoJob YAML"""
    return render_template(
        "worker_volcanojob.yaml.jinja",
        JOB_NAME=job_name,
        NAMESPACE=namespace,
        IDENTIFIER=identifier,
        WORKER_IMAGE=worker_image,
        TRAIN_ROUTER_URL=train_router_url,
        WORKER_CONFIG_PATH=worker_config_path,
        STORAGE_PATH=storage_path,
        WORLD_SIZE=world_size,
        GPUS_PER_POD=gpus_per_pod,
        **kwargs,
    )


def render_nexrl_controller_volcanojob(
    job_name: str,
    namespace: str,
    identifier: str,
    controller_image: str,
    train_router_url: str,
    rollout_router_url: str,
    storage_path: str,
    job_config_path: str,
    **kwargs,
) -> str:
    """Render NexRL Controller VolcanoJob YAML"""
    return render_template(
        "nexrl_controller_volcanojob.yaml.jinja",
        JOB_NAME=job_name,
        NAMESPACE=namespace,
        IDENTIFIER=identifier,
        CONTROLLER_IMAGE=controller_image,
        TRAIN_ROUTER_URL=train_router_url,
        ROLLOUT_ROUTER_URL=rollout_router_url,
        STORAGE_PATH=storage_path,
        JOB_CONFIG_PATH=job_config_path,
        **kwargs,
    )


def render_rollout_workers_deployment(
    job_name: str,
    namespace: str,
    served_model_name: str,
    inference_image: str,
    model_path: str,
    storage_path: str,
    inference_gpus: int,
    **kwargs,
) -> str:
    """Render Rollout Workers Deployment YAML"""
    return render_template(
        "rollout_workers.yaml.jinja",
        JOB_NAME=job_name,
        NAMESPACE=namespace,
        SERVED_MODEL_NAME=served_model_name,
        INFERENCE_IMAGE=inference_image,
        MODEL_PATH=model_path,
        STORAGE_PATH=storage_path,
        INFERENCE_GPUS=inference_gpus,
        **kwargs,
    )


def render_volcano_queue(
    queue_name: str,
    weight: int = 1,
    capability: Dict[str, str] = {},
    reclaimable: bool = True,
) -> str:
    """Render Volcano Queue YAML"""
    return render_template(
        "volcano_queue.yaml.jinja",
        QUEUE_NAME=queue_name,
        WEIGHT=weight,
        CAPABILITY=capability,
        RECLAIMABLE=reclaimable,
    )


def validate_required_fields(
    template_name: str, required_fields: list, kwargs: Dict[str, Any]
) -> bool:
    """Validate required fields"""
    missing_fields = []
    for field in required_fields:
        if field not in kwargs or kwargs[field] is None:
            missing_fields.append(field)

    if missing_fields:
        logger.error(f"Template {template_name} missing required fields: {missing_fields}")
        return False

    return True
