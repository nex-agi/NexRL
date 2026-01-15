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

import os
from pathlib import Path

from jinja2 import Template


def render_api_server_volcanojob(
    name: str,
    namespace: str,
    image: str,
    nexrl_path: str,
    storage_path: str,
    experiment_path: str,
    queue: str,
    priority_class_name: str,
) -> str:
    """Render API server VolcanoJob YAML."""
    template_path = Path(__file__).parent / "templates" / "api_server_job.yaml.jinja"
    with open(template_path, encoding="utf-8") as f:
        template = Template(f.read())

    return template.render(
        name=name,
        namespace=namespace,
        image=image,
        nexrl_path=nexrl_path,
        nexrl_user=os.getenv("USER", "nexrl"),
        storage_path=storage_path,
        experiment_path=experiment_path,
        queue=queue,
        priorityClassName=priority_class_name,
        reclaimableByVolcano="false",
        minAvailable=1,
        globalMinAvailable=1,
    )


def render_gpu_worker_volcanojob(
    name: str,
    namespace: str,
    identifier: str,
    image: str,
    api_server_url: str,
    nexrl_path: str,
    storage_path: str,
    experiment_path: str,
    world_size: int,
    queue: str,
    priority_class_name: str,
) -> str:
    """Render GPU worker VolcanoJob YAML."""
    template_path = Path(__file__).parent / "templates" / "gpu_worker_job.yaml.jinja"
    with open(template_path, encoding="utf-8") as f:
        template = Template(f.read())

    # Calculate replicas: world_size determines number of pods
    # Since each pod has 8 GPUs, we need world_size/8 pods (rounded up)
    gpus_per_pod = 8
    replicas = (world_size + gpus_per_pod - 1) // gpus_per_pod
    min_available = replicas

    return template.render(
        name=name,
        namespace=namespace,
        identifier=identifier,
        image=image,
        api_server_url=api_server_url,
        nexrl_path=nexrl_path,
        nexrl_user=os.getenv("USER", "nexrl"),
        storage_path=storage_path,
        experiment_path=experiment_path,
        world_size=world_size,
        replicas=replicas,
        minAvailable=min_available,
        globalMinAvailable=min_available,
        queue=queue,
        priorityClassName=priority_class_name,
        reclaimableByVolcano="false",
    )


def render_inference_deployment(
    name: str,
    namespace: str,
    served_model_name: str,
    image: str,
    model_path: str,
    storage_path: str,
    replicas: int,
    gpus_per_replica: int,
    tensor_parallel_size: int = 1,
) -> str:
    """Render inference Deployment YAML (SGLang)."""
    template_path = Path(__file__).parent / "templates" / "inference_deployment.yaml.jinja"
    with open(template_path, encoding="utf-8") as f:
        template = Template(f.read())

    return template.render(
        name=name,
        namespace=namespace,
        served_model_name=served_model_name,
        image=image,
        model_path=model_path,
        storage_path=storage_path,
        replicas=replicas,
        gpus_per_replica=gpus_per_replica,
        tensor_parallel_size=tensor_parallel_size,
    )


def render_driver_volcanojob(
    name: str,
    namespace: str,
    identifier: str,
    image: str,
    api_server_url: str,
    nexrl_path: str,
    storage_path: str,
    experiment_path: str,
    train_config: str,
    served_model_name: str,
    num_agent_workers: int,
    num_agents_per_worker: int,
    user_id: str,
    queue: str,
    priority_class_name: str,
    cmd: str,
    service_account_name: str = "nexrl-job-manager",
    environment_setup_script: str | None = None,
    inference_base_url: str = "",
    wandb_key: str | None = None,
    wandb_host: str | None = None,
    tinker_api_key: str | None = None,
    tinker_base_url: str | None = None,
    weaver_api_key: str | None = None,
    weaver_base_url: str | None = None,
) -> str:
    """Render NexRL driver VolcanoJob YAML."""
    template_path = Path(__file__).parent / "templates" / "driver_job.yaml.jinja"
    with open(template_path, encoding="utf-8") as f:
        template = Template(f.read())

    return template.render(
        name=name,
        namespace=namespace,
        identifier=identifier,
        image=image,
        api_server_url=api_server_url,
        nexrl_path=nexrl_path,
        nexrl_user=os.getenv("USER", "nexrl"),
        storage_path=storage_path,
        experiment_path=experiment_path,
        train_config=train_config,
        served_model_name=served_model_name,
        num_agent_workers=num_agent_workers,
        num_agents_per_worker=num_agents_per_worker,
        user_id=user_id,
        queue=queue,
        priorityClassName=priority_class_name,
        reclaimableByVolcano="false",
        cmd=cmd,
        serviceAccountName=service_account_name,
        environment_setup_script=environment_setup_script,
        inference_base_url=inference_base_url,
        wandb_key=wandb_key,
        wandb_host=wandb_host,
        tinker_api_key=tinker_api_key,
        tinker_base_url=tinker_base_url,
        weaver_api_key=weaver_api_key,
        weaver_base_url=weaver_base_url,
        minAvailable=1,
        globalMinAvailable=1 + num_agent_workers,
    )


def render_rollout_router_deployment(
    name: str,
    namespace: str,
    image: str,
    service_account_name: str | None = None,
    redis_host: str | None = None,
    redis_port: str = "6379",
    redis_username: str | None = None,
    redis_password: str | None = None,
    cpu: str = "8",
    memory: str = "16Gi",
    served_model_name: str | None = None,
) -> str:
    """Render rollout router Deployment YAML (SGLang router)."""
    template_path = Path(__file__).parent / "templates" / "rollout_router_deployment.yaml.jinja"
    with open(template_path, encoding="utf-8") as f:
        template = Template(f.read())

    return template.render(
        name=name,
        namespace=namespace,
        image=image,
        service_account_name=service_account_name,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_username=redis_username,
        redis_password=redis_password,
        cpu=cpu,
        memory=memory,
        served_model_name=served_model_name,
    )
