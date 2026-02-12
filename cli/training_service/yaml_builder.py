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


def render_driver_volcanojob(
    name: str,
    namespace: str,
    image: str,
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
    wandb_key: str | None = None,
    wandb_host: str | None = None,
    tinker_api_key: str | None = None,
    tinker_base_url: str | None = None,
    weaver_api_key: str | None = None,
    weaver_base_url: str | None = None,
    debug_hydra_overrides: str = "",
) -> str:
    """Render NexRL driver VolcanoJob YAML for training-service mode."""
    template_path = Path(__file__).parent / "templates" / "driver_job.yaml.jinja"
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
        wandb_key=wandb_key,
        wandb_host=wandb_host,
        tinker_api_key=tinker_api_key,
        tinker_base_url=tinker_base_url,
        weaver_api_key=weaver_api_key,
        weaver_base_url=weaver_base_url,
        debug_hydra_overrides=debug_hydra_overrides,
        minAvailable=1,
        globalMinAvailable=1 + num_agent_workers,
    )
