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
NexRL CLI - Main Entry Point

Command-line interface for managing NexRL training infrastructure.
"""

import click

from . import admin_setup, init, launch


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    NexRL CLI - A tool for managing reinforcement learning training on Kubernetes.

    Commands:
      admin-setup  Initialize cluster environment (run once by admin)
      init         Initialize user configurations (WandB, etc.)
      launch       Launch a training job (run by users)

    Examples:
      # Initialize cluster (admin only, run once)
      nexrl admin-setup

      # Initialize user configurations
      nexrl init

      # Launch a training job
      nexrl launch --job-name my-job --job-path /path/to/job_config

    For more information, visit: https://github.com/nex-agi/NexRL
    """
    pass


# Register commands
cli.add_command(admin_setup.admin_setup, name="admin-setup")
cli.add_command(init.init, name="init")
cli.add_command(launch.launch, name="launch")


if __name__ == "__main__":
    cli()
