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


def _find_git_root():
    """Find the git root directory"""
    import os

    # Start from the current file's directory and traverse up
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != "/":
        if os.path.exists(os.path.join(current_dir, ".git")):
            return current_dir
        current_dir = os.path.dirname(current_dir)

    raise ValueError("Cannot find git root")


def capture_git_change():
    """Get the diff information of the current git repository and log it to wandb"""
    import subprocess

    try:
        git_root = _find_git_root()

        # add to safe path
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", git_root])

        # Get git diff
        diff_output = subprocess.check_output(["git", "diff"], stderr=subprocess.STDOUT).decode(
            "utf-8"
        )

        # Get git status
        status_output = subprocess.check_output(["git", "status"], stderr=subprocess.STDOUT).decode(
            "utf-8"
        )

        # Get current commit hash
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
            .decode("utf-8")
            .strip()
        )

        # Get current branch name
        branch_name = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )

        data_to_log = {
            "git/root": git_root,
            "git/diff": diff_output,
            "git/status": status_output,
            "git/commit_hash": commit_hash,
            "git/branch": branch_name,
            "git/working_dir": os.getcwd(),
        }
        return data_to_log
    except subprocess.CalledProcessError as e:
        print(f"Error getting git information: {e.output}")
    except Exception as e:
        print(f"Error logging git information to wandb: {str(e)}")
