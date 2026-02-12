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

"""Debug utilities for trajectory dump and load workflow."""

from datetime import datetime
from pathlib import Path


def find_most_recent_experiment_with_trajectory(
    logs_dir: Path, experiment_name: str, step: int = 0
) -> tuple[Path, Path] | None:
    """Find the most recent run that HAS a trajectory file.

    Algorithm:
    1. List all run directories in logs/<experiment_name>/ (format: YYYYMMDD-HHMMSS)
    2. For EACH run, check if <run_dir>/debug_dump/trajectory/step_{step:06d}.pt exists
    3. FILTER to only keep runs that have trajectories
    4. SORT filtered runs by timestamp (directory name)
    5. Return (run_path, trajectory_path) for the most recent, or None

    This is important because runs can reuse trajectories without generating new ones:
    - Run 1: Normal rollout → generates trajectory ✓
    - Run 2: Reuses Run 1's trajectory (mock mode) → no trajectory ✗
    - Run 3: Should find Run 1, not Run 2

    Args:
        logs_dir: Base logs directory (e.g., NEXRL_PATH/logs)
        experiment_name: Name of the experiment
        step: Training step number (default: 0)

    Returns:
        (run_directory, trajectory_file_path) if found, None otherwise
    """
    experiment_dir = logs_dir / experiment_name
    if not experiment_dir.exists():
        return None

    # List all subdirectories (run directories)
    try:
        run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
    except (OSError, PermissionError):
        return None

    # Filter: keep only runs that have trajectories
    runs_with_trajectories = []
    for run_dir in run_dirs:
        trajectory_path = run_dir / "debug_dump" / "trajectory" / f"step_{step:06d}.pt"
        if trajectory_path.exists() and trajectory_path.is_file():
            runs_with_trajectories.append((run_dir, trajectory_path))

    if not runs_with_trajectories:
        return None

    # Sort by directory name (timestamp format: YYYYMMDD-HHMMSS)
    # Most recent first
    runs_with_trajectories.sort(key=lambda x: x[0].name, reverse=True)

    return runs_with_trajectories[0]


def check_trajectory_exists(baseline_path: Path, step: int = 0) -> Path | None:
    """Check if trajectory exists at specific baseline path.

    Args:
        baseline_path: Path to the baseline experiment run directory
        step: Training step number (default: 0)

    Returns:
        Full path to trajectory file if exists, None otherwise
    """
    trajectory_path = baseline_path / "debug_dump" / "trajectory" / f"step_{step:06d}.pt"
    if trajectory_path.exists() and trajectory_path.is_file():
        return trajectory_path
    return None


def prompt_user_confirmation(run_path: Path, trajectory_path: Path) -> bool:
    """Display trajectory info and prompt user for confirmation.

    Args:
        run_path: Path to the run directory
        trajectory_path: Path to the trajectory file

    Returns:
        True if user confirms (Y/y/yes), False otherwise
    """
    # Get file info
    print(f"[INFO] Found trajectory from run: {run_path.name}")
    try:
        stat = trajectory_path.stat()
        size_mb = stat.st_size / (1024 * 1024)
        modified_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        print(f"Trajectory: {trajectory_path}")
        print(f"Size: {size_mb:.1f} MB | Modified: {modified_time}")
    except (OSError, PermissionError):
        print(f"Trajectory: {trajectory_path}")

    # Prompt user
    while True:
        response = input("Use this trajectory for mock rollout? [Y/n]: ").strip().lower()
        if response in ["y", "yes", ""]:  # Empty input defaults to yes
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter Y or N")


def build_debug_overrides(trajectory_path: Path | None) -> str:
    """Build Hydra CLI overrides string for debug configuration.

    Args:
        trajectory_path: Path to trajectory file if using mock mode, None for dump mode

    Returns:
        Space-separated string of Hydra overrides
    """
    overrides = []

    if trajectory_path:
        # Mock mode: reuse existing trajectory
        # Use ++ to force override existing fields, + for new fields
        overrides.append("+debug.enable_data_dump=false")
        overrides.append("++rollout_worker.type=mock")
        overrides.append("++rollout_worker.need_llm_inference=false")
        # Use absolute path to avoid resolution issues
        abs_path = trajectory_path.resolve()
        overrides.append(f"++rollout_worker.trajectory_load_path={abs_path}")
        overrides.append("++rollout_worker.trajectory_format=pt")
        # Reduce to 1 worker in mock mode to avoid loading trajectory on multiple workers
        # (Mock workers just replay data, no benefit from parallelism)
        overrides.append("++rollout_worker.resource.num_workers=1")
    else:
        # Dump mode: generate new trajectory
        overrides.append("+debug.enable_data_dump=true")
        overrides.append("+debug.dump_options.trajectory=true")
        overrides.append("+debug.dump_options.old_log_probs=false")
        overrides.append("+debug.dump_options.forward_data=false")
        overrides.append("+debug.dump_options.loss=false")
        overrides.append("+debug.dump_dir=${oc.env:EXPERIMENT_PATH}/debug_dump")

    return " ".join(overrides)
