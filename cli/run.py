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
Unified entry point for NexRL training (Open-Source Version).

Two modes available:
  - self-hosted: Runs training backend (API server, GPU workers) on your cluster
  - training-service: Uses external training service (only runs driver)
"""

import argparse
import sys
from pathlib import Path

CLI_DIR = Path(__file__).parent
sys.path.insert(0, str(CLI_DIR.parent))


def main():
    parser = argparse.ArgumentParser(
        description="NexRL Training Launcher (Open-Source)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  self-hosted        Run training backend (API server, GPU workers, inference)
                     on your Kubernetes cluster. Full control over infrastructure.

  training-service   Use external training service. Only launches driver pod.
                     Lighter weight, relies on external infrastructure.

Examples:
  # Self-hosted mode: full control
  python cli/run.py --mode self-hosted --train-config examples/job/rl_train.yaml --run-nexrl
  python cli/run.py -m self-hosted --train-config examples/job/rl_train.yaml --run-nexrl

  # Training service mode: external backend
  python cli/run.py --mode training-service --train-config examples/job/rl_train.yaml --run-nexrl
  python cli/run.py -m training-service --train-config examples/job/rl_train.yaml --run-nexrl

  # Self-hosted with external inference service
  python cli/run.py -m self-hosted --train-config examples/job/rl_train.yaml --inference-url my-service:8000
        """,
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["self-hosted", "training-service"],
        required=True,
        metavar="MODE",
        help="Deployment mode (required): 'self-hosted' or 'training-service'",
    )
    parser.add_argument(
        "--train-config",
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to training config YAML (required)",
    )
    parser.add_argument("--tag", "-t", type=str, default="", help="Custom tag for job names")
    parser.add_argument(
        "--run-nexrl", "-r", action="store_true", help="Run NexRL training automatically"
    )
    parser.add_argument(
        "--inference-url",
        type=str,
        default="",
        help="[self-hosted only] Use existing inference service URL (skips launching inference). Do not include http:// prefix.",
    )

    args = parser.parse_args()

    # Simple validation
    if args.mode == "training-service" and args.inference_url:
        parser.error("--inference-url is only for self-hosted mode")

    print(f"[INFO] Running in {args.mode} mode (open-source)")

    # Forward to appropriate launcher
    if args.mode == "self-hosted":
        from cli.self_hosted.run import main as launcher_main
    else:
        from cli.training_service.run import main as launcher_main

    # Reconstruct argv for the launcher (remove mode argument)
    sys.argv = ["launcher", "--train-config", args.train_config]
    if args.run_nexrl:
        sys.argv.append("--run-nexrl")
    if args.tag:
        sys.argv.extend(["--tag", args.tag])
    if args.inference_url:
        sys.argv.extend(["--inference-url", args.inference_url])

    launcher_main()


if __name__ == "__main__":
    main()
