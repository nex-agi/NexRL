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

"""Training service mode specific configuration utilities."""

# Import common utilities
import sys
from pathlib import Path

# Add parent directory to path to import common utilities
CLI_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CLI_DIR))


def load_model_name(cfg: dict) -> str:
    """Load model name from service.inference_service.model in config."""
    service_cfg = cfg.get("service") or {}
    if not isinstance(service_cfg, dict):
        raise ValueError("service must be a mapping")

    inference_service_cfg = service_cfg.get("inference_service") or {}
    if not isinstance(inference_service_cfg, dict):
        raise ValueError("service.inference_service must be a mapping")

    model_name = inference_service_cfg.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("service.inference_service.model must be a non-empty string")

    return model_name
