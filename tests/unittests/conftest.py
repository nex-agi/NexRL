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
Pytest configuration and shared fixtures
"""

import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def basic_config():
    """Basic configuration for testing"""
    return OmegaConf.create(
        {
            "batch_size": 4,
            "vocab_size": 151936,
            "max_prompt_length": 256,
            "max_response_length": 128,
            "model_tag": "test_model",
        }
    )


@pytest.fixture
def data_loader_config():
    """Configuration for data loader testing"""
    return OmegaConf.create(
        {
            "batch_size": 4,
            "mock_api_type": "completion",
            "num_workers": 1,
        }
    )


@pytest.fixture
def rollout_worker_config():
    """Configuration for rollout worker testing"""
    return OmegaConf.create(
        {
            "batch_size": 4,
            "model_tag": "test_model",
            "temperature": 0.7,
            "inference_service": {
                "base_url": "http://localhost:8000",
                "api_key": "mock_key",
                "model": "mock_model",
                "model_tag": "default",
                "max_tokens": 512,
                "max_retries": 1,
                "freeze_for_weight_sync": False,
            },
        }
    )


@pytest.fixture
def train_batch_pool_config():
    """Configuration for train batch pool testing"""
    return OmegaConf.create(
        {
            "max_batches": 10,
            "model_tag": "test_model",
        }
    )
