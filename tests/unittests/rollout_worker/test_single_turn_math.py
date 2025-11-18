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
Tests for SingleTurnMathAgent using real implementation with mocked LLM client
"""

from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from nexrl.mock import MockLLMServiceClient
from nexrl.rollout_worker.single_turn_math import SingleTurnMathAgent


@pytest.fixture
def single_turn_math_config(rollout_worker_config):
    """Configuration for SingleTurnMathAgent testing"""
    # Add additional fields required by SingleTurnMathAgent
    config = OmegaConf.create(rollout_worker_config)
    config.judge_mode = "rule"  # Use rule-based judging to avoid LLM calls
    return config


@pytest.fixture
def single_turn_math_agent(single_turn_math_config):
    """Create SingleTurnMathAgent with MockLLMServiceClient"""
    # Patch LLMServiceClient to return MockLLMServiceClient instead
    with patch("nexrl.rollout_worker.base_rollout_worker.LLMServiceClient", MockLLMServiceClient):
        worker = SingleTurnMathAgent(single_turn_math_config)
    return worker


def test_single_turn_math_agent_initialization(single_turn_math_agent):
    """Test SingleTurnMathAgent initialization"""
    assert single_turn_math_agent._config is not None
    assert single_turn_math_agent._llm_client is not None
    assert isinstance(single_turn_math_agent._llm_client, MockLLMServiceClient)


def test_single_turn_math_agent_step_missing_prompt(single_turn_math_agent):
    """Test SingleTurnMathAgent step with missing prompt"""
    # Mock the _put_trajectory method
    single_turn_math_agent._put_trajectory = lambda x: "id"

    # Execute a step without prompt
    task = {"id": 1, "reward_model": {"ground_truth": "4"}}
    result = single_turn_math_agent.step(task)

    # Should return None when prompt is missing
    assert result is None
