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
Rollout Worker module for NexRL framework
"""

from .agent_rollout_worker import AgentRolloutWorker
from .base_nexau_rollout_worker import (
    BaseEvaluationTarget,
    BaseNexAURolloutWorker,
    EvaluationRunResult,
    Evaluator,
    NexAUEvaluationTarget,
)
from .base_rollout_worker import BaseRolloutWorker
from .pig_latin_rollout_worker import PigLatinRolloutWorker
from .simple_rollout_worker import SimpleRolloutWorker

# Alias for backward compatibility
DefaultNexAURolloutWorker = BaseNexAURolloutWorker

__all__ = [
    "BaseRolloutWorker",
    "AgentRolloutWorker",
    "SimpleRolloutWorker",
    "PigLatinRolloutWorker",
    "BaseNexAURolloutWorker",
    "DefaultNexAURolloutWorker",
    # Evaluator classes (now part of NexAU rollout worker)
    "BaseEvaluationTarget",
    "EvaluationRunResult",
    "Evaluator",
    "NexAUEvaluationTarget",
]
