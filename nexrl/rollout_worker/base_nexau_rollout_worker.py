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
Base NexAU Rollout Worker for NexRL framework.
Contains common functionality shared by all NexAU-based rollout workers.
"""

import importlib.util
import json
import logging
import os
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import yaml  # type: ignore
from omegaconf import DictConfig
from transformers import AutoTokenizer

from ..nexrl_types import Trajectory
from ..utils.path_utils import (
    resolve_agent_config_path,
    resolve_evaluator_module_path,
)
from .base_rollout_worker import BaseRolloutWorker

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Target Data Structures
# =============================================================================


@dataclass
class BaseEvaluationTarget:
    """Base class for evaluation targets."""

    final_answer: str


@dataclass
class NexAUEvaluationTarget(BaseEvaluationTarget):
    """Evaluation target for NexAU agents."""

    final_answer: str  # Final answer produced by the agent
    observation: list[dict[str, Any]]  # Complete execution trajectory containing
    # all intermediate steps and observations


# =============================================================================
# Evaluation Result Data Structure
# =============================================================================


@dataclass
class EvaluationRunResult:
    """
    Evaluator execution result.

    Attributes:
        reward: Evaluation score
        ground_truth: Ground truth answer
        metrics: Additional metrics (must be scalar floats)
        extra_info: Additional information (can be any type)
    """

    reward: float = 0.0
    ground_truth: str = ""
    metrics: dict[str, float] = field(
        default_factory=dict
    )  # pyright: ignore[reportUnknownVariableType]
    extra_info: dict[str, Any] = field(
        default_factory=dict
    )  # pyright: ignore[reportUnknownVariableType]


# =============================================================================
# Evaluator Base Class
# =============================================================================


class Evaluator(ABC):
    """
    Abstract base class for evaluators.

    Subclasses must implement the evaluate method to define specific evaluation logic.
    """

    def __init__(self) -> None:
        """Initialize the evaluator."""

    @abstractmethod
    def evaluate(
        self,
        data: dict[str, Any],
        evaluation_target: BaseEvaluationTarget,
    ) -> EvaluationRunResult:
        """
        Perform evaluation.

        Args:
            data: Input data, should contain at least query information
            evaluation_target: Agent output target for evaluation

        Returns:
            EvaluationRunResult: Evaluation result
        """
        # Default implementation: simple exact match evaluation
        ground_truth = data.get("ground_truth", "")
        reward = 0.0

        if isinstance(ground_truth, str):
            reward = float(evaluation_target.final_answer == ground_truth)

        return EvaluationRunResult(
            reward=reward,
            ground_truth=str(ground_truth),
            metrics={},
            extra_info={},
        )


class BaseNexAURolloutWorker(BaseRolloutWorker):
    """
    Base class for NexAU-based rollout workers.

    Provides common functionality:
    - Agent config loading
    - Evaluator loading
    - Trace processing
    - Workspace setup
    - Token/loss mask generation
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the BaseNexAURolloutWorker.

        Args:
            config: Configuration with keys:
                - nexau_agent_config_path: Path to agent config YAML
                - evaluator_module_path: Path to evaluator module (format: "path.py:Class")
                - nexau_agent_workspace: Workspace directory to add to sys.path
                - trace_path: Directory to save trace files
                - save_trace: Whether to save traces (default: True)
                - tokenizer: Path to tokenizer
                - temperature: Sampling temperature
                - enable_trace_prefix_merge: Whether to merge trajectory prefixes
        """
        super().__init__(config)

        # Get config file path for relative path resolution (required)
        config_file_path = getattr(config, "_config_file_path", None)

        if not config_file_path:
            raise ValueError(
                "Config file path not available. Ensure the config has _config_file_path attribute set."
            )

        # Resolve paths relative to config file
        self.nexau_agent_config_path = resolve_agent_config_path(
            config.get("nexau_agent_config_path", None), recipe_config_path=config_file_path
        )

        # Resolve evaluator path relative to config file
        self.evaluator_module_path = resolve_evaluator_module_path(
            config.get("evaluator_module_path", None),
            config_file_path=config_file_path,
        )

        self.nexau_agent_workspace = config.get("nexau_agent_workspace", None)
        if self.nexau_agent_workspace:
            self.nexau_agent_workspace = resolve_agent_config_path(
                self.nexau_agent_workspace, recipe_config_path=config_file_path
            )

        self.task_name = config.get("task_name", None)
        self.trace_path = config.get("trace_path", None)
        self.save_trace_enabled = config.get("save_trace", False)

        # Initialize tokenizer
        tokenizer_path = config.get("tokenizer", None)
        if tokenizer_path is None:
            raise ValueError("Tokenizer path must be provided in config for NexAU workers")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Setup trace directory
        if self.save_trace_enabled and self.trace_path:
            os.makedirs(self.trace_path, exist_ok=True)
        else:
            self.trace_path = None

        # Setup workspace
        self._setup_workspace()

        # Load evaluator
        self.evaluator = self._load_evaluator()

    def _setup_workspace(self):
        """Add workspace path to sys.path for imports."""
        if self.nexau_agent_workspace and os.path.exists(self.nexau_agent_workspace):
            workspace_path = os.path.abspath(self.nexau_agent_workspace)
            if workspace_path not in sys.path:
                sys.path.insert(0, workspace_path)
                logger.info(f"Added workspace to sys.path: {workspace_path}")

    def _load_evaluator(self):
        """Load evaluator class from module path."""
        if not self.evaluator_module_path:
            logger.warning("No evaluator_module_path provided")
            return None

        if ":" not in self.evaluator_module_path:
            raise ValueError(
                f"Invalid evaluator module path format. "
                f"Expected 'path:ClassName', got: {self.evaluator_module_path}"
            )

        file_path, class_name = self.evaluator_module_path.rsplit(":", 1)
        abs_path = os.path.abspath(file_path)

        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Evaluator file not found: {abs_path}")

        # Dynamically load module
        module_name = f"_evaluator_module_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, abs_path)

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load evaluator module from {abs_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get evaluator class and instantiate
        evaluator_class = getattr(module, class_name, None)
        if evaluator_class is None:
            raise AttributeError(f"Class '{class_name}' not found in {abs_path}")

        return evaluator_class()

    def load_agent_from_config(self, custom_llm_client_provider=None):
        """
        Load NexAU agent from config file.

        Args:
            custom_llm_client_provider: Optional function to override LLM client

        Returns:
            Loaded agent instance
        """
        if not self.nexau_agent_config_path:
            raise ValueError("nexau_agent_config_path not configured")

        if not os.path.exists(self.nexau_agent_config_path):
            raise FileNotFoundError(f"Agent config not found: {self.nexau_agent_config_path}")

        from nexau.archs.config.config_loader import load_agent_config

        agent = load_agent_config(self.nexau_agent_config_path)
        # Setup custom LLM client if provided
        if custom_llm_client_provider:
            # Read agent name from config
            with open(self.nexau_agent_config_path, encoding="utf-8") as f:
                agent_config = yaml.safe_load(f)
            main_agent_name = agent_config.get("name")

            # Provide custom client override
            def client_provider(agent_name: str):
                if agent_name == main_agent_name:
                    return custom_llm_client_provider()
                return None

            # Note: This assumes agent.run() supports custom_llm_client_provider
            # If not, subclasses need to handle this differently
            return agent, client_provider

        return agent, None

    def child_processor(self, child: dict, trajectories: list[dict]):
        """
        Process a single trace child node and extract trajectory information.

        Args:
            child: Trace child node
            trajectories: List to append trajectory info to
        """
        if child.get("type") != "LLM":
            return

        outputs = child.get("outputs", {})

        # Extract response tokens
        if "nexrl_train" in outputs:
            # Process children first if present (even when nexrl_train exists)
            if len(child.get("children", [])) > 0:
                for sub_child in child["children"]:
                    self.child_processor(sub_child, trajectories)
            response_tokens = outputs["nexrl_train"].get("response_tokens", [])
        else:
            # Try to extract from logprobs
            choices = outputs.get("choices", [])
            if not choices:
                return

            logprobs = choices[0].get("logprobs", {})
            if "tokens" in logprobs:
                response_tokens = [int(t) for t in logprobs["tokens"] if t is not None]
            elif "content" in logprobs:
                response_tokens = [
                    int(item["token"]) for item in logprobs["content"] if item is not None
                ]
            else:
                logger.warning(f"Cannot extract response tokens from logprobs: {logprobs}")
                return

        # Extract other info
        choices = outputs.get("choices", [])
        trajectory_info = {
            "prompt_messages": child.get("inputs", {}).get("messages", []),
            "tools": child.get("inputs", {}).get("tools", []),
            "response_message": choices[0].get("message", {}) if choices else {},
            "response_tokens": response_tokens,
            "finish_reason": choices[0].get("finish_reason", "") if choices else "",
            **outputs,
        }
        trajectories.append(trajectory_info)

    def trace_processor(self, traces: list[dict]) -> list[dict]:
        """
        Process traces and extract trajectory information.

        Args:
            traces: Raw traces from InMemoryTracer

        Returns:
            List of trajectory dictionaries
        """
        trajectories: list[dict] = []
        for trace in traces:
            for child in trace.get("children", []):
                self.child_processor(child, trajectories)
        return trajectories

    def add_loss_mask(
        self,
        prompt_tokens: list[int],
        response_tokens: list[int],
        response_logprobs: list[float] | None = None,
    ) -> dict[str, list]:
        """
        Generate loss mask and combine prompt+response tokens.

        Args:
            prompt_tokens: Prompt token IDs
            response_tokens: Response token IDs
            response_logprobs: Response log probabilities (optional)

        Returns:
            Dictionary with keys:
                - tokens: Combined token sequence
                - loss_mask: Mask (0 for prompt, 1 for response)
                - logprobs: Log probabilities (0.0 for prompt, actual for response)
        """
        if response_logprobs is None:
            response_logprobs = []

        # Create masks and logprobs
        prompt_loss_mask = [0] * len(prompt_tokens)
        prompt_logprobs = [0.0] * len(prompt_tokens)
        response_loss_mask = [1] * len(response_tokens)

        # Combine
        tokens = prompt_tokens + response_tokens
        loss_mask = prompt_loss_mask + response_loss_mask
        logprobs = prompt_logprobs + response_logprobs

        assert len(tokens) == len(
            loss_mask
        ), f"Length mismatch: tokens={len(tokens)}, loss_mask={len(loss_mask)}"
        assert len(tokens) == len(
            logprobs
        ), f"Length mismatch: tokens={len(tokens)}, logprobs={len(logprobs)}"

        return {
            "tokens": tokens,
            "loss_mask": loss_mask,
            "logprobs": logprobs,
        }

    def trace_prefix_merge(self, trajectories: list[dict]) -> list[dict]:
        """
        Merge trajectories that share a common prefix.

        When a trajectory's prompt matches the full sequence of the previous trajectory,
        merge them by appending new tokens.

        Args:
            trajectories: List of trajectory dicts

        Returns:
            List of merged trajectories
        """
        if not trajectories:
            return []

        merged = [trajectories[0]]
        current_prefix = merged[0]["tokens"]

        for traj in trajectories[1:]:
            traj_tokens = traj["tokens"]

            # Check if current trajectory starts with previous prefix
            if traj_tokens[: len(current_prefix)] == current_prefix:
                # Merge: append new tokens
                new_tokens = traj_tokens[len(current_prefix) :]
                new_loss_mask = traj["loss_mask"][len(current_prefix) :]
                new_logprobs = traj["logprobs"][len(current_prefix) :]

                merged[-1]["tokens"] += new_tokens
                merged[-1]["loss_mask"] += new_loss_mask
                merged[-1]["logprobs"] += new_logprobs

                current_prefix = merged[-1]["tokens"]
            else:
                # No match, add as new trajectory
                merged.append(traj)
                current_prefix = traj["tokens"]

        return merged

    def save_trace(self, group_id: str, run_id: int, step: int, logs: dict[str, Any]):
        """Save trace to disk."""
        if not self.save_trace_enabled or not self.trace_path:
            return

        step = self._activity_tracker.get_training_step()
        uuid_str = str(uuid.uuid4())
        filename = f"group_{group_id}_run_{run_id}_step_{step}-uuid-{uuid_str}.json"
        filepath = os.path.join(self.trace_path, filename)

        logger.debug(f"Saving trace to: {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    def run_agent(self, task: dict[str, Any]) -> tuple[Any, EvaluationRunResult]:
        """
        Run the agent and evaluate the result.

        This is a default implementation that works for most tasks.
        Subclasses can override this method for task-specific behavior (e.g., custom query formatting).

        Args:
            task: Task dictionary containing input data (must have 'prompt', 'query', or 'question' field)

        Returns:
            Tuple of (agent_output, evaluation_result)
        """
        from nexau.archs.tracer.adapters import InMemoryTracer

        # Extract query (can be 'prompt', 'query', or 'question' field)
        query = task.get("prompt") or task.get("query") or task.get("question", "")
        if not query:
            raise ValueError("Task must contain 'prompt', 'query', or 'question' field")

        # Load agent with custom LLM client
        agent, client_provider_func = self.load_agent_from_config(
            custom_llm_client_provider=lambda: self._inference_client
        )

        # Run agent
        response = agent.run(message=query, custom_llm_client_provider=client_provider_func)

        # Extract traces
        traces = []
        for tracer in agent.config.tracers:
            if isinstance(tracer, InMemoryTracer):
                traces = tracer.dump_traces()
                break

        # Process traces into trajectory format
        trajectories = self.trace_processor(traces)

        # Create agent output structure
        @dataclass
        class AgentOutput:
            final_answer: str
            observation: list
            rl_params: dict = field(default_factory=dict)

        agent_output = AgentOutput(
            final_answer=response, observation=agent.history, rl_params={"trajectory": trajectories}
        )

        # Evaluate
        if self.evaluator is None:
            raise ValueError("Evaluator not initialized")

        evaluation_result = self.evaluator.evaluate(
            task,
            NexAUEvaluationTarget(
                final_answer=agent_output.final_answer, observation=agent_output.observation
            ),
        )

        # Add reward and score to each trajectory
        for traj in trajectories:
            traj["reward"] = evaluation_result.reward
            traj["score"] = {
                "reward_score": evaluation_result.reward,
                **evaluation_result.metrics,
            }

        return agent_output, evaluation_result

    def rollout(self, task: dict[str, Any]) -> str | None:
        """
        Execute rollout: run agent, process traces, create trajectories.

        Args:
            task: Task dictionary

        Returns:
            Result status from _put_trajectory
        """
        logger.debug(f"Starting rollout for task: {task.get('query', '')[:50]}")

        # Run agent and get evaluation result
        agent_output, evaluation_result = self.run_agent(task)

        # Save trace
        self.save_trace(
            task.get("group_id", ""),
            task.get("run_id", 0),
            task.get("step", 0),
            {
                "history": getattr(agent_output, "observation", []),
                "reward": evaluation_result.reward,
                "metrics": evaluation_result.metrics,
                "extra_info": evaluation_result.extra_info,
            },
        )

        # Process trajectories
        rl_params = getattr(agent_output, "rl_params", {})
        nexau_trajectories = rl_params.get("trajectory", [])

        processed_trajectories = []
        for nexau_traj in nexau_trajectories:
            nexrl_train = nexau_traj.get("nexrl_train", {})
            processed = self.add_loss_mask(
                prompt_tokens=nexrl_train.get("prompt_tokens", []),
                response_tokens=nexrl_train.get("response_tokens", []),
                response_logprobs=nexrl_train.get("response_logprobs", []),
            )
            processed["finish_reason"] = nexau_traj.get("finish_reason", "stop")
            processed_trajectories.append(processed)

        # Merge trajectories if enabled
        if self._config.get("enable_trace_prefix_merge", True):
            merged_trajectories = self.trace_prefix_merge(processed_trajectories)
        else:
            merged_trajectories = processed_trajectories

        logger.debug(
            f"Processed {len(processed_trajectories)} trajectories, "
            f"merged to {len(merged_trajectories)}"
        )

        # Create Trajectory objects and put into pool
        last_result = None
        for traj_dict in merged_trajectories:
            trajectory = Trajectory(
                tokens=traj_dict["tokens"],
                loss_mask=traj_dict["loss_mask"],
                reward=evaluation_result.reward,
                is_val=task.get("is_val", False),
                extra_fields={
                    "ground_truth": evaluation_result.ground_truth,
                    "group_id": task.get("group_id", ""),
                    "run_id": task.get("run_id", 0),
                    "task_id": task.get("task_id", 0),
                    "temperature": self._config.get("temperature", 1.0),
                    "finish_reason": traj_dict.get("finish_reason", "stop"),
                    "score": {"reward_score": evaluation_result.reward},
                    "logprobs": traj_dict["logprobs"],
                },
            )

            # Add extra task fields
            excluded_keys = {"prompt", "ground_truth", "is_val"}
            for k, v in task.items():
                if k not in excluded_keys:
                    trajectory[k] = v

            last_result = self._put_trajectory(trajectory)

        return last_result
