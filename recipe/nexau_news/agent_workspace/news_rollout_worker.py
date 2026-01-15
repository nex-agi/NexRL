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
News NexAU Rollout Worker for NexRL framework.
Specialized worker for news classification tasks.
"""

import logging
from typing import Any

from nexau.archs.tracer.adapters import InMemoryTracer

from nexrl.rollout_worker import EvaluationRunResult, NexAUEvaluationTarget
from nexrl.rollout_worker.base_nexau_rollout_worker import BaseNexAURolloutWorker

logger = logging.getLogger(__name__)


class NewsNexAURolloutWorker(BaseNexAURolloutWorker):
    """
    News-specific NexAU Rollout Worker.

    Handles news classification tasks with specialized query formatting.
    """

    def _format_news_query(self, data_item: dict[str, Any]) -> str:
        """
        Format news data into a query string.

        Args:
            data_item: News data item containing:
                - company_name: Target company/entity name
                - title: News title
                - summary: News summary
                - created_at: Current time
                - date_str: News date
                - tags: News tags
                - source: News source

        Returns:
            Formatted query string
        """
        company_name = data_item.get("company_name", "")
        news_title = "新闻标题: " + data_item.get("title", "")
        news_summary = "新闻摘要: " + data_item.get("summary", "")

        # Historical news context (currently empty)
        news_has_been_pushed = """<之前已经推送的新闻>\n \n</之前已经推送的新闻>"""

        time_now = "现在时间：" + data_item.get("created_at", "")
        time_news = "新闻时间：" + data_item.get("date_str", "")
        goal_entity = "目标企业/人/组织名: " + company_name
        news_tags = "新闻标签: " + data_item.get("tags", "")
        news_source = "新闻来源: " + data_item.get("source", "")

        news_to_be_judged = (
            f"<需要判断的新闻>\n"
            f"    {time_now}\n"
            f"    {time_news}\n"
            f"    {goal_entity}\n"
            f"    {news_title}\n"
            f"    {news_summary}\n"
            f"    {news_tags}\n"
            f"    {news_source}\n"
            f"</需要判断的新闻>"
        )

        return f"{news_has_been_pushed}\n{news_to_be_judged}"

    def run_agent(self, task: dict[str, Any]) -> tuple[Any, EvaluationRunResult]:
        """
        Run the news agent and evaluate the result.

        Args:
            task: Task dictionary containing news data

        Returns:
            Tuple of (agent_output, evaluation_result)
        """
        # Format query
        query = self._format_news_query(task)

        # Load agent with custom LLM client
        agent, client_provider_func = self.load_agent_from_config(
            custom_llm_client_provider=lambda: self._inference_client
        )

        # Run agent
        response = agent.run(query, custom_llm_client_provider=client_provider_func)

        # Extract traces
        traces = []
        for tracer in agent.config.tracers:
            if isinstance(tracer, InMemoryTracer):
                traces = tracer.dump_traces()
                break

        # Process traces into trajectory format
        trajectories = self.trace_processor(traces)

        # Create agent output structure
        from dataclasses import dataclass, field

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
