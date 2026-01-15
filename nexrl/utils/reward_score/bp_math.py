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
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to
validate answers when necessary.
"""

import json
import logging
import os
import random
from typing import Any

import openai

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass

from .evaluation_utils.deepscaler_utlis import (
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy,
)
from .evaluation_utils.judge_prompt import scale_verify_promptv5
from .evaluation_utils.llm_judge import call_oai_rm_llm
from .evaluation_utils.reward_config import RewardConfig, RewardFn

THINKING_START = "<think>"
THINKING_END = "</think>"

logger = logging.getLogger(__name__)


class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):

        # Extract solution based on thinking format usage.
        # Logic:
        # 1. If THINKING_START exists → model is using thinking format
        #    - If THINKING_END also exists → extract solution after </think>
        #    - If THINKING_END missing → truncated thinking, return format_error
        # 2. If THINKING_START doesn't exist → model doesn't use thinking format
        #    - Use full response as solution
        combined_text = prompt + model_response
        if THINKING_START in combined_text:
            # Model is using thinking format
            if THINKING_END in combined_text:
                # Complete thinking block - extract solution after </think>
                model_solution = model_response.split(THINKING_END)[1]
            else:
                # Thinking started but not finished - truncated
                logger.debug(
                    f"Reward = {self.config.format_error_reward}, due to thinking truncation"
                )
                return self.config.format_error_reward, False
        else:
            # Model doesn't use thinking format - use full response
            logger.debug("No thinking tags found, using full response as solution")
            model_solution = model_response

        model_answer = extract_answer(model_solution)
        logger.debug(f"model_answer: {model_answer}")
        if model_answer is None:
            logger.debug(f"Reward = {self.config.format_error_reward}, due to model_answer is None")
            return self.config.format_error_reward, False

        # Convert single answer to list for uniform processing
        if isinstance(ground_truth, (str, float, int)):
            ground_truths = [ground_truth]
        else:
            ground_truths = ground_truth

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        # print('ground_truths', ground_truths)
        # print('processed_ground_truths', processed_ground_truths)

        if not processed_ground_truths:
            logger.debug(
                f"Reward = {self.config.unk_error_reward}, due to processed_ground_truths is empty"
            )
            return self.config.unk_error_reward, False

        # Check against all possible correct answers
        for gt in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, gt) or grade_answer_sympy(
                model_answer, gt
            )
            if is_correct:
                logger.debug(f"Reward = {self.config.correct_reward}, due to correct answer")
                return self.config.correct_reward, True

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            for gt in processed_ground_truths:
                try:
                    orm_response = call_oai_rm_llm(
                        prompt=prompt,
                        model_output=model_answer,
                        ground_truth=gt,
                        temperature=0.0,
                    )

                    if orm_response and "[[YES]]" in orm_response:
                        return self.config.correct_reward, True
                    else:
                        return self.config.format_reward, False
                except Exception as e:
                    print("Error calling LLM ORM:", e)
                    continue

        logger.debug(f"Reward = {self.config.format_reward}, due to no correct answer")
        return self.config.format_reward, False


class MathJudgeV5Fn(RewardFn):

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        urls_env = os.environ.get("LLM_JUDGE_URL")
        if urls_env is None:
            raise ValueError("LLM_JUDGE_URL environment variable is not set")
        urls = urls_env.split("\n")
        self.clients = [openai.OpenAI(base_url=url, api_key="_") for url in urls]
        self.client = random.choice(self.clients)

        model_name = os.environ.get("LLM_JUDGE_MODEL")
        if model_name is None:
            raise ValueError("LLM_JUDGE_MODEL environment variable is not set")
        self.model_name: str = model_name

    def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):

        if THINKING_START in prompt + model_response and THINKING_END in prompt + model_response:
            model_solution = model_response.split(THINKING_END)[1]
        else:
            return self.config.format_error_reward, False

        if raw_prompt is None:
            return self.config.format_error_reward, False

        question = raw_prompt[-1]["content"]
        answer = model_solution
        # print('【question】', question)
        # print('【answer】', answer)
        prompt_j = scale_verify_promptv5.replace("【problem】", question)
        # Convert ground_truth to string for prompt
        gt_str = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
        prompt_j = prompt_j.replace("【ground_truth】", gt_str)
        prompt_j = prompt_j.replace("【answer】", answer)
        for i in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_j,
                        }
                    ],
                )
                content = response.choices[0].message.content
                print("【llm judge response】", content)
                if "```json" in content:
                    content = content.split("```json")[1].strip()
                elif content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                result = json.loads(content)
            except Exception as e:
                print(f"[Error in ScaleVerifyV5: {e}, retrying {i+1}/5]" f"\n[Prompt: {prompt_j}]")
                continue
            final_student_answer = result.get("final_student_answer")
            if (
                "Incomplete" in final_student_answer
                or "None" in final_student_answer
                or final_student_answer == ""
                or final_student_answer is None
            ):
                return self.config.format_reward, False
            if result.get("judge"):
                return self.config.llm_correct_reward, True

            return self.config.format_reward, False


class HybridMathJudgeFnV5(RewardFn):
    def __call__(  # pylint: disable=redefined-outer-name
        self,
        prompt: str,
        model_response: str,
        ground_truth: str | list[str],
        raw_prompt: list[dict[str, Any]] | None = None,
    ):
        # print('prompt', prompt)
        # print('model_response', model_response)

        rule_fn = RewardMathFn(self.config)
        reward_score, is_correct = rule_fn(
            prompt=prompt,
            model_response=model_response,
            ground_truth=ground_truth,
            raw_prompt=raw_prompt,
        )
        if reward_score == 1:
            return reward_score, is_correct
        llm_fn = MathJudgeV5Fn(self.config)
        reward_score, is_correct = llm_fn(
            prompt=prompt,
            model_response=model_response,
            ground_truth=ground_truth,
            raw_prompt=raw_prompt,
        )
        return reward_score, is_correct


def compute_score(  # pylint: disable=redefined-outer-name
    prompt_str: str,
    solution_str: str,
    ground_truth: str | list[str],
    stage: str,
    raw_prompt: list[dict[str, Any]] | None = None,
    judge_mode: str | None = None,
):
    reward_config = RewardConfig()
    if stage == "val":
        reward_config.use_math_orm = False

    reward_fn: RewardFn
    if judge_mode is None or judge_mode == "rule":
        reward_fn = RewardMathFn(reward_config)
        raw_prompt = None
    elif judge_mode == "llm_judge":
        reward_fn = MathJudgeV5Fn(reward_config)
    elif judge_mode == "rule+llm_judge":
        reward_fn = HybridMathJudgeFnV5(reward_config)
    else:
        raise ValueError(f"Invalid judge mode: {judge_mode}")

    try:
        reward_score, is_correct = reward_fn(
            prompt=prompt_str,
            model_response=solution_str,
            ground_truth=ground_truth,
            raw_prompt=raw_prompt,
        )
        logger.debug(f"Reward: {reward_score}, is_correct: {is_correct}")
        return reward_score
    except Exception as e:
        print("Error computing score:", e)
        return reward_config.unk_error_reward


if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig())

    problem = "Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$."
    model_response = "<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}"
    ground_truth = ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24"]
    output = reward(prompt=problem, model_response=model_response, ground_truth=ground_truth)
    print(output)
