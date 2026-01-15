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

import re
import string
from typing import Any

from nexrl.rollout_worker import EvaluationRunResult, Evaluator, NexAUEvaluationTarget


def extract_answer(response: str) -> str:
    answer_match = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        return answer_match[-1].strip()
    else:
        return None


def f1_preprocess_text(text: str) -> str:
    """预处理文本，用于数据集的评分

    处理步骤:
    1. 转换为小写
    2. 移除标点符号 (.,!?;:'"()[]{}...)
    3. 去除多余空格
    """
    # 将标点符号替换为空格
    text = text.lower()
    for punct in string.punctuation:
        text = text.replace(punct, " ")

    # 替换多个空格为单个空格
    text = re.sub(r"\s+", " ", text)

    # 去除首尾空格
    text = text.strip()
    return text


def calc_score(answer_content: str, ground_truth: str) -> float:
    """Calculate F1 score between the answer and ground truth."""
    answer_content = f1_preprocess_text(answer_content)

    # Handle multiple ground truths separated by <|answer_split|>
    ground_truths = [f1_preprocess_text(ground_truth)]
    if isinstance(ground_truth, str) and "<|answer_split|>" in ground_truth:
        ground_truths = [f1_preprocess_text(_) for _ in ground_truth.split("<|answer_split|>")]

    max_score = 0.0

    for gt_option in ground_truths:
        # Preprocess ground truth

        # Tokenize answer and ground truth
        pred_tokens = set(answer_content.split())
        gt_tokens = set(gt_option.split())

        if not gt_tokens:  # Avoid division by zero
            continue
        if not pred_tokens:
            continue

        # Calculate common tokens
        common_tokens = pred_tokens & gt_tokens

        # Calculate precision and recall
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
        recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0

        # Calculate F1 score
        if precision + recall > 0:  # Avoid division by zero
            f1 = 2 * (precision * recall) / (precision + recall)
            max_score = max(max_score, f1)

    return max_score


def reward_function(response: str, ground_truth: str) -> float:
    extracted_answer = extract_answer(response)
    if extracted_answer is None:
        print(f"No answer found in response: {response}")
        return 0.0
    return calc_score(extracted_answer, ground_truth)


class DeepResearchEvaluator(Evaluator):
    def evaluate(self, data: Any, evaluation_target: NexAUEvaluationTarget) -> EvaluationRunResult:
        reward = reward_function(evaluation_target.final_answer, data["ground_truth"])
        ground_truth = data["ground_truth"]
        return EvaluationRunResult(
            reward=reward, ground_truth=ground_truth, metrics={}, extra_info={}
        )
