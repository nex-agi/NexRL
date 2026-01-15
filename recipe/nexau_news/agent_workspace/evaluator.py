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

"""News Evaluator for NexRL framework."""

import re
from typing import Any

import pandas as pd

from nexrl.rollout_worker import BaseEvaluationTarget, EvaluationRunResult, Evaluator


class NewsEvaluator(Evaluator):
    def extract_answer(self, response: str) -> str | None:
        answer_match = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer_match:
            return answer_match[-1].strip()
        else:
            return None

    def extract_reasoning(self, response: str) -> str | None:
        reasoning_match = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)
        if reasoning_match:
            return reasoning_match[-1].strip()
        else:
            return None

    def parse_answer_to_bool(self, answer_str):
        """Convert answer string to boolean value."""
        if answer_str is None:
            return None
        answer_lower = answer_str.lower().strip()
        # Support multiple formats: True, true, False, false, 是, 否, yes, no, etc.
        if answer_lower in ["true", "是", "yes", "1", "t"]:
            return True
        elif answer_lower in ["false", "否", "no", "0", "f"]:
            return False
        else:
            return None

    def evaluate(self, data: Any, evaluation_target: BaseEvaluationTarget) -> EvaluationRunResult:
        extracted_answer_str = self.extract_answer(evaluation_target.final_answer)
        extracted_answer_bool = self.parse_answer_to_bool(extracted_answer_str)
        _ = self.extract_reasoning(evaluation_target.final_answer)
        # Get true label
        # Use pd.isna() to check NaN values (pandas NaN cannot be checked with == or is None)
        # Safely access "标注" (annotation) field, if not present use "通过" (pass) field
        annotation = data.get("标注")
        if annotation is None or pd.isna(annotation):
            true_label = bool(data.get("通过", False))
            if extracted_answer_bool == true_label:
                accuracy_flex = 1
            else:
                accuracy_flex = 0
        else:
            true_label = annotation in ("必须推", "都可以")
            if extracted_answer_bool == true_label or annotation == "都可以":
                accuracy_flex = 1
            else:
                accuracy_flex = 0

        # Initialize metrics
        tp = 0  # True Positive: predicted True, actual True
        fp = 0  # False Positive: predicted True, actual False
        fn = 0  # False Negative: predicted False, actual True
        accuracy = 0

        # Check if answer was successfully extracted
        answer_extracted = extracted_answer_bool is not None
        # If answer was successfully extracted, calculate metrics
        if answer_extracted:
            if extracted_answer_bool == true_label:
                accuracy = 1
                if true_label:
                    tp = 1
            else:
                # Prediction error
                if extracted_answer_bool and not true_label:
                    fp = 1  # Predicted True, actual False
                elif not extracted_answer_bool and true_label:
                    fn = 1  # Predicted False, actual True
        else:
            # If unable to extract answer, treat as prediction failure
            # In news filtering scenario, inability to extract answer usually means should not push (conservative strategy)
            if true_label:
                fn = 1  # Should push but didn't (false negative)
            else:
                # Should not push and cannot predict, can be treated as correct conservative decision (True Negative)
                accuracy = 1  # Although unable to extract answer, result is correct (not push)

        # Calculate precision, recall, and f1
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Create metrics dictionary
        metrics = {
            "accuracy_strict": accuracy,
            "accuracy_flex": accuracy_flex,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        return EvaluationRunResult(
            reward=accuracy_flex,
            ground_truth=str(true_label),
            metrics=metrics,
            extra_info={},
        )
