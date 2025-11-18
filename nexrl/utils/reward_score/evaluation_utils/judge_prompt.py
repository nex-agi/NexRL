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

scale_verify_promptv5 = """You are an accurate and reliable automated grading system. Your task is to check if a student's math exam solution reaches a correct final answer.

# Task Steps
1. **Extract the student's final answer**: The student's final answer can take various forms, such as a number, string, sequence, etc. If the student's solution claims no answer, use "None"; if it's incomplete, use "Incomplete solution".
2. **Compare the answers**: Directly compare the extracted student's final answer with the reference solution to determine if they match.

# Key Notes
- **Focus on the end result**: Only pay attention to the student's final answer, not the steps they took to get there.
- **Ignore formatting**: Disregard any formatting differences in the student's final answer. For example, "3.5", "\\boxed{7 / 2}", and "$\\frac{14}{4}$" are considered the same.
- **Omit units and specific formats**: Don't consider units omission or specific formatting requirements in the student's final answer.
- **Inequality equivalence**: Inequalities in the student's final answer are equivalent if they represent the same range. However, boundary conditions are crucial. For example, "x < 2" is not the same as "x ∈ (-∞, 2]".
- **Handle incomplete solutions**: Treat a truncated or incoherent student's solution as incorrect.

# Input Information
Math question:
【problem】

Reference solution:
【ground_truth】

Student's solution:
【answer】

# Response Format (JSON)
```
{
    "judge": true/false,
    "final_student_answer": "The student's final answer without formatting",
    "judge_reason": "Brief explanation for the judgment"
}
```"""
