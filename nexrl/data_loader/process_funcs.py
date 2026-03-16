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
Built-in process functions for StreamingDataset tokenizer wrappers.

These functions convert raw data formats (e.g. ``messages`` list) into the
``prompt``/``output`` pairs expected by ``sft_multi_round``.

Each function is automatically registered in :data:`BUILTIN_PROCESS_FUNCS`
and can be referenced by name in YAML configs::

    subset_params:
      whole:
        process_func: "qwen3_chat"
"""

from __future__ import annotations

from typing import Any


def qwen3_chat(data: dict[str, Any]) -> list[dict[str, str]]:
    """Convert ``messages`` format to prompt/output pairs with Qwen3 chat template.

    Matches the ``process_func`` used in Megatron-BPLM CI
    (``examples/models/qwen3/common_setting.py``).

    If no system message is present, a default reasoning prompt is prepended.
    """
    res: list[dict[str, str]] = []

    has_system = any(c["role"] == "system" for c in data["messages"])
    if not has_system:
        res.append(
            {
                "prompt": (
                    "<|im_start|>system\n"
                    "Please reason step by step, and put your final answer within \\boxed{}."
                    "<|im_end|>\n"
                ),
                "output": "",
            }
        )

    for conversation in data["messages"]:
        role = conversation["role"]
        content = conversation["content"]
        if role == "system":
            res.append(
                {
                    "prompt": f"<|im_start|>system\n{content}<|im_end|>\n",
                    "output": "",
                }
            )
        elif role == "user":
            res.append(
                {
                    "prompt": f"<|im_start|>user\n{content}<|im_end|>\n",
                    "output": "",
                }
            )
        elif role == "assistant":
            res.append(
                {
                    "prompt": "",
                    "output": f"<|im_start|>assistant\n{content}<|im_end|><|endoftext|>",
                }
            )

    return res


def messages_to_chatml(data: dict[str, Any]) -> list[dict[str, str]]:
    """Convert ``messages`` format to prompt/output pairs using ChatML tags.

    Unlike :func:`qwen3_chat`, no default system prompt is injected.
    """
    res: list[dict[str, str]] = []
    for conversation in data["messages"]:
        role = conversation["role"]
        content = conversation["content"]
        if role in ("system", "user"):
            res.append(
                {
                    "prompt": f"<|im_start|>{role}\n{content}<|im_end|>\n",
                    "output": "",
                }
            )
        elif role == "assistant":
            res.append(
                {
                    "prompt": "",
                    "output": f"<|im_start|>assistant\n{content}<|im_end|><|endoftext|>",
                }
            )
    return res


# Registry of built-in process functions, keyed by name.
BUILTIN_PROCESS_FUNCS: dict[str, Any] = {
    "qwen3_chat": qwen3_chat,
    "messages_to_chatml": messages_to_chatml,
}
