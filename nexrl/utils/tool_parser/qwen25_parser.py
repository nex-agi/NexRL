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
Qwen25 Parser - Parses tool calls in Qwen 2.5/3.0 format
"""

import json
import logging
import re
import uuid

from .base_tool_parser import BaseToolParser
from .core_types import ParseResult, ToolCallItem

logger = logging.getLogger(__name__)


class Qwen25Parser(BaseToolParser):
    """
    Parser for Qwen 2.5 and Qwen 3 model function call format.

    Format Structure:
    ```
    <tool_call>
    {"name":"func1", "arguments":{...}}
    </tool_call>
    <tool_call>
    {"name":"func2", "arguments":{...}}
    </tool_call>
    ```

    Key Components:
    - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
    - Function Call Object: JSON object with "name" and "arguments" fields

    Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    """

    def __init__(self):
        """
        Initialize the Qwen parser with necessary state variables.
        """
        super().__init__()
        self._tag_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def detect_tool_call(self, text: str) -> bool:
        """
        Check if text contains <tool_call> tags.

        Args:
            text: The text to check

        Returns:
            bool: True if <tool_call> tags are present
        """
        return "<tool_call>" in text

    def extract_tool_string(self, text: str) -> str | None:
        """
        Extract all tool call blocks from the response.

        Args:
            text: The full response text

        Returns:
            str | None: The full text containing all tool calls, or None if not found
        """
        if self.detect_tool_call(text):
            return text
        return None

    def parse_tool_string(self, tool_string: str) -> ParseResult:
        """
        Parse all tool call blocks into structured tool calls.

        Args:
            tool_string: The text containing tool calls

        Returns:
            ParseResult: The parsed tool calls and validation status
        """
        matches = self._tag_pattern.findall(tool_string)

        if not matches:
            logger.warning("No tool call blocks found")
            return ParseResult(tool_calls=None, is_valid=False)

        tool_calls = []
        for idx, match in enumerate(matches):
            try:
                tool_call = json.loads(match.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")
                continue

            if not isinstance(tool_call, dict):
                logger.warning(f"Tool call is not a dict: {type(tool_call)}")
                continue

            # Extract name
            name = tool_call.get("name")
            if not isinstance(name, str):
                logger.warning(f"Tool call name is missing or not a string: {name}")
                continue

            # Extract arguments (support both "arguments" and "parameters")
            args = tool_call.get("arguments") or tool_call.get("parameters")

            # Normalize arguments to JSON string
            if args is None:
                arguments_str = "{}"
            elif isinstance(args, str):
                arguments_str = args
            elif isinstance(args, dict):
                try:
                    arguments_str = json.dumps(args)
                except Exception as e:
                    logger.warning(f"Failed to serialize tool call args: {e}")
                    arguments_str = str(args)
            else:
                arguments_str = str(args)

            # Extract or generate tool ID
            tool_id = tool_call.get("id")
            if tool_id is not None and not isinstance(tool_id, str):
                tool_id = None
            if tool_id is None:
                tool_id = f"call-{uuid.uuid4().hex}-{idx}"

            # Create normalized tool call item
            tool_call_item = ToolCallItem(
                id=tool_id, type="function", function={"name": name, "arguments": arguments_str}
            )
            tool_calls.append(tool_call_item)

        if not tool_calls:
            return ParseResult(tool_calls=None, is_valid=False)

        logger.debug(f"Parsed {len(tool_calls)} tool calls")
        return ParseResult(tool_calls=tool_calls, is_valid=True)
