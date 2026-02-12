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
DeepSeekV31 Parser - Parses tool calls in DeepSeek V3.1 format
"""

import json
import logging
import re
import uuid

from .base_tool_parser import BaseToolParser
from .core_types import ParseResult, ToolCallItem

logger = logging.getLogger(__name__)


class DeepseekV31Parser(BaseToolParser):
    """
    Parser for DeepSeek V3.1 model function call format.

    The DeepSeek V3.1 format uses special Unicode tokens to delimit function calls
    with JSON code blocks for arguments.

    Format Structure:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>{json_arguments}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    ```

    Examples:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    ```

    Multiple calls:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜tool▁calls▁begin｜>` and `<｜tool▁calls▁end｜>`
    - Individual Tool Call: Wrapped between `<｜tool▁call▁begin｜>` and `<｜tool▁call▁end｜>`
    - Function Declaration: `<｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>`
    - Arguments: JSON code block between `<｜tool▁sep｜>` and `<｜tool▁call▁end｜>`

    Reference: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.1
    """

    def __init__(self):
        """
        Initialize the DeepSeek V3.1 parser with necessary regex patterns.
        """
        super().__init__()
        self._begin_token = "<｜tool▁calls▁begin｜>"
        self._end_token = "<｜tool▁calls▁end｜>"
        self._call_pattern = re.compile(
            r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>", re.DOTALL
        )

    def detect_tool_call(self, text: str) -> bool:
        """
        Check if text contains DeepSeek V3.1 tool call markers.

        Args:
            text: The text to check

        Returns:
            bool: True if tool call markers are present
        """
        return self._begin_token in text

    def extract_tool_string(self, text: str) -> str | None:
        """
        Extract the tool calls section from the response.

        Args:
            text: The full response text

        Returns:
            str | None: The extracted tool calls section, or None if not found
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
        # Find all individual tool calls
        call_matches = self._call_pattern.findall(tool_string)

        if not call_matches:
            logger.warning("No tool call blocks found")
            return ParseResult(tool_calls=None, is_valid=False)

        tool_calls = []
        for idx, (func_name, func_args_str) in enumerate(call_matches):
            func_name = func_name.strip()
            func_args_str = func_args_str.strip()

            if not func_name:
                logger.warning("Empty function name")
                continue

            # Parse JSON arguments
            try:
                func_args = json.loads(func_args_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON arguments: {e}")
                continue

            # Convert to JSON string for OpenAI format
            try:
                arguments_str = json.dumps(func_args)
            except Exception as e:
                logger.warning(f"Failed to serialize arguments: {e}")
                arguments_str = str(func_args)

            # Generate tool call ID
            tool_id = f"call-{uuid.uuid4().hex}-{idx}"

            # Create normalized tool call item
            tool_call_item = ToolCallItem(
                id=tool_id,
                type="function",
                function={"name": func_name, "arguments": arguments_str},
            )
            tool_calls.append(tool_call_item)

        if not tool_calls:
            return ParseResult(tool_calls=None, is_valid=False)

        logger.debug(f"Parsed {len(tool_calls)} tool calls")
        return ParseResult(tool_calls=tool_calls, is_valid=True)
