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
Qwen3 Coder Parser - Parses tool calls in Qwen 3 Coder format
"""

import ast
import html
import json
import logging
import re
import uuid
from typing import Any

from .base_tool_parser import BaseToolParser
from .core_types import ParseResult, ToolCallItem

logger = logging.getLogger(__name__)


def _safe_val(raw: str) -> Any:
    """
    Safely parse a value string, trying JSON, then Python literal eval, then raw string.

    Args:
        raw: The raw string value to parse

    Returns:
        Parsed value (dict, list, str, int, float, etc.)
    """
    raw = html.unescape(raw.strip())
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw


class Qwen3CoderParser(BaseToolParser):
    """
    Parser for Qwen 3 Coder model function call format.

    Format Structure:
    ```
    <tool_call>
    <function=execute_bash>
    <parameter=command>
    pwd && ls
    </parameter>
    </function>
    </tool_call>
    ```

    Multiple parameters:
    ```
    <tool_call>
    <function=create_file>
    <parameter=filename>
    test.py
    </parameter>
    <parameter=content>
    print("Hello World")
    </parameter>
    </function>
    </tool_call>
    ```

    Key Components:
    - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each call
    - Function Declaration: `<function=name>` ... `</function>`
    - Parameters: `<parameter=name>value</parameter>` (can have multiple)

    Reference: Qwen3 Coder model format
    """

    def __init__(self):
        """
        Initialize the Qwen3 Coder parser with necessary regex patterns.
        """
        super().__init__()
        self._tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self._function_pattern = re.compile(r"<function=(.*?)</function>", re.DOTALL)
        self._parameter_pattern = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

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
        tool_call_matches = self._tool_call_pattern.findall(tool_string)

        if not tool_call_matches:
            logger.warning("No tool call blocks found")
            return ParseResult(tool_calls=None, is_valid=False)

        tool_calls = []
        for idx, tool_call_content in enumerate(tool_call_matches):
            parsed_call = self._parse_tool_call_block(tool_call_content, idx)
            if parsed_call:
                tool_calls.append(parsed_call)

        if not tool_calls:
            return ParseResult(tool_calls=None, is_valid=False)

        logger.debug(f"Parsed {len(tool_calls)} tool calls")
        return ParseResult(tool_calls=tool_calls, is_valid=True)

    def _parse_tool_call_block(self, block: str, idx: int) -> ToolCallItem | None:
        """
        Parse a single tool call block.

        Args:
            block: The content within <tool_call>...</tool_call>
            idx: Index for generating unique tool call ID

        Returns:
            ToolCallItem | None: Parsed tool call or None if parsing failed
        """
        # Extract function block
        function_matches = self._function_pattern.findall(block)
        if not function_matches:
            logger.warning("No function block found in tool call")
            return None

        function_content = function_matches[0]

        # Extract function name (before first >)
        if ">" not in function_content:
            logger.warning("Invalid function format: missing >")
            return None

        idx_sep = function_content.index(">")
        function_name = function_content[:idx_sep].strip()
        function_body = function_content[idx_sep + 1 :]

        if not function_name:
            logger.warning("Empty function name")
            return None

        # Extract parameters
        parameters = {}
        param_matches = self._parameter_pattern.findall(function_body)
        for param_name, param_value in param_matches:
            param_name = param_name.strip()
            # Strip leading/trailing newlines but preserve content
            param_value = param_value.lstrip("\n").rstrip("\n")
            parameters[param_name] = _safe_val(param_value)

        # Convert to JSON string for OpenAI format
        try:
            arguments_str = json.dumps(parameters)
        except Exception as e:
            logger.warning(f"Failed to serialize parameters: {e}")
            arguments_str = str(parameters)

        # Generate tool call ID
        tool_id = f"call-{uuid.uuid4().hex}-{idx}"

        # Create normalized tool call item
        tool_call_item = ToolCallItem(
            id=tool_id,
            type="function",
            function={"name": function_name, "arguments": arguments_str},
        )

        logger.debug(f"Parsed tool call: {function_name} with parameters: {parameters}")
        return tool_call_item
