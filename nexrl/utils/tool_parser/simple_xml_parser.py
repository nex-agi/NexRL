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
Simple XML Parser - Parses tool calls in <tool_call>...</tool_call> format
"""

import json
import logging
import re
import uuid

from .base_tool_parser import BaseToolParser
from .core_types import ParseResult, ToolCallItem

logger = logging.getLogger(__name__)


class SimpleXmlParser(BaseToolParser):
    """
    Parser for tool calls wrapped in <tool_call>...</tool_call> XML tags.

    Expected format:
        <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>

    The JSON content should have:
    - "name": The function name (required)
    - "arguments" or "parameters": The function arguments as a dict
    - "id": Optional tool call ID
    """

    def __init__(self):
        """Initialize the XML tool parser."""
        super().__init__()
        self._tag_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

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
        Extract the content within <tool_call>...</tool_call> tags.

        Args:
            text: The full response text

        Returns:
            str | None: The extracted JSON string, or None if not found
        """
        match = self._tag_pattern.search(text)
        if match:
            return match.group(1).strip()
        return None

    def parse_tool_string(self, tool_string: str) -> ParseResult:
        """
        Parse the JSON string into structured tool calls.

        Args:
            tool_string: The JSON string to parse

        Returns:
            ParseResult: The parsed tool calls and validation status
        """
        try:
            tool_call = json.loads(tool_string)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON: {e}")
            return ParseResult(tool_calls=None, is_valid=False)

        if not isinstance(tool_call, dict):
            logger.warning(f"Tool call is not a dict: {type(tool_call)}")
            return ParseResult(tool_calls=None, is_valid=False)

        # Extract name
        name = tool_call.get("name")
        if not isinstance(name, str):
            logger.warning(f"Tool call name is missing or not a string: {name}")
            return ParseResult(tool_calls=None, is_valid=False)

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
            tool_id = f"call-{uuid.uuid4().hex}"

        # Create normalized tool call item
        tool_call_item = ToolCallItem(
            id=tool_id, type="function", function={"name": name, "arguments": arguments_str}
        )

        logger.debug(f"Parsed tool call: {tool_call_item}")
        return ParseResult(tool_calls=[tool_call_item], is_valid=True)
