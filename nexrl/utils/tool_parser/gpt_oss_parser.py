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
GPT OSS Parser - Parses tool calls in GPT OSS/Harmony format
"""

import json
import logging
import re
import uuid

from .base_tool_parser import BaseToolParser
from .core_types import ParseResult, ToolCallItem

logger = logging.getLogger(__name__)


class GptOssParser(BaseToolParser):
    """
    Parser for GPT OSS (T4-style) function call format.

    This format uses harmony-style tags with commentary channels for tool calls.

    Format Structure:
    ```
    <|channel|>commentary to={namespace.function}<|constrain|>json<|message|>{args}<|call|>
    ```

    Examples:
    ```
    <|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location": "Tokyo"}<|call|>
    ```

    With namespace:
    ```
    <|channel|>commentary to=tools.calculate<|constrain|>json<|message|>{"operation": "add", "a": 5, "b": 3}<|call|>
    ```

    Key Components:
    - Channel Declaration: `<|channel|>commentary to={function_name}`
    - Constraint: `<|constrain|>json` (indicates JSON arguments)
    - Arguments: `<|message|>{json_args}`
    - End Token: `<|call|>`

    Note: This is a simplified version. Full streaming support requires HarmonyParser.
    """

    def __init__(self):
        """
        Initialize the GPT OSS parser with necessary regex patterns.
        """
        super().__init__()
        self._bot_token = "<|channel|>commentary"
        self._tool_pattern = re.compile(
            r"<\|channel\|>commentary\s+to=([a-zA-Z_][a-zA-Z0-9_.-]*)\s*<\|constrain\|>json<\|message\|>(.*?)(?:<\|call\|>|$)",
            re.DOTALL,
        )

    def detect_tool_call(self, text: str) -> bool:
        """
        Check if text contains GPT OSS tool call markers.

        Args:
            text: The text to check

        Returns:
            bool: True if tool call markers are present
        """
        return self._bot_token in text

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
        # Find all tool call matches
        matches = self._tool_pattern.findall(tool_string)

        if not matches:
            logger.warning("No tool call blocks found")
            return ParseResult(tool_calls=None, is_valid=False)

        tool_calls = []
        for idx, (full_function_name, json_content) in enumerate(matches):
            # Extract function name (last part after . if namespaced)
            function_name = (
                full_function_name.split(".")[-1]
                if "." in full_function_name
                else full_function_name
            )

            if not function_name:
                logger.warning("Empty function name")
                continue

            # Parse JSON arguments
            try:
                arguments = json.loads(json_content.strip()) if json_content.strip() else {}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON arguments: {e}")
                continue

            # Convert to JSON string for OpenAI format
            try:
                arguments_str = json.dumps(arguments, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to serialize arguments: {e}")
                arguments_str = str(arguments)

            # Generate tool call ID
            tool_id = f"call-{uuid.uuid4().hex}-{idx}"

            # Create normalized tool call item
            tool_call_item = ToolCallItem(
                id=tool_id,
                type="function",
                function={"name": function_name, "arguments": arguments_str},
            )
            tool_calls.append(tool_call_item)

        if not tool_calls:
            return ParseResult(tool_calls=None, is_valid=False)

        logger.debug(f"Parsed {len(tool_calls)} tool calls")
        return ParseResult(tool_calls=tool_calls, is_valid=True)
