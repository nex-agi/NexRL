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
Base Tool Parser - Abstract base class for tool call parsing
"""

import logging
from abc import ABC, abstractmethod

from .core_types import ParseResult

logger = logging.getLogger(__name__)


class BaseToolParser(ABC):
    """
    Abstract base class for tool call parsers.

    Different models may use different formats for tool calls (e.g., XML tags,
    JSON blocks, custom formats). This base class defines the interface that
    all tool parsers must implement.

    Subclasses should implement:
    - detect_tool_call: Check if text contains tool calls in expected format
    - extract_tool_string: Extract the raw tool call string from response
    - parse_tool_string: Parse the extracted tool string into structured format
    """

    def __init__(self):
        """Initialize the tool parser."""

    @abstractmethod
    def detect_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains tool calls in this parser's format.

        Args:
            text: The text to check

        Returns:
            bool: True if tool calls are detected, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def extract_tool_string(self, text: str) -> str | None:
        """
        Extract the raw tool call string from the response text.

        Args:
            text: The full response text

        Returns:
            str | None: The extracted tool call string, or None if not found
        """
        raise NotImplementedError()

    @abstractmethod
    def parse_tool_string(self, tool_string: str) -> ParseResult:
        """
        Parse the extracted tool string into structured tool calls.

        Args:
            tool_string: The raw tool call string to parse

        Returns:
            ParseResult: The parsed tool calls and validation status
        """
        raise NotImplementedError()

    def parse(self, text: str) -> ParseResult:
        """
        Convenience method that combines detection, extraction, and parsing.

        Args:
            text: The full response text

        Returns:
            ParseResult: The parsed tool calls and validation status
        """
        if not self.detect_tool_call(text):
            return ParseResult(tool_calls=None, is_valid=True)

        tool_string = self.extract_tool_string(text)
        if tool_string is None:
            return ParseResult(tool_calls=None, is_valid=False)

        return self.parse_tool_string(tool_string)
