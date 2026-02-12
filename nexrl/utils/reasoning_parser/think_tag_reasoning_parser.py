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

"""Parser for reasoning content wrapped in <think></think> tags."""

import logging
import re

from .base_reasoning_parser import BaseReasoningParser
from .core_types import ReasoningParseResult

logger = logging.getLogger(__name__)


class ThinkTagReasoningParser(BaseReasoningParser):
    """
    Parser for reasoning content wrapped in <think></think> tags.

    Format:
        <think>reasoning content here</think>actual response

    Note: The <think> tag is typically added automatically at the end of
    the user message when thinking is enabled, so we primarily detect
    the closing </think> tag.

    Example:
        Input: "<think>Let me think about this...</think>The answer is 42."
        Output:
            reasoning_content: "Let me think about this..."
            cleaned_content: "The answer is 42."
    """

    def __init__(self):
        """Initialize the parser with tag patterns."""
        self._start_tag = "<think>"
        self._end_tag = "</think>"
        # Pattern to extract everything between optional <think> and </think>
        # Group 1: optional <think> tag
        # Group 2: reasoning content
        self._reasoning_pattern = re.compile(r"^(<think>)?(.*?)</think>", re.DOTALL)

    def detect_reasoning(self, text: str) -> bool:
        """
        Check if text contains </think> closing tag.

        Args:
            text: The text to check

        Returns:
            bool: True if </think> tag is present
        """
        return self._end_tag in text

    def extract_reasoning_string(self, text: str) -> str | None:
        """
        Extract the portion of text containing reasoning.

        Args:
            text: The full response text

        Returns:
            str | None: The full text if reasoning detected, None otherwise
        """
        if self.detect_reasoning(text):
            return text  # Return full text for parsing
        return None

    def parse_reasoning_string(self, text: str) -> ReasoningParseResult:
        """
        Parse text to extract reasoning and remove it from content.

        The reasoning content is between optional <think> and </think>.
        The cleaned content is everything after </think>.

        Args:
            text: The text containing reasoning to parse

        Returns:
            ReasoningParseResult: The parsed reasoning and cleaned content
        """
        match = self._reasoning_pattern.search(text)

        if not match:
            logger.warning("</think> detected but pattern match failed")
            return ReasoningParseResult(
                reasoning_content=None, cleaned_content=text, is_valid=False
            )

        # Extract reasoning content (group 2 - between tags, excluding the tags)
        reasoning_content = match.group(2).strip()

        # Extract cleaned content (everything after </think>)
        end_pos = match.end()
        cleaned_content = text[end_pos:].strip()

        return ReasoningParseResult(
            reasoning_content=reasoning_content if reasoning_content else None,
            cleaned_content=cleaned_content,
            is_valid=True,
        )
