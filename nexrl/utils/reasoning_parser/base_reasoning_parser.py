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

"""Base class for reasoning content parsers."""

from abc import ABC, abstractmethod

from .core_types import ReasoningParseResult


class BaseReasoningParser(ABC):
    """
    Abstract base class for reasoning content parsers.

    Different models may use different formats for reasoning/thinking output.
    This base class defines the interface that all reasoning parsers must implement.

    Subclasses should implement:
    - detect_reasoning: Check if text contains reasoning markers
    - extract_reasoning_string: Extract the raw reasoning string from response
    - parse_reasoning_string: Parse the extracted string into structured format
    """

    @abstractmethod
    def detect_reasoning(self, text: str) -> bool:
        """
        Check if text contains reasoning markers in this parser's format.

        Args:
            text: The text to check

        Returns:
            bool: True if reasoning markers are detected
        """
        raise NotImplementedError()

    @abstractmethod
    def extract_reasoning_string(self, text: str) -> str | None:
        """
        Extract the raw reasoning string from the response text.

        Args:
            text: The full response text

        Returns:
            str | None: The extracted reasoning string, or None if not found
        """
        raise NotImplementedError()

    @abstractmethod
    def parse_reasoning_string(self, text: str) -> ReasoningParseResult:
        """
        Parse text to extract reasoning content and clean main content.

        Args:
            text: The text containing reasoning to parse

        Returns:
            ReasoningParseResult: The parsed reasoning and cleaned content
        """
        raise NotImplementedError()

    def parse(self, text: str) -> ReasoningParseResult:
        """
        Convenience method that combines detection, extraction, and parsing.

        Args:
            text: The full response text

        Returns:
            ReasoningParseResult: The parsed reasoning and cleaned content
        """
        if not self.detect_reasoning(text):
            return ReasoningParseResult(reasoning_content=None, cleaned_content=text, is_valid=True)

        reasoning_string = self.extract_reasoning_string(text)
        if reasoning_string is None:
            return ReasoningParseResult(
                reasoning_content=None, cleaned_content=text, is_valid=False
            )

        return self.parse_reasoning_string(text)


class BaseReasoningFormatDetector:
    """
    Base class providing two sets of interfaces: one-time and streaming incremental.

    This class supports both non-streaming (one-time) parsing and streaming incremental
    parsing for reasoning content extraction.
    """

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning

        self._buffer = ""
        self.stripped_think_start = False

        self.continue_final_message = continue_final_message
        if self.continue_final_message:
            self.previous_content = previous_content
            self.previous_count = len(previous_content)
        else:
            self.previous_content = ""
            self.previous_count = 0

        if self.think_start_token in self.previous_content:
            self._in_reasoning = True
        if self.think_end_token in self.previous_content:
            self._in_reasoning = False

    def detect_and_parse(self, text: str) -> ReasoningParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return ReasoningParseResult(reasoning_content=None, cleaned_content=text, is_valid=True)

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(self.think_start_token, "").strip()

        if (
            self.think_end_token not in processed_text
            and self.think_end_token not in self.previous_content
        ):
            # Assume reasoning was truncated before end token
            return ReasoningParseResult(
                reasoning_content=processed_text, cleaned_content="", is_valid=True
            )

        # Extract reasoning content
        if self.think_end_token in processed_text:
            splits = processed_text.split(self.think_end_token, maxsplit=1)
            reasoning_text = splits[0]
            normal_text = splits[1].strip()

            return ReasoningParseResult(
                reasoning_content=reasoning_text, cleaned_content=normal_text, is_valid=True
            )
        else:
            # think_end_token is in self.previous_content for continue_final_message=True case
            return ReasoningParseResult(
                reasoning_content=None, cleaned_content=processed_text, is_valid=True
            )

    def parse_streaming_increment(self, new_text: str) -> ReasoningParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current_text = self._buffer

        # If the current text is a prefix of the think token, keep buffering
        if any(
            token.startswith(current_text) and token != current_text
            for token in [self.think_start_token, self.think_end_token]
        ):
            return ReasoningParseResult(reasoning_content=None, cleaned_content="", is_valid=True)

        # Strip start token if present
        if not self.stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return ReasoningParseResult(
                reasoning_content=reasoning_text.rstrip(),
                cleaned_content=normal_text,
                is_valid=True,
            )

        # Continue with reasoning content
        if self._in_reasoning:
            if self.stream_reasoning:
                # Stream the content immediately
                self._buffer = ""
                return ReasoningParseResult(
                    reasoning_content=current_text, cleaned_content="", is_valid=True
                )
            else:
                return ReasoningParseResult(
                    reasoning_content=None, cleaned_content="", is_valid=True
                )

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            return ReasoningParseResult(
                reasoning_content=None, cleaned_content=current_text, is_valid=True
            )

        return ReasoningParseResult(reasoning_content=None, cleaned_content="", is_valid=True)
