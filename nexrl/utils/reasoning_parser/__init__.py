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
Reasoning Parser Framework - Wrapper around SGLang's reasoning parser

This module wraps SGLang's reasoning parser to provide NexRL's interface.
Falls back to a minimal implementation if sglang is not installed.
"""

import logging
from typing import Optional, Tuple

from .core_types import ReasoningParseResult

logger = logging.getLogger(__name__)

__all__ = [
    "ReasoningParseResult",
    "ReasoningParser",
    "create_reasoning_parser",
]

# Try to import from SGLang
try:
    from sglang.srt.parser.reasoning_parser import (
        BaseReasoningFormatDetector,
    )
    from sglang.srt.parser.reasoning_parser import ReasoningParser as SglangReasoningParser

    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    logger.warning(
        "SGLang not available. Install with: pip install 'NexRL[sglang]'. "
        "Falling back to minimal reasoning parser implementation."
    )


class NexRLReasoningParser:
    """
    Wrapper around SGLang's ReasoningParser that converts to NexRL's interface.

    Converts SGLang's StreamingParseResult (reasoning_text, normal_text)
    to NexRL's ReasoningParseResult (reasoning_content, cleaned_content, is_valid).
    """

    def __init__(self, sglang_parser):
        """
        Initialize the wrapper.

        Args:
            sglang_parser: SGLang ReasoningParser instance
        """
        self.sglang_parser = sglang_parser
        self.detector = sglang_parser.detector

    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Non-streaming call: one-time parsing

        Args:
            full_text: The complete text to parse

        Returns:
            Tuple of (reasoning_content, cleaned_content)
        """
        reasoning_text, normal_text = self.sglang_parser.parse_non_stream(full_text)
        return reasoning_text, normal_text

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Streaming call: incremental parsing

        Args:
            chunk_text: The text chunk to parse incrementally

        Returns:
            Tuple of (reasoning_content, cleaned_content)
        """
        reasoning_text, normal_text = self.sglang_parser.parse_stream_chunk(chunk_text)
        return reasoning_text, normal_text

    def parse(self, text: str) -> ReasoningParseResult:
        """
        Convenience method for non-streaming parsing that returns ReasoningParseResult.

        This method maintains backward compatibility with the old parser interface.

        Args:
            text: The text to parse

        Returns:
            ReasoningParseResult with reasoning_content and cleaned_content
        """
        # Use SGLang's detect_and_parse method
        sglang_result = self.detector.detect_and_parse(text)

        # Convert SGLang's StreamingParseResult to NexRL's ReasoningParseResult
        return ReasoningParseResult(
            reasoning_content=sglang_result.reasoning_text,
            cleaned_content=sglang_result.normal_text,
            is_valid=True,  # SGLang doesn't have is_valid, always consider valid
        )


class MinimalReasoningParser:
    """
    Minimal fallback parser when SGLang is not available.
    Only handles basic <think>...</think> tags.
    """

    def __init__(self):
        """Initialize the minimal parser."""
        self.detector = None  # For compatibility

    def parse(self, text: str) -> ReasoningParseResult:
        """
        Parse text to extract reasoning between <think>...</think> tags.

        Args:
            text: The text to parse

        Returns:
            ReasoningParseResult with reasoning_content and cleaned_content
        """
        if "</think>" not in text:
            return ReasoningParseResult(reasoning_content=None, cleaned_content=text, is_valid=True)

        # Simple extraction
        start_tag = "<think>"
        end_tag = "</think>"

        # Find the end tag
        end_idx = text.find(end_tag)
        if end_idx == -1:
            return ReasoningParseResult(reasoning_content=None, cleaned_content=text, is_valid=True)

        # Find start (may or may not be present)
        start_idx = text.find(start_tag)
        if start_idx != -1 and start_idx < end_idx:
            reasoning_content = text[start_idx + len(start_tag) : end_idx].strip()
        else:
            reasoning_content = text[:end_idx].strip()

        cleaned_content = text[end_idx + len(end_tag) :].strip()

        return ReasoningParseResult(
            reasoning_content=reasoning_content if reasoning_content else None,
            cleaned_content=cleaned_content,
            is_valid=True,
        )

    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Non-streaming call."""
        result = self.parse(full_text)
        return result.reasoning_content, result.cleaned_content

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Streaming call - not supported in minimal implementation."""
        return None, chunk_text


# Type alias for compatibility
ReasoningParser = NexRLReasoningParser


def create_reasoning_parser(
    parser_type: str = "think_tag",
    stream_reasoning: bool = False,
    force_reasoning: Optional[bool] = None,
) -> NexRLReasoningParser | MinimalReasoningParser:
    """
    Factory function to create reasoning parsers.

    Args:
        parser_type: Type of detector to create.
                    Supported: "think_tag", "deepseek_r1", "qwen3", "kimi",
                              "minimax_append_think", "step3", "nano_v3", "gpt-oss"
        stream_reasoning: Enable streaming mode
        force_reasoning: Force reasoning mode (if None, uses detector default)

    Returns:
        NexRLReasoningParser or MinimalReasoningParser: An instance of the requested parser

    Raises:
        ValueError: If parser_type is not recognized (when SGLang is available)
    """
    if not SGLANG_AVAILABLE:
        logger.warning(
            f"SGLang not available, using minimal parser. Requested type '{parser_type}' ignored."
        )
        return MinimalReasoningParser()

    # Map NexRL parser types to SGLang model types
    parser_type_lower = parser_type.lower().replace("-", "_").replace("_", "-")

    # Type mapping from NexRL to SGLang
    type_map = {
        "think-tag": "qwen3",
        "think": "qwen3",
        "default": "qwen3",
        "deepseek-r1": "deepseek-r1",
        "qwen3": "qwen3",
        "kimi": "kimi",
        "minimax-append-think": "minimax-append-think",
        "step3": "step3",
        "step3p5": "step3p5",
        "nano-v3": "nano_v3",
        "gpt-oss": "gpt-oss",
        "glm45": "glm45",
        "deepseek-v3": "deepseek-v3",
        "kimi-k2": "kimi_k2",
        "minimax": "minimax",
        "interns1": "interns1",
    }

    sglang_model_type = type_map.get(parser_type_lower)
    if not sglang_model_type:
        raise ValueError(
            f"Unknown reasoning parser type: {parser_type}. "
            f"Supported types: {', '.join(type_map.keys())}"
        )

    # Create SGLang parser
    try:
        sglang_parser = SglangReasoningParser(
            model_type=sglang_model_type,
            stream_reasoning=stream_reasoning,
            force_reasoning=force_reasoning,
        )
        return NexRLReasoningParser(sglang_parser)
    except Exception as e:
        logger.error(f"Failed to create SGLang reasoning parser: {e}")
        logger.warning("Falling back to minimal parser")
        return MinimalReasoningParser()
