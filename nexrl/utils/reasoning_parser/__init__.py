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
Reasoning Parser Framework

Provides a flexible, extensible system for parsing reasoning/thinking content
from different model formats.
"""

from typing import Dict, Optional, Tuple, Type

from .base_reasoning_parser import BaseReasoningFormatDetector, BaseReasoningParser
from .core_types import ReasoningParseResult
from .detectors import (
    DeepSeekR1Detector,
    KimiDetector,
    MiniMaxAppendThinkDetector,
    NanoV3Detector,
    Qwen3Detector,
    Step3Detector,
)
from .think_tag_reasoning_parser import ThinkTagReasoningParser

__all__ = [
    "BaseReasoningParser",
    "BaseReasoningFormatDetector",
    "ReasoningParseResult",
    "ThinkTagReasoningParser",
    "ReasoningParser",
    "create_reasoning_parser",
    # Detectors
    "DeepSeekR1Detector",
    "Qwen3Detector",
    "KimiDetector",
    "MiniMaxAppendThinkDetector",
    "NanoV3Detector",
    "Step3Detector",
]


class ReasoningParser:
    """
    Parser that handles both streaming and non-streaming scenarios for extracting
    reasoning content from model outputs.

    This is the main parser class that wraps a detector for convenience.

    Args:
        detector (BaseReasoningFormatDetector): The detector instance to use
    """

    def __init__(self, detector: BaseReasoningFormatDetector):
        self.detector = detector

    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Non-streaming call: one-time parsing

        Args:
            full_text: The complete text to parse

        Returns:
            Tuple of (reasoning_content, cleaned_content)
        """
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_content, ret.cleaned_content

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Streaming call: incremental parsing

        Args:
            chunk_text: The text chunk to parse incrementally

        Returns:
            Tuple of (reasoning_content, cleaned_content)
        """
        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_content, ret.cleaned_content

    def parse(self, text: str) -> ReasoningParseResult:
        """
        Convenience method for non-streaming parsing that returns ReasoningParseResult.

        This method maintains backward compatibility with the old parser interface.

        Args:
            text: The text to parse

        Returns:
            ReasoningParseResult with reasoning_content and cleaned_content
        """
        return self.detector.detect_and_parse(text)


def create_reasoning_parser(
    parser_type: str = "think_tag",
    stream_reasoning: bool = False,
    force_reasoning: Optional[bool] = None,
) -> BaseReasoningParser | ReasoningParser:
    """
    Factory function to create reasoning parsers.

    Args:
        parser_type: Type of detector to create.
                    Supported: "think_tag", "deepseek_r1", "qwen3", "kimi",
                              "minimax_append_think", "step3", "nano_v3"
        stream_reasoning: Enable streaming mode
        force_reasoning: Force reasoning mode (if None, uses detector default)

    Returns:
        BaseReasoningParser or ReasoningParser: An instance of the requested parser

    Raises:
        ValueError: If parser_type is not recognized
    """
    parser_type_lower = parser_type.lower().replace("-", "_")

    # Map detector types to detector classes
    detector_map: Dict[str, Type[BaseReasoningFormatDetector]] = {
        "think_tag": Qwen3Detector,
        "think": Qwen3Detector,
        "default": Qwen3Detector,
        "deepseek_r1": DeepSeekR1Detector,
        "qwen3": Qwen3Detector,
        "kimi": KimiDetector,
        "minimax_append_think": MiniMaxAppendThinkDetector,
        "step3": Step3Detector,
        "nano_v3": NanoV3Detector,
    }

    # Legacy mode: Use old ThinkTagReasoningParser for backward compatibility
    if parser_type_lower in ["think_tag", "think"] and not stream_reasoning:
        return ThinkTagReasoningParser()

    # Get detector class
    detector_class = detector_map.get(parser_type_lower)
    if not detector_class:
        raise ValueError(
            f"Unknown reasoning parser type: {parser_type}. "
            f"Supported types: {', '.join(detector_map.keys())}"
        )

    # Build kwargs for detector
    kwargs = {"stream_reasoning": stream_reasoning}
    if force_reasoning is not None:
        kwargs["force_reasoning"] = force_reasoning

    # Create detector and wrap in ReasoningParser
    detector = detector_class(**kwargs)  # type: ignore[arg-type]
    return ReasoningParser(detector=detector)
