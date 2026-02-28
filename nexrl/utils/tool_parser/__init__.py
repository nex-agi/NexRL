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
Tool Parser - Framework for parsing tool calls from model responses

This module wraps SGLang's function call parser to provide NexRL's interface.
Falls back to a minimal implementation if sglang is not installed.
"""

import json
import logging
import uuid
from typing import Any

from .core_types import ParseResult, ToolCallItem

logger = logging.getLogger(__name__)

__all__ = ["ToolCallItem", "ParseResult", "create_tool_parser"]

# Try to import from SGLang
try:
    from sglang.srt.entrypoints.openai.protocol import Function, Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    logger.warning(
        "SGLang not available. Install with: pip install 'NexRL[sglang]'. "
        "Falling back to minimal tool parser implementation."
    )


class SglangToolParserAdapter:
    """
    Adapter that wraps SGLang's FunctionCallParser to provide NexRL's interface.

    Handles conversion between:
    - NexRL tool dicts <-> SGLang Tool objects
    - SGLang ToolCallItem(tool_index, name, parameters) <-> NexRL ToolCallItem(id, type, function)
    """

    def __init__(self, parser_type: str):
        """
        Initialize the adapter.

        Args:
            parser_type: SGLang parser type (e.g., "qwen25", "deepseekv31")
        """
        self.parser_type = parser_type
        # Get the detector class from SGLang's parser enum
        detector_class = FunctionCallParser.ToolCallParserEnum.get(parser_type)
        if not detector_class:
            raise ValueError(
                f"Unknown tool parser type: {parser_type}. "
                f"Supported types: {', '.join(FunctionCallParser.ToolCallParserEnum.keys())}"
            )
        self.detector = detector_class()

    def _convert_tool_dict_to_sglang(self, tool_dict: dict[str, Any]) -> Tool:
        """
        Convert NexRL/OpenAI tool dict to SGLang Tool object.

        Args:
            tool_dict: Tool dictionary with OpenAI format

        Returns:
            SGLang Tool object
        """
        function_dict = tool_dict.get("function", {})
        return Tool(
            type=tool_dict.get("type", "function"),
            function=Function(
                name=function_dict.get("name"),
                description=function_dict.get("description"),
                parameters=function_dict.get("parameters"),
            ),
        )

    def _convert_sglang_tool_call_to_nexrl(self, sglang_item, idx: int) -> ToolCallItem:
        """
        Convert SGLang ToolCallItem to NexRL ToolCallItem.

        Args:
            sglang_item: SGLang ToolCallItem with tool_index, name, parameters
            idx: Index for generating unique ID

        Returns:
            NexRL ToolCallItem with id, type, function
        """
        # Parse parameters if it's a string
        if isinstance(sglang_item.parameters, str):
            try:
                # Try to parse as JSON to ensure it's valid
                params_dict = json.loads(sglang_item.parameters)
                arguments_str = json.dumps(params_dict)
            except json.JSONDecodeError:
                # If it's not valid JSON, wrap it as is
                arguments_str = sglang_item.parameters
        else:
            # If it's already a dict, convert to JSON string
            arguments_str = json.dumps(sglang_item.parameters)

        # Generate unique ID
        tool_id = f"call-{uuid.uuid4().hex}-{idx}"

        return ToolCallItem(
            id=tool_id,
            type="function",
            function={"name": sglang_item.name, "arguments": arguments_str},
        )

    def parse(self, text: str, tools: list[dict[str, Any]] | None = None) -> ParseResult:
        """
        Parse tool calls from text.

        Args:
            text: The full response text
            tools: Optional list of available tools (for validation)

        Returns:
            ParseResult with parsed tool calls and validation status
        """
        # Convert tools to SGLang format
        sglang_tools = []
        if tools:
            for tool_dict in tools:
                try:
                    sglang_tools.append(self._convert_tool_dict_to_sglang(tool_dict))
                except Exception as e:
                    logger.warning(f"Failed to convert tool dict: {e}")
                    continue

        # Use SGLang detector's detect_and_parse
        try:
            sglang_result = self.detector.detect_and_parse(text, sglang_tools)
        except Exception as e:
            logger.error(f"SGLang detector failed: {e}")
            return ParseResult(tool_calls=None, is_valid=False)

        # Check if any tool calls were found
        if not sglang_result.calls:
            return ParseResult(tool_calls=None, is_valid=True)

        # Convert SGLang tool calls to NexRL format
        nexrl_tool_calls = []
        for idx, sglang_item in enumerate(sglang_result.calls):
            try:
                nexrl_item = self._convert_sglang_tool_call_to_nexrl(sglang_item, idx)
                nexrl_tool_calls.append(nexrl_item)
            except Exception as e:
                logger.warning(f"Failed to convert tool call: {e}")
                continue

        if not nexrl_tool_calls:
            return ParseResult(tool_calls=None, is_valid=False)

        return ParseResult(tool_calls=nexrl_tool_calls, is_valid=True)


class MinimalToolParser:
    """
    Minimal fallback parser when SGLang is not available.
    Only handles basic <tool_call>JSON</tool_call> format.
    """

    # pylint: disable=unused-argument
    def parse(self, text: str, tools: list[dict[str, Any]] | None = None) -> ParseResult:
        """
        Parse tool calls from text using simple pattern matching.

        Args:
            text: The full response text
            tools: Optional list of available tools (ignored in minimal parser)

        Returns:
            ParseResult with parsed tool calls and validation status
        """
        import re

        if "<tool_call>" not in text:
            return ParseResult(tool_calls=None, is_valid=True)

        # Simple regex to extract tool_call blocks
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return ParseResult(tool_calls=None, is_valid=False)

        tool_calls = []
        for idx, match in enumerate(matches):
            try:
                tool_call = json.loads(match.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")
                continue

            if not isinstance(tool_call, dict):
                continue

            name = tool_call.get("name")
            if not isinstance(name, str):
                continue

            # Get arguments
            args = tool_call.get("arguments") or tool_call.get("parameters")
            if args is None:
                arguments_str = "{}"
            elif isinstance(args, str):
                arguments_str = args
            elif isinstance(args, dict):
                arguments_str = json.dumps(args)
            else:
                arguments_str = str(args)

            # Generate ID
            tool_id = tool_call.get("id") or f"call-{uuid.uuid4().hex}-{idx}"

            tool_calls.append(
                ToolCallItem(
                    id=tool_id,
                    type="function",
                    function={"name": name, "arguments": arguments_str},
                )
            )

        if not tool_calls:
            return ParseResult(tool_calls=None, is_valid=False)

        return ParseResult(tool_calls=tool_calls, is_valid=True)


def create_tool_parser(parser_type: str):
    """
    Factory function to create tool parsers based on type.

    Args:
        parser_type: Type of parser to create. Must match SGLang's parser names exactly.
            Supported values (from SGLang's FunctionCallParser.ToolCallParserEnum):
            - "qwen25" or "qwen": Qwen 2.5/3.0 format
            - "qwen3_coder": Qwen 3 Coder format with XML-style parameters
            - "deepseekv3": DeepSeek V3 format
            - "deepseekv31": DeepSeek V3.1 format
            - "deepseekv32": DeepSeek V3.2 format
            - "gpt-oss": GPT OSS/Harmony format
            - "llama3": Llama 3.x format
            - "mistral": Mistral format
            - "pythonic": Pythonic tool call format
            - "kimi_k2": Kimi K2 format
            - "step3": Step3 format
            - "step3p5": Step3.5 format
            - "glm" or "glm45": GLM format
            - "glm47": GLM 4.7 format
            - "hermes": Hermes format
            - "interns1": InternLM S1 format
            - "minimax-m2": MiniMax M2 format
            - "trinity": Trinity format
            - "gigachat3": GigaChat3 format
            - "mimo": MiMo format
            - "lfm2": LFM2 format

    Returns:
        Tool parser instance (SglangToolParserAdapter or MinimalToolParser)

    Raises:
        ValueError: If parser_type is not recognized (when SGLang is available)
    """
    if not SGLANG_AVAILABLE:
        logger.warning(
            f"SGLang not available, using minimal parser. Requested type '{parser_type}' ignored."
        )
        return MinimalToolParser()

    # Use parser_type directly - no mapping needed
    # SGLang will validate if it's in FunctionCallParser.ToolCallParserEnum
    try:
        return SglangToolParserAdapter(parser_type)
    except ValueError as e:
        logger.error(f"Failed to create SGLang tool parser: {e}")
        logger.warning("Falling back to minimal parser")
        return MinimalToolParser()
