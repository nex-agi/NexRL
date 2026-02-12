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

This module provides a flexible framework for parsing tool calls from different
model formats. Different models may use different formats for tool calls:

- Simple XML: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
- Qwen 2.5/3.0: <tool_call>\n{"name": "func", "arguments": {...}}\n</tool_call>
- Custom formats can be added by extending BaseToolParser

Usage:
    # Create a parser based on config
    parser = create_tool_parser("simple_xml")

    # Parse tool calls from response
    result = parser.parse(response_text)
    if result.is_valid and result.tool_calls:
        for tool_call in result.tool_calls:
            print(f"Call {tool_call.function['name']} with {tool_call.function['arguments']}")
"""

import logging

from .base_tool_parser import BaseToolParser
from .core_types import ParseResult, ToolCallItem
from .deepseekv31_parser import DeepseekV31Parser
from .gpt_oss_parser import GptOssParser
from .qwen3_coder_parser import Qwen3CoderParser
from .qwen25_parser import Qwen25Parser
from .simple_xml_parser import SimpleXmlParser

logger = logging.getLogger(__name__)

__all__ = ["BaseToolParser", "ToolCallItem", "ParseResult", "create_tool_parser"]


def create_tool_parser(parser_type: str) -> BaseToolParser:
    """
    Factory function to create tool parsers based on type.

    Args:
        parser_type: Type of parser to create. Supported values:
            - "simple_xml" or "xml": Simple XML tag format
            - "qwen25" or "qwen" or "qwen3": Qwen 2.5/3.0 format
            - "qwen3_coder": Qwen 3 Coder format with XML-style parameters
            - "deepseekv31" or "deepseek_v31": DeepSeek V3.1 format
            - "gpt_oss": GPT OSS/Harmony format

    Returns:
        BaseToolParser: An instance of the requested tool parser

    Raises:
        ValueError: If parser_type is not recognized
    """
    parser_type = parser_type.lower()

    if parser_type in ("simple_xml", "xml"):
        return SimpleXmlParser()
    elif parser_type in ("qwen", "qwen25", "qwen3"):
        return Qwen25Parser()
    elif parser_type == "qwen3_coder":
        return Qwen3CoderParser()
    elif parser_type in ("deepseekv31", "deepseek_v31", "deepseek"):
        return DeepseekV31Parser()
    elif parser_type == "gpt_oss":
        return GptOssParser()
    else:
        raise ValueError(
            f"Unknown tool parser type: {parser_type}. "
            f"Supported types: simple_xml, xml, qwen, qwen25, qwen3, qwen3_coder, deepseekv31, gpt_oss"
        )
