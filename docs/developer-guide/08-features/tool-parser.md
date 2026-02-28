# Tool Parser Framework

The Tool Parser Framework provides a flexible, extensible system for parsing tool calls from different model formats. Different language models use different formats for tool calls, and this framework allows you to easily switch between parsers or add new ones via configuration.

## Dependencies

The tool parser now uses [SGLang's](https://github.com/sgl-project/sglang) function call parser implementation under the hood. To use the full functionality, install with:

```bash
pip install 'NexRL[sglang]'
```

If SGLang is not installed, NexRL will fall back to a minimal parser that only handles basic `<tool_call>JSON</tool_call>` format.

## Overview

Previously, tool parsing logic was hardcoded in the service holders (Weaver and Tinker). The tool parser framework refactors this into a centralized, configurable system that separates concerns:

- **Service Holders** detect and extract raw tool strings
- **Tool Parsers** parse tool strings into structured format (via SGLang detectors)
- **Inference Service Client** coordinates parsing and returns OpenAI-format results

## Architecture

The framework consists of three main components:

### 1. SGLang Format Detectors

NexRL uses SGLang's `BaseFormatDetector` implementations for parsing. SGLang provides detectors for:
- `Qwen25Detector` - Qwen 2.5/3.0 multi-line JSON format
- `Qwen3CoderDetector` - Qwen 3 Coder XML-style parameters
- `DeepSeekV3Detector` - DeepSeek V3 format
- `DeepSeekV31Detector` - DeepSeek V3.1 special Unicode tokens
- `DeepSeekV32Detector` - DeepSeek V3.2 format
- `GptOssDetector` - GPT OSS/Harmony commentary format
- `Llama32Detector` - Llama 3.x format
- `MistralDetector` - Mistral format
- `PythonicDetector` - Pythonic tool call format
- `KimiK2Detector` - Kimi K2 format
- `Step3Detector` - Step3 format
- `GlmDetector` - GLM format
- And many more...

### 2. SglangToolParserAdapter

NexRL's adapter wraps SGLang detectors to provide a unified interface:
- Converts NexRL tool dicts to SGLang `Tool` objects
- Converts SGLang `ToolCallItem` to NexRL `ToolCallItem` format
- Handles tool name validation when tools are provided

### 3. Factory Function

`create_tool_parser(parser_type: str)` instantiates parsers by type:

```python
from nexrl.utils.tool_parser import create_tool_parser

parser = create_tool_parser("qwen25")
```

### 4. Core Types

Data classes for structured results (maintained for backward compatibility):

```python
@dataclass
class ToolCallItem:
    id: str                    # Unique identifier
    type: str                  # "function"
    function: dict[str, Any]   # {"name": str, "arguments": str}

@dataclass
class ParseResult:
    tool_calls: list[ToolCallItem] | None
    is_valid: bool
```

**Note:** Internally, SGLang uses a different format (`tool_index`, `name`, `parameters`), but NexRL's adapter transparently converts to the OpenAI-compatible format above.

## Supported Formats

All formats are provided by SGLang's detectors. See the [Config Options](#all-config-options) table below for the complete list.

### Qwen 2.5/3.0 (`qwen25`)

Multi-line JSON format supporting multiple calls:

```xml
<tool_call>
{"name": "get_weather", "arguments": {"city": "NYC"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"operation": "add", "a": 5, "b": 3}}
</tool_call>
```

**Config:** `qwen25`, `qwen`, or `qwen3`

**Use case:** Qwen family models, multiple tool calls per response

### Qwen 3 Coder (`qwen3_coder`)

XML-style function and parameter tags:

```xml
<tool_call>
<function=execute_bash>
<parameter=command>
pwd && ls
</parameter>
<parameter=flag>
-la
</parameter>
</function>
</tool_call>
```

**Config:** `qwen3_coder`

**Use case:** Qwen 3 Coder models, multi-parameter functions with text content

**Features:**
- Multiple named parameters per function
- Text-based parameter values (not just JSON)
- Safe value parsing (JSON → Python literals → raw string)

### DeepSeek V3.1 (`deepseekv31`)

Special Unicode tokens with JSON arguments:

```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

Multiple calls:

```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>calculate<｜tool▁sep｜>{"operation": "add"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

**Config:** `deepseekv31`, `deepseek_v31`, or `deepseek`

**Use case:** DeepSeek V3.1 models

**Features:**
- Special Unicode tokens for delimiting
- Function name followed by JSON arguments
- Supports multiple tool calls in one response

### GPT OSS (`gpt_oss`)

Harmony-style commentary channel format:

```
<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location": "Tokyo"}<|call|>
```

With namespace:

```
<|channel|>commentary to=tools.calculate<|constrain|>json<|message|>{"operation": "add", "a": 5, "b": 3}<|call|>
```

**Config:** `gpt_oss`

**Use case:** GPT OSS/T4-style models with harmony format

**Features:**
- Commentary channel with `to=` directive
- Supports namespaced functions (e.g., `functions.name`, `tools.name`)
- JSON constraint specification

**Note:** This is a simplified version for basic parsing. Full streaming support requires HarmonyParser.

## Configuration

Specify the parser type in your config:

```yaml
inference_service:
  tool_parser: "qwen25"  # Default
```

If not specified, `qwen25` is used as the default.

### All Config Options

**Important:** Use SGLang's parser names directly in your config. No mapping or aliases.

| Parser | Config Value | Format | Notes |
|--------|-------------|--------|-------|
| Qwen 2.5/3.0 | `qwen25` or `qwen` | Multi-line JSON with tags | Default |
| Qwen 3 Coder | `qwen3_coder` | XML-style parameters | |
| DeepSeek V3 | `deepseekv3` | DeepSeek V3 format | |
| DeepSeek V3.1 | `deepseekv31` | Unicode tokens | |
| DeepSeek V3.2 | `deepseekv32` | DeepSeek V3.2 format | |
| GPT OSS | `gpt-oss` | Harmony commentary | |
| Llama 3.x | `llama3` | Llama tool format | |
| Mistral | `mistral` | Mistral format | |
| Pythonic | `pythonic` | Python function calls | |
| Kimi K2 | `kimi_k2` | Kimi K2 format | |
| Step3 | `step3` | Step3 format | |
| Step3.5 | `step3p5` | Step3.5 format | |
| GLM | `glm` or `glm45` | GLM format | |
| GLM 4.7 | `glm47` | GLM 4.7 format | |
| Hermes | `hermes` | Hermes format | |
| InternLM | `interns1` | InternLM S1 format | |
| MiniMax M2 | `minimax-m2` | MiniMax M2 format | |
| Trinity | `trinity` | Trinity format | |
| GigaChat3 | `gigachat3` | GigaChat3 format | |
| MiMo | `mimo` | MiMo format | |
| LFM2 | `lfm2` | LFM2 format | |

**Note:** All parsers are powered by SGLang's detectors. Parser names must match SGLang's `FunctionCallParser.ToolCallParserEnum` exactly.

## Usage

### Basic Usage

```python
from nexrl.utils.tool_parser import create_tool_parser

# Create a parser
parser = create_tool_parser("qwen25")

# Define available tools (optional, for validation)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }
]

# Parse response text
response = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
result = parser.parse(response, tools=tools)

if result.is_valid and result.tool_calls:
    for tool_call in result.tool_calls:
        print(f"Function: {tool_call.function['name']}")
        print(f"Arguments: {tool_call.function['arguments']}")
        print(f"ID: {tool_call.id}")
```

**Note:** Passing `tools` is optional but recommended for validation. SGLang will validate tool names against the provided tools list.

### Integration with NexRL

The tool parser is automatically integrated with the inference service client:

1. **Service Holder** (Tinker/Weaver) detects tool calls and returns raw `tool_string`
2. **Inference Service Client** uses configured parser to parse the `tool_string`
3. **Parsed tool calls** are converted to OpenAI format and returned in the response

```python
# In remote_api_inference_service_client.py
parser_type = config.inference_service.get("tool_parser", "simple_xml")
self._tool_parser = create_tool_parser(parser_type)

# Later, during generation
if tool_string:
    parse_result = self._tool_parser.parse(tool_string)
    if parse_result.is_valid and parse_result.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": tc.function,
            }
            for tc in parse_result.tool_calls
        ]
```

## Adding a New Parser

To add support for a new tool call format:

### Step 1: Create Parser Class

Create a new file in `nexrl/utils/tool_parser/` (e.g., `my_model_parser.py`):

```python
from .base_tool_parser import BaseToolParser
from .core_types import ParseResult, ToolCallItem
import json
import uuid
import re

class MyModelParser(BaseToolParser):
    def __init__(self):
        super().__init__()
        self._pattern = re.compile(r"YOUR_REGEX_PATTERN")

    def detect_tool_call(self, text: str) -> bool:
        """Check if text contains your model's tool call markers"""
        return "YOUR_MARKER" in text

    def extract_tool_string(self, text: str) -> str | None:
        """Extract the tool call string"""
        if self.detect_tool_call(text):
            return text  # Or extract specific portion
        return None

    def parse_tool_string(self, tool_string: str) -> ParseResult:
        """Parse the tool string into structured tool calls"""
        matches = self._pattern.findall(tool_string)

        if not matches:
            return ParseResult(tool_calls=None, is_valid=False)

        tool_calls = []
        for idx, match in enumerate(matches):
            # Extract function name and arguments
            func_name = match[0]
            func_args = json.loads(match[1])

            # Create tool call item
            tool_call_item = ToolCallItem(
                id=f"call-{uuid.uuid4().hex}-{idx}",
                type="function",
                function={
                    "name": func_name,
                    "arguments": json.dumps(func_args)
                }
            )
            tool_calls.append(tool_call_item)

        return ParseResult(tool_calls=tool_calls, is_valid=True)
```

### Step 2: Register in Factory

Add import and registration in `nexrl/utils/tool_parser/__init__.py`:

```python
from .my_model_parser import MyModelParser

def create_tool_parser(parser_type: str) -> BaseToolParser:
    parser_type = parser_type.lower()

    if parser_type == "my_model":
        return MyModelParser()
    # ... existing parsers
```

### Step 3: Update Documentation

Document the new parser format and config options.

## Implementation Details

### Naming Convention

- **Base class:** `BaseToolParser`
- **Concrete classes:** Model name prefix + `Parser` (e.g., `Qwen25Parser`, `DeepseekV31Parser`)
- **Files:** Model name prefix + `_parser.py` (e.g., `qwen25_parser.py`, `deepseekv31_parser.py`)
- **Config values:** Model name (e.g., `qwen25`, `qwen3_coder`, `deepseekv31`)

### Modified Components

#### Service Holders

Both `weaver_service_holder.py` and `tinker_service_holder.py` were modified:

**Before:**
```python
# Hardcoded parsing logic
def _parse_and_normalize_tool_call(self, tool_call_str: str):
    # 50+ lines of parsing code
    pass
```

**After:**
```python
# Just extract raw string
tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
if tool_call_match:
    tool_string = tool_call_match.group(0)  # Return full match

return {
    "tool_string": tool_string,  # Not tool_calls
    # ... other fields
}
```

#### Inference Service Client

`remote_api_inference_service_client.py` now handles parsing:

```python
# Initialize parser based on config
tool_parser_type = config.inference_service.get("tool_parser", "qwen25")
self._tool_parser = create_tool_parser(tool_parser_type)

# Use parser during generation
if tool_string:
    parse_result = self._tool_parser.parse(tool_string)
    if parse_result.is_valid and parse_result.tool_calls:
        tool_calls = [...]  # Convert to OpenAI format
```

## Benefits

### 1. Separation of Concerns
- Service holders: detect and extract (format-agnostic)
- Parsers: parse and validate (format-specific)
- Inference client: coordinate and format results

### 2. Flexibility
- Switch parsers via configuration
- No code changes required
- Support multiple formats simultaneously

### 3. Extensibility
- Add new parsers without modifying existing code
- Follow simple interface pattern
- Register in factory function

### 4. Maintainability
- Parsing logic centralized in one place
- Easier to test individual parsers
- Clear separation of responsibilities

### 5. Format Agnostic
- Service holders don't need to know format details
- Adding new formats doesn't impact service holders
- Clean abstraction boundary

## Backward Compatibility

The implementation maintains full backward compatibility:

- **Default behavior:** Uses `qwen25` parser (multi-line JSON format)
- **Service holder API:** Compatible (returns `tool_string` field)
- **Client API:** Returns tool calls in OpenAI format (unchanged)
- **Configuration:** No changes required (will use default)

## File Structure

```
nexrl/utils/tool_parser/
├── __init__.py                    # Factory function, SglangToolParserAdapter
└── core_types.py                  # ToolCallItem, ParseResult

Note: Parser implementations are now provided by SGLang.
```

## Reference

This implementation is inspired by SGLang's tool parser framework (`.vibe/tool_parser/`), adapted for NexRL's architecture and needs.

## See Also

- [Reasoning Parser](./reasoning-parser.md) - Extract and structure reasoning content
- [Inference Service](../07-services/inference-service.md) - Inference service architecture
- [Service Holders](../07-services/service-holders.md) - Tinker and Weaver service holders
- [Configuration Reference](../11-configuration-reference/service-config.md) - Service configuration options
