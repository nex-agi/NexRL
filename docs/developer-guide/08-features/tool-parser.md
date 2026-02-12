# Tool Parser Framework

The Tool Parser Framework provides a flexible, extensible system for parsing tool calls from different model formats. Different language models use different formats for tool calls, and this framework allows you to easily switch between parsers or add new ones via configuration.

## Overview

Previously, tool parsing logic was hardcoded in the service holders (Weaver and Tinker). The tool parser framework refactors this into a centralized, configurable system that separates concerns:

- **Service Holders** detect and extract raw tool strings
- **Tool Parsers** parse tool strings into structured format
- **Inference Service Client** coordinates parsing and returns OpenAI-format results

## Architecture

The framework consists of four main components:

### 1. BaseToolParser

Abstract base class defining the parser interface. All parsers must implement three methods:

```python
class BaseToolParser(ABC):
    @abstractmethod
    def detect_tool_call(self, text: str) -> bool:
        """Check if text contains tool calls in this parser's format"""

    @abstractmethod
    def extract_tool_string(self, text: str) -> str | None:
        """Extract the raw tool call string from response text"""

    @abstractmethod
    def parse_tool_string(self, tool_string: str) -> ParseResult:
        """Parse the extracted tool string into structured tool calls"""
```

### 2. Concrete Parsers

Implementations for specific model formats:
- `SimpleXmlParser` - Simple XML tag format
- `Qwen25Parser` - Qwen 2.5/3.0 multi-line JSON format
- `Qwen3CoderParser` - Qwen 3 Coder XML-style parameters
- `DeepseekV31Parser` - DeepSeek V3.1 special Unicode tokens
- `GptOssParser` - GPT OSS/Harmony commentary format

### 3. Factory Function

`create_tool_parser(parser_type: str)` instantiates parsers by type:

```python
from nexrl.utils.tool_parser import create_tool_parser

parser = create_tool_parser("qwen25")
```

### 4. Core Types

Data classes for structured results:

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

## Supported Formats

### Simple XML (`simple_xml`)

Single-line JSON within XML tags:

```xml
<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>
```

**Config:** `simple_xml` or `xml`

**Use case:** Default format, simple tool calls with inline JSON

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

| Parser | Config Values | Format |
|--------|---------------|--------|
| Simple XML | `simple_xml`, `xml` | `<tool_call>{json}</tool_call>` |
| Qwen 2.5/3.0 | `qwen25`, `qwen`, `qwen3` | Multi-line JSON with tags |
| Qwen 3 Coder | `qwen3_coder` | XML-style parameters |
| DeepSeek V3.1 | `deepseekv31`, `deepseek_v31`, `deepseek` | Unicode tokens |
| GPT OSS | `gpt_oss` | Harmony commentary |

## Usage

### Basic Usage

```python
from nexrl.utils.tool_parser import create_tool_parser

# Create a parser
parser = create_tool_parser("simple_xml")

# Parse response text
response = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
result = parser.parse(response)

if result.is_valid and result.tool_calls:
    for tool_call in result.tool_calls:
        print(f"Function: {tool_call.function['name']}")
        print(f"Arguments: {tool_call.function['arguments']}")
        print(f"ID: {tool_call.id}")
```

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
tool_parser_type = config.inference_service.get("tool_parser", "simple_xml")
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

- **Default behavior:** Uses `simple_xml` parser (same format as original)
- **Service holder API:** Compatible (returns `tool_string` field)
- **Client API:** Returns tool calls in OpenAI format (unchanged)
- **Configuration:** No changes required (will use default)

## File Structure

```
nexrl/utils/tool_parser/
├── __init__.py                    # Factory function and exports
├── core_types.py                  # ToolCallItem, ParseResult
├── base_tool_parser.py           # BaseToolParser abstract class
├── simple_xml_parser.py          # SimpleXmlParser
├── qwen25_parser.py              # Qwen25Parser
├── qwen3_coder_parser.py         # Qwen3CoderParser
├── deepseekv31_parser.py         # DeepseekV31Parser
└── gpt_oss_parser.py             # GptOssParser
```

## Reference

This implementation is inspired by SGLang's tool parser framework (`.vibe/tool_parser/`), adapted for NexRL's architecture and needs.

## See Also

- [Reasoning Parser](./reasoning-parser.md) - Extract and structure reasoning content
- [Inference Service](../07-services/inference-service.md) - Inference service architecture
- [Service Holders](../07-services/service-holders.md) - Tinker and Weaver service holders
- [Configuration Reference](../11-configuration-reference/service-config.md) - Service configuration options
