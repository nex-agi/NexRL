# Reasoning Parser

The Reasoning Parser extracts thinking/reasoning content from model responses and separates it into a dedicated field, matching OpenAI's o1 model format.

## Overview

When models use chain-of-thought reasoning, they output thinking process wrapped in special tags. The reasoning parser:

1. Extracts reasoning content
2. Removes reasoning and tool call tags from main content
3. Returns clean, structured responses
4. Supports multiple model-specific formats
5. Supports streaming incremental parsing

## Format

**Model Output:**
```
<think>reasoning content here</think>actual response<tool_call>...</tool_call>
```

**Parsed Response:**
```python
{
  "content": "actual response",           # Clean content only
  "reasoning_content": "reasoning...",    # Separated reasoning
  "tool_calls": [...]                     # Parsed tool calls
}
```

## Configuration

The reasoning parser uses detector types to handle different model formats:

```yaml
inference_service:
  reasoning_parser: "think_tag"  # Default - simple, stable parser
```

### Supported Detector Types

| Detector | Format | Use For | Streaming |
|----------|--------|---------|-----------|
| `think_tag` | `<think>...</think>` | Default/Generic (legacy) | ❌ |
| `qwen3` | `<think>...</think>` | Qwen3, DeepSeek-V3, GLM-4.5, MiniMax | ✅ |
| `deepseek_r1` | `(<think>)?...</think>` | DeepSeek R1 (force reasoning) | ✅ |
| `kimi` | `◁think▷...◁/think▷` | Kimi Thinking model | ✅ |
| `step3` | `<think>...</think>` | Step3, Step3.5 models | ✅ |
| `nano_v3` | `<think>...</think>` | NanoV3 model | ✅ |
| `minimax_append_think` | Special | MiniMax (prepends tag) | ✅ |

**Examples:**
```yaml
# For Qwen3 models
inference_service:
  reasoning_parser: "qwen3"

# For DeepSeek R1
inference_service:
  reasoning_parser: "deepseek_r1"

# For Kimi model
inference_service:
  reasoning_parser: "kimi"
```

## Usage

### Automatic Integration

No code changes needed. The reasoning parser is automatically integrated:

1. **Service holders** detect reasoning tags and extract `reasoning_string`
2. **Inference client** parses reasoning and cleans content
3. **Response** includes `reasoning_content` field if present

### Basic Example

**Input:**
```python
messages = [{"role": "user", "content": "What's 2+2?"}]
response = client.generate(messages)
```

**Output:**
```python
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The answer is 4.",
      "reasoning_content": "Let me calculate: 2 + 2 = 4"
    }
  }]
}
```

### Programmatic Usage

You can also use the reasoning parser directly:

```python
from nexrl.utils.reasoning_parser import create_reasoning_parser

# Create parser with specific detector
parser = create_reasoning_parser("qwen3")

# Parse complete text
result = parser.parse(text)
print(result.reasoning_content)  # Extracted reasoning
print(result.cleaned_content)    # Clean response
print(result.is_valid)           # Parsing success

# Or get as tuple
reasoning, content = parser.parse_non_stream(text)
```

### Streaming Usage

For streaming applications, use detector-based parsers with streaming enabled:

```python
from nexrl.utils.reasoning_parser import create_reasoning_parser

# Create streaming parser
parser = create_reasoning_parser(
    parser_type="qwen3",
    stream_reasoning=True
)

# Process streaming chunks
for chunk in stream:
    reasoning, content = parser.parse_stream_chunk(chunk)
    if reasoning:
        print(f"Thinking: {reasoning}")
    if content:
        print(f"Response: {content}")
```

**Note:** The `think_tag` detector (default) does not support streaming. Use `qwen3` or other detector types for streaming support.

## Content Cleanup

The parser processes content in order:

1. **Service Holder** extracts raw `reasoning_string` and cleans reasoning tags
2. **Inference Client** parses `reasoning_string` using reasoning parser
3. **Tool Parser** extracts and removes tool call tags
4. Return clean content with separated fields

This ensures the `content` field contains only the actual response, free from reasoning and tool call markup.

## Architecture

### Component Flow

```
┌─────────────────────────────────────────────────────┐
│           Service Holder (Tinker/Weaver)            │
│  - Detects reasoning markers (</think>)             │
│  - Extracts reasoning_string                        │
│  - Cleans reasoning tags from content               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│          Inference Service Client                    │
│  - Creates reasoning parser from config             │
│  - Parses reasoning_string                          │
│  - Adds reasoning_content to message                │
└─────────────────────────────────────────────────────┘
```

### Parser Hierarchy

```
BaseReasoningFormatDetector (streaming support)
├── DeepSeekR1Detector
├── Qwen3Detector
├── KimiDetector
├── Step3Detector
├── NanoV3Detector
└── MiniMaxAppendThinkDetector

BaseReasoningParser (legacy)
└── ThinkTagReasoningParser

ReasoningParser (main wrapper)
└── Wraps detector instances
```

## Advanced Features

### Force Reasoning Mode

Some models always output reasoning content. Use `force_reasoning` parameter:

```python
parser = create_reasoning_parser(
    parser_type="deepseek_r1",
    force_reasoning=True  # Assume reasoning until </think>
)
```

### Custom Detector Implementation

To add support for a new reasoning format:

1. Create a detector class inheriting from `BaseReasoningFormatDetector`
2. Implement `detect_and_parse()` and `parse_streaming_increment()`
3. Add to the detector map in `create_reasoning_parser()`

```python
from nexrl.utils.reasoning_parser import BaseReasoningFormatDetector

class CustomDetector(BaseReasoningFormatDetector):
    def __init__(self, stream_reasoning=True, force_reasoning=False, **kwargs):
        super().__init__(
            think_start_token="<custom>",
            think_end_token="</custom>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            **kwargs
        )
```

## Backward Compatibility

The implementation maintains full backward compatibility:

- Legacy `ThinkTagReasoningParser` still available
- Default `think_tag` parser unchanged
- Existing configs continue to work
- No breaking changes to APIs

## Choosing the Right Detector

| Use Case | Recommended Detector | Reason |
|----------|---------------------|---------|
| Generic/Unknown | `think_tag` | Stable, simple, works for most cases |
| Qwen3 family | `qwen3` | Optimized for Qwen3 models, streaming support |
| DeepSeek R1 | `deepseek_r1` | Handles R1's forced reasoning mode |
| Kimi model | `kimi` | Supports Kimi's special markers |
| Streaming apps | `qwen3` or others | All except `think_tag` support streaming |
| Production stable | `think_tag` | Battle-tested, no surprises |

## Troubleshooting

### Reasoning Not Extracted

**Problem:** `reasoning_content` field is missing

**Solutions:**
1. Check if model is outputting `</think>` tag
2. Verify correct detector type in config
3. Check service holder logs for `reasoning_string`

### Incorrect Parsing

**Problem:** Reasoning content includes unwanted text

**Solutions:**
1. Try a model-specific detector instead of `think_tag`
2. Check if model format matches detector expectations
3. Enable debug logging to see parsing steps

### Streaming Not Working

**Problem:** Streaming parser not returning incremental results

**Solution:** Ensure you're not using `think_tag` detector - it doesn't support streaming. Use `qwen3` or another detector type.

## Performance Considerations

- **Legacy parser (`think_tag`)**: Regex-based, fast for small responses
- **Detector-based parsers**: String operations, optimized for streaming
- **Memory**: Minimal overhead, detectors maintain small buffers
- **Latency**: Streaming detectors add <1ms per chunk

## See Also

- [Tool Parser](./tool-parser.md) - Tool call parsing framework (similar architecture)
- [Inference Service](../07-services/inference-service.md) - Inference service architecture
- [Service Holders](../07-services/service-holders.md) - Tinker/Weaver integration
