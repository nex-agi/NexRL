# Inference Service

The inference service provides LLM inference capabilities for rollout workers. NexRL supports multiple inference backends through a unified client interface.

## Architecture

```
InferenceServiceClient (Abstract)
        ↓
    ┌────────────────────────────┐
    ↓                            ↓
OpenAIInferenceServiceClient  RemoteApiInferenceServiceClient
 (OpenAI-compatible API)       (Tinker/Weaver)
```

## InferenceServiceClient

Abstract base class for all inference service clients. Located in `nexrl/inference_service_client/base_inference_service_client.py`.

### Core Interface

```python
class InferenceServiceClient(ABC):
    @abstractmethod
    def completion(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Call LLM inference for prompt completion."""

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """Call LLM inference for chat completion."""

    @abstractmethod
    def apply_chat_template(
        self, messages: list[dict], tools: list[dict] | None = None, ...
    ) -> str | list[int]:
        """Apply chat template to messages using tokenizer."""
```

### OpenAI-Compatible API

The client provides OpenAI-style access patterns:

```python
# Using the client directly
response = client.completion(prompt="Hello", max_tokens=100)

# Using OpenAI-style interface
response = client.completions.create(prompt="Hello", max_tokens=100)
response = client.chat.completions.create(messages=[...])
```

### Weight Synchronization Integration

Inference clients integrate with the weight sync controller to ensure model consistency:

```python
def set_weight_sync_controller(self, controller, model_tag: str):
    """Set the weight sync controller reference."""
    self._weight_sync_controller = controller
    self._model_tag = model_tag

def _wait_for_weight_sync(self):
    """Block inference until weight sync completes."""
    if self._weight_sync_controller is not None:
        self._weight_sync_controller.wait_for_weight_sync(self._model_tag)
```

When `freeze_for_weight_sync: true`, inference calls automatically wait for weight updates.

## OpenAIInferenceServiceClient

Inference client for OpenAI-compatible APIs (vLLM, OpenAI, etc.). Located in `nexrl/inference_service_client/openai_inference_service_client.py`.

### Constructor

```python
def __init__(self, config: DictConfig)
```

**Configuration:**
```yaml
inference_service:
  api_key: "EMPTY"
  base_url: "http://localhost:8000"
  model: "meta-llama/Llama-3.1-8B-Instruct"
  tokenizer: "meta-llama/Llama-3.1-8B-Instruct"  # optional
  max_tokens: 2048
  max_retries: 10
  freeze_for_weight_sync: true
```

### Key Features

#### 1. OpenAI Client Initialization

```python
self._oai_llm = openai.OpenAI(
    api_key=config.inference_service.api_key,
    base_url=config.inference_service.base_url + "/v1",
    timeout=1000,
)
```

#### 2. Tokenizer Management

```python
# Uses HuggingFace tokenizer for chat templates
tokenizer_path = config.inference_service.get("tokenizer", config.inference_service.model)
self.tokenizer = hf_tokenizer(tokenizer_path)
```

#### 3. NexRL Training Data Extraction

The client extracts training-specific data from responses:

```python
def completion(self, prompt: str, **kwargs) -> dict[str, Any]:
    # Get completion from OpenAI API
    completion = self._oai_llm.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
        extra_body={
            "include_stop_str_in_output": True,
            "no_stop_trim": True,
            "return_tokens_as_token_ids": True,
            "skip_special_tokens": False,
        },
        **kwargs
    )

    # Extract NexRL-specific training data
    completion_dict["nexrl_train"] = {
        "prompt_tokens": prompt_tokens,      # Token IDs
        "response_tokens": response_tokens,  # Token IDs
        "response_logprobs": response_logprobs  # Log probabilities
    }

    return completion_dict
```

### Usage Example

```python
from nexrl.inference_service_client import OpenAIInferenceServiceClient
from omegaconf import DictConfig

config = DictConfig({
    "inference_service": {
        "api_key": "EMPTY",
        "base_url": "http://localhost:8000",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "max_tokens": 2048
    },
    "temperature": 0.7
})

client = OpenAIInferenceServiceClient(config)

# Prompt completion
response = client.completion(prompt="What is 2+2?", max_tokens=100)

# Chat completion
messages = [{"role": "user", "content": "Hello!"}]
response = client.generate(messages=messages, max_tokens=100)
```

## RemoteApiInferenceServiceClient

Inference client for Tinker/Weaver service backends. Located in `nexrl/inference_service_client/remote_api_inference_service_client.py`.

### Constructor

```python
def __init__(self, config: DictConfig)
```

**Configuration:**
```yaml
inference_service:
  backend: "tinker"  # or "weaver"
  model: "meta-llama/Llama-3.1-8B-Instruct"
  max_tokens: 2048
  freeze_for_weight_sync: true
```

### Service Holder Pattern

The client uses a service holder for backend communication:

```python
def set_service_holder(self, service_holder):
    """Set the service holder (TinkerServiceHolder or WeaverServiceHolder)."""
    self._service_holder = service_holder

def completion(self, prompt: str, **kwargs) -> dict[str, Any]:
    # Call service holder's sampling method
    result = self._service_holder.sample_from_prompt(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        num_samples=1,
    )

    # Convert to OpenAI-compatible format
    return self._format_as_openai_response(result)
```

### Key Differences from OpenAI Client

| Feature | OpenAIInferenceServiceClient | RemoteApiInferenceServiceClient |
|---------|------------------------------|----------------------------------|
| **Backend** | OpenAI-compatible HTTP API | Tinker/Weaver service holder |
| **Tokenizer** | HuggingFace tokenizer | Service-side tokenizer |
| **Weight Sync** | Via API model updates | Via service holder path updates |
| **Retry Logic** | Built-in retries | Service holder handles retries |

## Tokenizer Utilities

NexRL provides tokenizer utilities in `base_inference_service_client.py`:

### hf_tokenizer()

```python
def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a HuggingFace tokenizer with corrections."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)

    # Set pad_token_id if missing
    if correct_pad_token and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Handle Gemma2 EOS token ambiguity
    if correct_gemma2 and "gemma-2-2b-it" in name_or_path:
        tokenizer.eos_token = "<end_of_turn>"
        tokenizer.eos_token_id = 107

    return tokenizer
```

## Configuration Examples

### OpenAI-Compatible (vLLM)

```yaml
inference_service:
  backend: "openai"
  api_key: "EMPTY"
  base_url: "http://localhost:8000"
  model: "meta-llama/Llama-3.1-8B-Instruct"
  max_tokens: 2048
  max_retries: 10
  freeze_for_weight_sync: true
  model_tag: "default"

temperature: 0.7
```

### Tinker Service

```yaml
inference_service:
  backend: "tinker"
  base_url: "http://tinker-service:8000"
  model: "meta-llama/Llama-3.1-8B-Instruct"
  max_tokens: 2048
  freeze_for_weight_sync: true

temperature: 0.7
```

### Weaver Service

```yaml
inference_service:
  backend: "weaver"
  base_url: "http://weaver-service:8000"
  model: "Qwen/Qwen2.5-7B-Instruct"
  max_tokens: 2048
  freeze_for_weight_sync: true

temperature: 0.7
```

## Response Format

All inference clients return responses in a consistent format:

```python
{
    # Standard OpenAI fields
    "id": "cmpl-...",
    "object": "text_completion",
    "created": 1234567890,
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "choices": [
        {
            "text": "The answer is 4.",
            "index": 0,
            "logprobs": {...},
            "finish_reason": "stop"
        }
    ],

    # NexRL training data
    "nexrl_train": {
        "prompt_tokens": [1234, 5678, ...],      # Token IDs
        "response_tokens": [9012, 3456, ...],    # Token IDs
        "response_logprobs": [-0.1, -0.2, ...]   # Log probabilities
    }
}
```

## Best Practices

### 1. Use Weight Sync Blocking

Enable `freeze_for_weight_sync` to ensure consistency:

```yaml
inference_service:
  freeze_for_weight_sync: true
```

This blocks inference calls during weight updates, preventing stale model usage.

### 2. Configure Retries

For production, configure retry logic:

```yaml
inference_service:
  max_retries: 10  # Retry failed requests
```

### 3. Token Management

Ensure tokenizer matches the model:

```yaml
inference_service:
  model: "meta-llama/Llama-3.1-8B-Instruct"
  tokenizer: "meta-llama/Llama-3.1-8B-Instruct"  # Same path
```

### 4. Temperature Settings

Configure temperature at the top level (not in inference_service):

```yaml
temperature: 0.7  # Global temperature setting
```

## Related Documentation

- [Weight Synchronization](../08-features/weight-synchronization.md) - Weight sync integration
- [Rollout Workers](../05-rollout-workers/overview.md) - Inference usage in rollout
- [Service Holders](./service-holders.md) - Tinker/Weaver service holders
- [Configuration Reference](../11-configuration-reference/service-config.md) - Complete service configuration
