# Service Holders

Service holders provide centralized management for Tinker and Weaver service backends. They encapsulate all service interactions, tokenization, and response parsing in a single, Ray-serializable component.

## Purpose

Service holders solve key challenges in distributed RL training:

1. **Centralized Service Management**: Single instance manages all service clients (training, sampling)
2. **Tokenization**: Handles tokenization internally to avoid passing non-serializable objects through Ray
3. **Response Parsing**: Parses and normalizes service responses to standard format
4. **Weight Management**: Manages weight updates and sampling client synchronization

## Architecture

```
ServiceHolder Pattern
    ↓
┌──────────────────────────────┐
↓                              ↓
TinkerServiceHolder      WeaverServiceHolder
(Tinker SDK)             (Weaver SDK)
```

Both holders provide a unified API for NexRL components.

## Common Interface

Both service holders implement these key methods:

### Tokenization

```python
def apply_chat_template(
    self,
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
) -> str | list[int]:
    """Apply chat template to messages."""
```

### Sampling

```python
def sample_from_prompt(
    self, prompt: str, max_tokens: int, temperature: float = 1.0, num_samples: int = 1
) -> dict:
    """Sample from the model given a prompt string."""

def sample_from_messages(
    self,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 1.0,
    num_samples: int = 1,
    tools: list[dict] | None = None,
) -> dict:
    """Sample from the model given chat messages."""
```

### Training

```python
def train(self, datums: list[dict], config: dict) -> dict:
    """Execute training step with datums."""

def update_sampling_weights(self, weight_path: str | None = None) -> str:
    """Update sampling client with new weights."""
```

## TinkerServiceHolder

Centralized manager for Tinker services. Located in `nexrl/tinker/tinker_service_holder.py`.

### Constructor

```python
def __init__(
    self,
    base_model: str = "",
    lora_rank: int = 32,
    base_url: str | None = None,
    renderer_name: str | None = None,
    tokenizer_path: str | None = None,
)
```

**Parameters:**
- `base_model`: Base model name for training
- `lora_rank`: LoRA rank for fine-tuning
- `base_url`: Tinker service URL
- `renderer_name`: Renderer for response parsing (auto-detected if None)
- `tokenizer_path`: Custom tokenizer path (uses base_model if None)

### Initialization Process

```python
# 1. Create service client
self._service_client = tinker.ServiceClient(base_url=base_url)

# 2. Initialize tokenizer and renderer
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

self._tokenizer = get_tokenizer(tokenizer_path or base_model)
renderer_name = renderer_name or model_info.get_recommended_renderer_name(base_model)
self._renderer = renderers.get_renderer(renderer_name, self._tokenizer)

# 3. Create training client
self._training_client = self._service_client.create_lora_training_client(
    base_model=base_model, rank=lora_rank
)

# 4. Save initial weights and create sampling client
sampling_path = self._training_client.save_weights_for_sampler(name="initial").result().path
self._sampling_client = self._service_client.create_sampling_client(
    model_path=sampling_path
)
```

### Sampling from Messages

```python
def sample_from_messages(
    self,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 1.0,
    num_samples: int = 1,
    tools: list[dict] | None = None,
) -> dict:
    """Sample using chat messages."""
    from tinker import types

    # 1. Apply chat template to get tokens
    prompt_tokens = self.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=True
    )

    # 2. Create model input
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

    # 3. Set sampling parameters
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=self._stop_sequences,
    )

    # 4. Sample from model
    sample_result = self._sampling_client.sample(
        prompt=model_input,
        num_samples=num_samples,
        sampling_params=sampling_params,
    ).result()

    # 5. Parse and normalize response
    sequence = sample_result.sequences[0]
    response_tokens = list(sequence.tokens)
    response_logprobs = list(sequence.logprobs)

    # 6. Decode response text
    response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=False)

    # 7. Parse for tool calls (if any)
    tool_calls = self._parse_tool_calls(response_text)

    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "response_logprobs": response_logprobs,
        "response_text": response_text,
        "tool_calls": tool_calls,
        "finish_reason": "stop" if sequence.finish_reason == "stop" else "length"
    }
```

### Training

```python
def train(self, datums: list[dict], config: dict) -> dict:
    """Execute training step."""
    from tinker import types

    # Convert datums to Tinker TrainingDatum format
    training_datums = []
    for d in datums:
        training_datums.append(types.TrainingDatum(
            prompt=d["prompt"],
            response=d["response"],
            advantage=d.get("advantage", 1.0),
            metadata=d.get("metadata", {})
        ))

    # Execute training
    result = self._training_client.train(
        datums=training_datums,
        optimizer_config=types.OptimizerConfig(
            learning_rate=config.get("learning_rate", 2e-6),
            # ... other optimizer params
        )
    ).result()

    return {"metrics": result.metrics, "meta_info": result.meta_info}
```

### Weight Synchronization

```python
def update_sampling_weights(self, weight_path: str | None = None) -> str:
    """Update sampling client with new weights."""
    if weight_path is None:
        # Save current training weights
        weight_path = self._training_client.save_weights_for_sampler(
            name=f"step_{self._step_counter}"
        ).result().path

    # Update current path
    self._current_sampling_path = weight_path

    # Create new sampling client
    self._sampling_client = self._service_client.create_sampling_client(
        model_path=weight_path
    )

    return weight_path
```

## WeaverServiceHolder

Centralized manager for Weaver services. Located in `nexrl/weaver/weaver_service_holder.py`.

### Constructor

```python
def __init__(
    self,
    base_model: str = "",
    lora_rank: int = 32,
    base_url: str | None = None,
    tokenizer_path: str | None = None,
)
```

**Similar to TinkerServiceHolder but uses Weaver SDK.**

### Key Differences from Tinker

| Feature | TinkerServiceHolder | WeaverServiceHolder |
|---------|---------------------|---------------------|
| **SDK** | Tinker (async) | Weaver (sync) |
| **Renderer** | tinker_cookbook.renderers | Built-in |
| **Tool Parsing** | Custom parsing | Built-in |
| **API Key** | Optional | Required (from WEAVER_API_KEY env) |

### Initialization Process

```python
# 1. Create service client
self._service_client = ServiceClient(
    base_url=base_url,
    api_key=os.getenv("WEAVER_API_KEY")
)
self._service_client.connect()

# 2. Create training client (Weaver model)
self._training_client = self._service_client.create_model(
    base_model=base_model,
    lora_config={"rank": lora_rank}
)

# 3. Get tokenizer from training client
self._tokenizer = self._training_client.get_tokenizer()

# 4. Save initial weights and create sampling client
sampling_path = self._training_client.save_weights_for_sampler(name="initial")
self._sampling_client = self._service_client.create_sampling_client(
    model_path=sampling_path,
    base_model=base_model
)
```

## Usage in NexRL Components

### In RemoteApiInferenceServiceClient

```python
class RemoteApiInferenceServiceClient(InferenceServiceClient):
    def __init__(self, config):
        self._service_holder = None  # Set later

    def set_service_holder(self, service_holder):
        """Set the service holder (Tinker or Weaver)."""
        self._service_holder = service_holder

    def completion(self, prompt: str, **kwargs) -> dict:
        """Call inference via service holder."""
        result = self._service_holder.sample_from_prompt(
            prompt=prompt,
            max_tokens=self._config.inference_service.max_tokens,
            temperature=self._config.temperature,
        )
        return self._format_as_openai_response(result)
```

### In RemoteApiTrainer

```python
class RemoteApiTrainer(BaseTrainer):
    def set_service_holder(self, service_holder):
        """Set the service holder for training."""
        self._service_holder = service_holder

    def train(self, trajectories):
        """Execute training step via service holder."""
        # Prepare trajectories
        trajectories = self._prepare_trajectories(trajectories, metrics)

        # Convert to datums
        datums = convert_trajectories_to_datums(trajectories)

        # Train via service holder
        result = self._service_holder.train(datums, config=train_config)

        return result
```

### In Controller Initialization

```python
# Controller creates and distributes service holder
if backend == "tinker":
    service_holder = TinkerServiceHolder(
        base_model=config.model,
        lora_rank=config.lora_rank,
        base_url=config.base_url
    )
elif backend == "weaver":
    service_holder = WeaverServiceHolder(
        base_model=config.model,
        lora_rank=config.lora_rank,
        base_url=config.base_url
    )

# Set service holder for inference and training
inference_client.set_service_holder(service_holder)
trainer.set_service_holder(service_holder)
```

## Configuration

### Tinker Configuration

```yaml
inference_service:
  backend: "tinker"
  base_url: "http://tinker-service:8000"

train_service:
  backend: "tinker"

algorithm:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  lora_rank: 32
  renderer_name: null  # Auto-detect
```

### Weaver Configuration

```yaml
inference_service:
  backend: "weaver"
  base_url: "http://weaver-service:8000"

train_service:
  backend: "weaver"

algorithm:
  base_model: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 32

# Set WEAVER_API_KEY environment variable
```

## Tool Call Handling

### Tinker Tool Call Parsing

```python
def _parse_and_normalize_tool_call(self, tool_call_str: str) -> list[dict] | None:
    """Parse tool call JSON and normalize to OpenAI format."""
    import json

    tool_call = json.loads(tool_call_str)

    # Normalize to OpenAI format
    normalized = [{
        "id": tool_call.get("id", f"tinker-tool-call-{uuid.uuid4().hex}"),
        "type": "function",
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["arguments"])
        }
    }]

    return normalized
```

### Weaver Tool Call Handling

Weaver has built-in tool call parsing, so the service holder uses it directly.

## Best Practices

### 1. Single Instance Per Backend

Create one service holder instance per backend and reuse it:

```python
# Good: Single instance shared across components
service_holder = TinkerServiceHolder(...)
inference_client.set_service_holder(service_holder)
trainer.set_service_holder(service_holder)

# Bad: Multiple instances (wastes resources)
holder1 = TinkerServiceHolder(...)
holder2 = TinkerServiceHolder(...)
```

### 2. Weight Synchronization

Always update sampling weights after training:

```python
# After training step
weight_path = service_holder.update_sampling_weights()
# Now inference uses updated weights
```

### 3. Error Handling

Handle service errors gracefully:

```python
try:
    result = service_holder.sample_from_messages(messages, ...)
except Exception as e:
    logger.error(f"Sampling failed: {e}")
    # Retry or fallback
```

## Related Documentation

- [Inference Service](./inference-service.md) - Inference service clients
- [Training Service](./training-service.md) - Training service clients
- [Remote API Trainers](../06-trainers/remote-api-trainers.md) - Trainers using service holders
- [Weight Synchronization](../08-features/weight-synchronization.md) - Weight sync coordination
