# Data Loader Configuration

Configuration options for data loading components.

## Data Loader Types

### TorchDataLoader (`type: "torch"`)

PyTorch-based data loader for production use.

```yaml
data:
  type: "torch"
  seed: 42

  # Data files
  data_files:
    - "${oc.env:NEXRL_DATA_PATH}/data/train.parquet"
    - "${oc.env:NEXRL_DATA_PATH}/data/train2.parquet"

  # Batch configuration
  batch_size: 32
  keep_batch_order: true
  rollout_repeat_n: 8

  # Data format
  prompt_key: "prompt"
  filter_prompts: false
  max_prompt_length: 4096
  max_response_length: 8192

  # Tokenizer
  tokenizer_path: "${oc.env:NEXRL_MODEL_PATH}/model"

  # Dataset options
  shuffle: true
  drop_last: true
```

### MockDataLoader (`type: "mock"`)

Mock data loader for testing.

```yaml
data:
  type: "mock"
  seed: 42
  mock_data_size: 1000
```

## Configuration Options

### type

**Type:** `string`
**Options:** `"torch"`, `"mock"`
**Default:** Required

Data loader implementation to use.

### seed

**Type:** `int`
**Default:** `42`

Random seed for data loading and shuffling.

### data_files

**Type:** `list[string]`
**Default:** Required (torch only)

List of data file paths. Supports:
- Parquet files (`.parquet`)
- Multiple files (concatenated)
- Environment variable interpolation

**Example:**
```yaml
data_files:
  - "${oc.env:NEXRL_DATA_PATH}/train1.parquet"
  - "${oc.env:NEXRL_DATA_PATH}/train2.parquet"
```

### batch_size

**Type:** `int`
**Default:** Required

Number of data items per batch.

**Notes:**
- Affects data loader batch retrieval
- Independent from trajectory pool batch size
- Used for `keep_batch_order` logic

### keep_batch_order

**Type:** `bool`
**Default:** `true`

Whether to maintain batch grouping.

**When `true`:**
- Data items from same batch stay together
- Workers process batch items as a unit
- Used with `rollout_repeat_n`

**When `false`:**
- Data items distributed independently
- No batch structure maintained

### rollout_repeat_n

**Type:** `int`
**Default:** `1`

Number of rollouts per data item (prompt).

**Example:**
```yaml
rollout_repeat_n: 8  # Generate 8 responses per prompt
```

**Use Cases:**
- GRPO: Multiple samples per prompt for group ranking
- Exploration: Sample diverse responses
- Evaluation: Multiple attempts per problem

### prompt_key

**Type:** `string`
**Default:** `"prompt"`

Column name containing prompts in data files.

**Example Data:**
```python
{
  "prompt": "Solve: 2 + 2 = ?",
  "answer": "4",
  "difficulty": "easy"
}
```

### filter_prompts

**Type:** `bool`
**Default:** `false`

Filter prompts exceeding `max_prompt_length`.

**When `true`:**
- Prompts longer than `max_prompt_length` are skipped
- Prevents tokenization errors
- May reduce dataset size

**When `false`:**
- All prompts loaded
- Long prompts may cause errors

### max_prompt_length

**Type:** `int`
**Default:** `4096`

Maximum prompt length in tokens.

**Used for:**
- Prompt filtering (if `filter_prompts=true`)
- Memory allocation hints
- Validation bounds

### max_response_length

**Type:** `int`
**Default:** `8192`

Maximum response length in tokens.

**Used for:**
- LLM generation limits
- Memory allocation
- Trajectory validation

### tokenizer_path

**Type:** `string`
**Default:** Required (torch only)

Path to tokenizer model.

**Example:**
```yaml
tokenizer_path: "${oc.env:NEXRL_MODEL_PATH}/Qwen/Qwen3-8B"
```

**Supports:**
- HuggingFace tokenizers
- Local tokenizer files
- Environment variable paths

### shuffle

**Type:** `bool`
**Default:** `true`

Shuffle data at epoch boundaries.

**When `true`:**
- Data order randomized each epoch
- Improves training diversity

**When `false`:**
- Fixed data order
- Deterministic iteration

### drop_last

**Type:** `bool`
**Default:** `true`

Drop incomplete final batch.

**When `true`:**
- Last batch dropped if smaller than `batch_size`
- Ensures consistent batch sizes

**When `false`:**
- All data used, including partial batch

## Validation Data Configuration

Validation data uses same structure under `validate.data`:

```yaml
validate:
  validate_before_train: true

  data:
    type: "torch"
    seed: ${data.seed}  # Can reference main data config
    data_files:
      - "${oc.env:NEXRL_DATA_PATH}/test.parquet"
    batch_size: 16
    prompt_key: "prompt"
    filter_prompts: false
    max_prompt_length: ${data.max_prompt_length}
    tokenizer_path: ${data.tokenizer_path}
    shuffle: true
    drop_last: false  # Use all validation data
```

## Common Patterns

### Small Dataset

```yaml
data:
  type: "torch"
  data_files:
    - "data/small_train.parquet"
  batch_size: 8
  rollout_repeat_n: 4
  shuffle: true
  drop_last: true
```

### Large Dataset

```yaml
data:
  type: "torch"
  data_files:
    - "${NEXRL_DATA_PATH}/shard_*.parquet"  # Multiple files
  batch_size: 128
  rollout_repeat_n: 8
  filter_prompts: true  # Filter long prompts
  shuffle: true
  drop_last: true
```

### Long Context Task

```yaml
data:
  type: "torch"
  max_prompt_length: 32768  # Extended context
  max_response_length: 8192
  filter_prompts: true  # Important for long context
  tokenizer_path: "${NEXRL_MODEL_PATH}/LongContextModel"
```

### Debug Mode

```yaml
data:
  type: "mock"
  seed: 42
  mock_data_size: 100  # Small for fast iteration
```

## Environment Variable Interpolation

Use Hydra's `oc.env` syntax for environment variables:

```yaml
data:
  # With default value
  data_files:
    - "${oc.env:NEXRL_DATA_PATH}/train.parquet"

  # With fallback
  tokenizer_path: "${oc.env:TOKENIZER_PATH,${oc.env:NEXRL_MODEL_PATH}/default}"
```

## Troubleshooting

### Data Not Found

**Symptom:** `FileNotFoundError` for data files

**Solutions:**
- Verify `NEXRL_DATA_PATH` environment variable
- Check file paths are correct
- Ensure files exist before training

### Out of Memory

**Symptom:** OOM during data loading

**Solutions:**
- Reduce `batch_size`
- Enable `filter_prompts` for long contexts
- Reduce `max_prompt_length`/`max_response_length`

### Inconsistent Batch Sizes

**Symptom:** Batch size varies across iterations

**Solutions:**
- Enable `drop_last=true` for consistent batches
- Check data file size vs batch_size
- Verify `keep_batch_order` setting

## Related Documentation

- [Data Loader](../03-data-loader/data-loader.md) - Data loader implementation
- [Complete Config](./complete-config.md) - Full configuration example
- [Configuration Setup](../01-getting-started/configuration-setup.md) - Environment variables
