# Data Loader

The Data Loader provides input data for rollout workers during training and validation. NexRL includes built-in data loaders and an abstract interface for custom implementations.

## Overview

**Purpose**: Manage dataset loading, shuffling, and distribution to rollout workers

**Key Features**:
- Sequential and random access patterns
- Data replay for re-rollout scenarios
- Validation and training data separation
- Threadsafe operation
- Configurable shuffling and batching

## BaseDataLoader

**Location**: `nexrl/data_loader/data_loader.py`

**Base class** for all data loaders providing the core interface.

### Constructor

```python
def __init__(self, config: DictConfig, is_validate: bool = False)
```

**Parameters**:
- `config`: Configuration for the data loader
- `is_validate`: Whether this dataloader is for validation (affects behavior and tracking)

**Attributes**:
- `_weight_sync_controller`: Reference to weight sync controller (set via `set_module_references()`)
- `_is_validate`: Flag indicating validation mode

### Abstract Methods

Subclasses must implement these methods:

#### `get_next_item() -> dict[str, Any] | None`

Get the next data item in sequence.

```python
item = dataloader.get_next_item()
if item is not None:
    process(item)
```

**Returns**: Next data item as dictionary, or `None` if exhausted

#### `is_finished() -> bool`

Check if all data has been consumed.

```python
if dataloader.is_finished():
    print("All data processed")
```

**Returns**: `True` if no more data available

#### `can_return_item() -> bool`

Check if dataloader can currently return an item.

```python
if dataloader.can_return_item():
    item = dataloader.get_next_item()
```

**Returns**: `True` if item is available now

**Usage**: Rollout workers check this before calling `get_next_item()`

#### `reset()`

Reset the dataloader to initial state.

```python
dataloader.reset()
# Can iterate through data again
```

**Usage**: Called between validation cycles

#### `add_item_front(item: dict[str, Any])`

Add item to beginning of queue (returned last by iteration).

```python
dataloader.add_item_front(item)
```

**Parameters**:
- `item`: Data item to add to front

**Usage**: Re-rollout scenarios when weight sync interrupts processing

#### `add_item_back(item: dict[str, Any])`

Add item to end of queue (returned next by iteration).

```python
dataloader.add_item_back(item)
```

**Parameters**:
- `item`: Data item to add to back

### Convenience Methods

#### `add_item(item: dict[str, Any])`

Add item (defaults to `add_item_back`).

```python
dataloader.add_item(item)  # Adds to back
```

### Module References

#### `set_module_references(weight_sync_controller)`

Set references to other modules.

```python
dataloader.set_module_references(
    weight_sync_controller=weight_sync_controller
)
```

**Parameters**:
- `weight_sync_controller`: Weight synchronization controller reference

## TorchDataLoader

**Location**: `nexrl/data_loader/torch_data_loader.py`

**Concrete implementation** using PyTorch DataLoader for file-based datasets.

### Features

- Loads data from Parquet files
- Supports shuffling with configurable seed
- Threadsafe with deque-based buffer
- Configurable max prompt length filtering
- Automatic data replenishment

### Constructor

```python
def __init__(self, config: DictConfig, is_validate: bool = False)
```

**Configuration**:

```yaml
data:
  type: "torch"
  data_files:
    - "/path/to/data1.parquet"
    - "/path/to/data2.parquet"
  shuffle: true
  seed: 42
  max_prompt_length: 16384  # Filter out long prompts
  batch_size: 1  # Internal batching for PyTorch DataLoader
  num_workers: 4  # PyTorch DataLoader workers
```

### Implementation Details

#### Data Structure

- Uses `collections.deque` for thread-safe FIFO buffer
- Preloads data from files into buffer
- Automatically refills buffer when low

#### Shuffling

- Shuffles at file level and sample level
- Uses configured seed for reproducibility
- Separate shuffle for each epoch

#### Filtering

- Filters samples exceeding `max_prompt_length`
- Logs filtered samples count
- Ensures prompts fit in model context window

### Methods

```python
def get_next_item(self) -> dict[str, Any] | None:
    """Get next item from buffer"""
    if len(self._data_buffer) > 0:
        return self._data_buffer.popleft()
    return None

def is_finished(self) -> bool:
    """Check if buffer is empty and no more files"""
    return len(self._data_buffer) == 0 and self._all_files_loaded

def can_return_item(self) -> bool:
    """Check if buffer has items"""
    return len(self._data_buffer) > 0

def reset(self):
    """Reset for validation cycles"""
    self._data_buffer.clear()
    self._all_files_loaded = False
    # Reload data from files
```

## Mock DataLoader

**Location**: `nexrl/mock/mock_data_loader.py`

**Testing implementation** with generated mock data.

### Features

- Generates fake data on-the-fly
- Configurable number of samples
- No external file dependencies
- Useful for testing and development

### Configuration

```yaml
data:
  type: "mock"
  total_task_size: 1000  # Number of mock samples
```

### Implementation

```python
def get_next_item(self) -> dict[str, Any] | None:
    if self._current_index < self._total_task_size:
        item = {
            "prompt": f"Mock prompt {self._current_index}",
            "answer": f"Mock answer {self._current_index}",
            "id": self._current_index,
        }
        self._current_index += 1
        return item
    return None
```

## Usage Patterns

### Basic Data Loading

```python
# In rollout worker
class MyRolloutWorker(BaseRolloutWorker):
    def _get_rollout_task(self) -> dict[str, Any] | None:
        # Check if data available
        if not execute(self._dataloader.can_return_item):
            time.sleep(0.1)
            return None

        # Get next task
        task = execute(self._dataloader.get_next_item)
        return task
```

### Re-Rollout Pattern

```python
# When weight sync interrupts processing
def step(self, task: dict[str, Any]) -> str | None:
    result = self._put_trajectory(trajectory)

    if result == "re-rollout":
        # Return task to dataloader
        execute(self._dataloader.add_item_front, task)
        return "re-rollout"

    return result
```

### Validation Cycle

```python
# Controller orchestrates validation
def _run_validate(self, model_tag):
    # Validation dataloader resets automatically
    execute(self.validate_dataloader.reset)

    # Switch workers to validation mode
    self._start_validate(model_tag)

    # Wait for validation to complete
    while not execute(self.validator.is_complete):
        time.sleep(1.0)

    # Switch back to training
    self._end_validate(model_tag)
```

## Custom Data Loader Implementation

### Minimal Implementation

```python
from nexrl.data_loader import BaseDataLoader
from omegaconf import DictConfig

class MyDataLoader(BaseDataLoader):
    """Custom data loader for my task"""

    def __init__(self, config: DictConfig, is_validate: bool = False):
        super().__init__(config, is_validate)
        # Load your data
        self._data = self._load_data(config.data_files)
        self._index = 0

    def _load_data(self, files):
        # Your data loading logic
        data = []
        for file in files:
            # Load from file
            pass
        return data

    def get_next_item(self) -> dict[str, Any] | None:
        if self._index < len(self._data):
            item = self._data[self._index]
            self._index += 1
            return item
        return None

    def is_finished(self) -> bool:
        return self._index >= len(self._data)

    def can_return_item(self) -> bool:
        return self._index < len(self._data)

    def reset(self):
        self._index = 0

    def add_item_front(self, item):
        self._data.insert(self._index, item)

    def add_item_back(self, item):
        self._data.append(item)
```

### Register Custom DataLoader

```python
# In controller.py MODULE_REGISTRY
MODULE_REGISTRY = {
    NexRLRole.DATA_LOADER: {
        "mock": MockDataLoader,
        "torch": TorchDataLoader,
        "my_custom": MyDataLoader,  # Add here
    },
    # ...
}
```

### Configuration

```yaml
data:
  type: "my_custom"
  data_files:
    - "/path/to/my/data.custom"
  # Your custom parameters
```

## Configuration Reference

### Common Parameters

```yaml
data:
  type: "torch"  # or "mock", "my_custom"
  seed: 42  # Random seed for shuffling
```

### TorchDataLoader Specific

```yaml
data:
  type: "torch"
  data_files:  # List of data files
    - "/path/to/file1.parquet"
    - "/path/to/file2.parquet"
  shuffle: true  # Shuffle data
  max_prompt_length: 16384  # Filter long prompts
  batch_size: 1  # PyTorch DataLoader internal batch
  num_workers: 4  # PyTorch DataLoader workers
```

### MockDataLoader Specific

```yaml
data:
  type: "mock"
  total_task_size: 1000  # Number of mock samples
```

## Best Practices

### Data Organization

1. **Use Parquet format** for efficient storage and loading
2. **Split large datasets** into multiple files for parallel loading
3. **Include all required fields** (prompt, answer, id, etc.)
4. **Validate data format** before training

### Configuration

1. **Set appropriate max_prompt_length** to filter oversized prompts
2. **Use consistent seeds** for reproducible experiments
3. **Enable shuffling** for better training dynamics
4. **Configure num_workers** based on CPU availability

### Performance

1. **Monitor buffer size** (avoid excessive memory use)
2. **Balance file count** vs file size
3. **Use SSD/fast storage** for data files
4. **Preprocess data** offline when possible

### Error Handling

1. **Validate file paths** before training starts
2. **Handle missing files** gracefully
3. **Check data format** compatibility
4. **Log loading errors** clearly

## Troubleshooting

### Issue: "No data items available"

**Symptoms**: Workers idle, no tasks being processed

**Solutions**:
1. Check `data_files` paths are correct
2. Verify files are not empty
3. Check `max_prompt_length` filter isn't too restrictive
4. Ensure file format is compatible

### Issue: "Data loading is slow"

**Symptoms**: Low rollout worker utilization

**Solutions**:
1. Increase `num_workers` in DataLoader
2. Use faster storage (SSD, parallel filesystem)
3. Reduce file size through compression
4. Preload data into buffer

### Issue: "Out of memory"

**Symptoms**: System crashes during data loading

**Solutions**:
1. Reduce `batch_size` for PyTorch DataLoader
2. Process smaller files
3. Reduce `num_workers`
4. Filter samples by length more aggressively

## Next Steps

- Understand [Trajectory Pool](../04-trajectory-pool/trajectory-pool.md) for trajectory batching
- Learn about [Rollout Workers](../05-rollout-workers/overview.md) that consume data
- Explore [Configuration Reference](../12-configuration-reference/data-config.md) for all options
- Review [Best Practices](../13-best-practices/module-development.md) for custom loaders
