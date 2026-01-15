# NexRL Developer Guide

## Overview

NexRL is a large-scale distributed reinforcement learning training framework designed for modern RL applications. NexRL provides a scalable, modular architecture that seamlessly supports various training and inference backends.

### Key Features

- **Multiple Launch Mode Support**: Seamlessly runs in both local and Ray distributed modes
- **Modular Design**: Clean separation of concerns with well-defined interfaces and extensible components
- **Training-as-a-Service & Rollout-as-a-Service**: Unified API architecture that seamlessly supports different training and inference frameworks through service abstraction
- **Resource Management**: Intelligent placement and co-location of services for optimal performance
- **Activity Tracking**: Comprehensive monitoring and health checking system for production deployments
- **Error Handling**: Centralized error reporting and recovery mechanisms

## Architecture Overview

NexRL follows a modular architecture where components communicate through explicit interfaces and APIs.

![NexRL Architecture](./imgs/nexrl_architecture.png)


### Core Components

1. **NexRLController**: Main orchestrator that initializes and coordinates all components
2. **DataLoader**: Provides input data for rollout workers (training and validation)
3. **RolloutWorkers**: Execute environment interactions and generate trajectories
4. **TrajectoryPool**: Collects and batches trajectories from rollout workers
5. **Trainer**: Processes trajectories and executes training (with integrated algorithm logic)
6. **WeightSyncController**: Manages model weights and synchronization
7. **Validator**: Collects validation trajectories and computes metrics
8. **ActivityTracker**: Monitors system health and activity, coordinates experiment logging
9. **RayResourceManager**: Handles distributed resource allocation and actor co-location

## Core Data Types

### nexrl_types.py

Core type definitions used throughout the framework.

#### ModelTag
```python
ModelTag = str  # Type alias for model identification
```

Used to identify different models within the system.

### Evaluation System (nexrl.rollout_worker.base_nexau_rollout_worker)

The evaluation system is integrated into the NexAU rollout worker module.

#### BaseEvaluationTarget
```python
@dataclass
class BaseEvaluationTarget:
    final_answer: str
```

Base class for evaluation targets containing the agent's final answer.

#### NexAUEvaluationTarget
```python
@dataclass
class NexAUEvaluationTarget(BaseEvaluationTarget):
    final_answer: str
    observation: list[dict[str, Any]]  # Complete execution trajectory
```

Extended evaluation target for NexAU agents that includes the full execution trace with intermediate steps and observations.

#### EvaluationRunResult
```python
@dataclass
class EvaluationRunResult:
    reward: float = 0.0                         # Primary RL reward signal
    ground_truth: str = ""                      # Reference answer
    metrics: dict[str, float] = field(default_factory=dict)  # Additional scalar metrics
    extra_info: dict[str, Any] = field(default_factory=dict) # Extra information (any type)
```

Contains evaluation results including:
- `reward`: Primary training signal (should be in [0, 1] range)
- `ground_truth`: Expected/reference answer for comparison
- `metrics`: Additional metrics for logging (must be scalar floats)
- `extra_info`: Any additional information (can be any type)

#### Evaluator
```python
class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        data: dict[str, Any],
        evaluation_target: BaseEvaluationTarget,
    ) -> EvaluationRunResult:
        """Evaluate agent output against data."""
```

Abstract base class for task-specific evaluators. Subclasses must implement the `evaluate` method to define custom evaluation logic.

**Location:** These classes are defined in `nexrl/rollout_worker/base_nexau_rollout_worker.py` and exported via `nexrl/rollout_worker/__init__.py` for easy importing in recipe evaluators.

**Usage in Recipes:**
```python
from nexrl.rollout_worker import (
    Evaluator,
    BaseEvaluationTarget,
    NexAUEvaluationTarget,
    EvaluationRunResult
)

class MyTaskEvaluator(Evaluator):
    def evaluate(self, data, evaluation_target):
        # Implementation
        return EvaluationRunResult(reward=1.0)
```

#### Trajectory
```python
Trajectory = dict[str, Any]
```

Represents a single trajectory containing environment interaction data. Common keys include:
- `prompt`: Input prompt for LLM
- `response`: LLM response
- `finish_reason`: Completion status
- `model_tag`: Associated model identifier

#### Batch
```python
@dataclass
class Batch:
    values: dict[str, Any]      # Tensor or data arrays, length = metadata['batch_size']
    metadata: dict[str, Any]    # Batch metadata including 'batch_size'
```

**Methods:**
- `__len__() -> int`: Returns batch size from metadata
- `copy() -> Batch`: Creates a deep copy of the batch
- `to_dict() -> dict[str, Any]`: Converts batch to single dictionary (metadata keys overwrite values keys)
- `remove_redundant_left_padding(data, pad_token_id, fields, anchor_field, max_strip_threshold) -> Batch`: Static method that removes redundant left padding tokens common across all sequences
- `remove_redundant_right_padding(data, pad_token_id, fields, anchor_field, max_strip_threshold) -> Batch`: Static method that removes redundant right padding tokens common across all sequences
- `to_nextrainer_batch() -> dict[str, Any]`: Converts batch to NexTrainer format with separated tensor/non-tensor values and metadata

#### NexRLRole
```python
class NexRLRole(Enum):
    ROLLOUT_WORKER = "rollout_worker"
    TRAINER = "trainer"
    TRAJECTORY_POOL = "trajectory_pool"
    WEIGHT_SYNC_CONTROLLER = "weight_sync_controller"
    DATA_LOADER = "data_loader"
    VALIDATE_DATALOADER = "validate_dataloader"
    VALIDATOR = "validator"
```

Defines different component roles for resource pool mapping and module registration.

## Core Module Classes

### NexRLModule

Base class for all NexRL components, enabling Ray colocation compatibility.

```python
class NexRLModule(ABC):
    def __init__(self):
        self._module_name: str = "invalid"
        self._activity_tracker: ActivityTrackerProxy = None
```

**Purpose**: Provides common interface for all NexRL modules to work with Ray resource management and activity tracking.

**Methods:**
- `set_activity_tracker(tracker: ActivityTrackerProxy)`: Sets the activity tracker for this module
- `set_module_name(module_name: str)`: Sets the name of this module
- `get_module_name() -> str`: Gets the name of this module
- `health_check() -> bool`: Health check method to verify the module is alive and responsive, used during initialization and monitoring
- `easy_dump(value, keys, value_formatter)`: Convenience method to dump values with automatic module context for debugging purposes

## Main Controller

### NexRLController

The main orchestrator responsible for initializing, coordinating, and monitoring all framework components.

#### Constructor
```python
def __init__(self, config: DictConfig)
```

**Parameters:**
- `config`: Hydra configuration containing all module settings

**Functionality:**
- Initializes all framework modules based on launch mode
- Sets up Ray resources in distributed mode
- Establishes inter-module references
- Dynamically loads custom rollout workers from recipe directories

#### Module Registry and Dynamic Loading

The controller maintains a registry of module types for each role:

```python
MODULE_REGISTRY = {
    NexRLRole.ROLLOUT_WORKER: {
        "mock": MockRolloutWorker,
        "simple": SimpleRolloutWorker,
        "agent": AgentRolloutWorker,
        "nexau": DefaultNexAURolloutWorker,  # Default NexAU worker
        "pig_latin": PigLatinRolloutWorker,
        # Custom workers loaded dynamically
    },
    NexRLRole.TRAINER: {
        "self_hosted": SelfHostedTrainer,
        "self_hosted_grpo": SelfHostedGrpoTrainer,
        "remote_api_grpo": RemoteApiGrpoTrainer,
        "remote_api_cross_entropy": RemoteApiCrossEntropyTrainer,
        # ...
    },
    # ... other roles
}
```

**Dynamic Worker Loading:**

For rollout workers, the controller supports loading custom implementations from recipe directories:

1. **Configuration Check**: If `custom_rollout_worker_module_path` and `custom_rollout_worker_class_name` are specified in the config, loads custom worker
2. **Module Import**: Resolves the path (relative to NEXRL_PATH or absolute) and dynamically imports the module
3. **Class Extraction**: Retrieves the specified class from the module
4. **Fallback**: If custom worker not specified, uses registered worker type from registry

**Method: _get_module_class_for_role**

```python
def _get_module_class_for_role(self, role: NexRLRole, config: DictConfig) -> type
```

**Parameters:**
- `role`: NexRL role to get module class for
- `config`: Full configuration (includes custom worker paths for rollout workers)

**Returns:** Module class to instantiate

**Process:**
1. For rollout workers with custom worker config, loads from specified path
2. Otherwise, looks up type from config and retrieves from MODULE_REGISTRY
3. Validates that the returned class is valid for the role

**Path Resolution:**
- Relative paths are resolved relative to `NEXRL_PATH` environment variable
- Absolute paths are used as-is
- Module is imported using `importlib` machinery
- Class is extracted using `getattr` on the loaded module

**Benefits:**
- Recipes are self-contained with their custom logic
- No need to modify core framework code for new tasks
- Easy to version control and share task-specific implementations
- Supports both absolute and relative path specifications

#### Core Methods

##### run()
```python
def run() -> None
```

Starts the training process by launching all components and entering monitoring loop.

**Process:**
1. Initializes train workers with final configuration
2. Loads initial checkpoint (or resumes from existing)
3. Optionally runs validation before training
4. Starts all worker components asynchronously
5. Monitors system health and activity
6. Checks for weight sync validation triggers
7. Checks for completion conditions
8. Handles graceful shutdown

##### stop()
```python
def _stop()
```

Gracefully stops all components and waits for activity completion.

**Features:**
- Signals all workers to stop
- Waits for quiescence with timeout
- Logs remaining activities on timeout

#### Internal Methods

##### _check_finish() -> bool
```python
def _check_finish() -> bool
```

Determines if training should stop based on:
- Maximum training steps reached
- System quiescence (all pools empty, no active work)

##### _check_module_liveness(timeout: float = 5.0) -> bool
```python
def _check_module_liveness(self, timeout: float = 5.0) -> bool
```

**Parameters:**
- `timeout`: Ray operation timeout in seconds

**Returns:** True if all modules are alive, False if any are dead

##### _check_module_exceptions() -> bool
```python
def _check_module_exceptions(self) -> bool
```

**Returns:** True if system is healthy, False if critical errors detected

##### _load_initial_checkpoint()
```python
def _load_initial_checkpoint()
```

Loads initial checkpoint or prepares for training from scratch. Creates sync weight buffer and performs initial weight sync to inference service.

##### _load_resume_checkpoint()
```python
def _load_resume_checkpoint()
```

Loads checkpoint based on resume configuration (auto or from_path). Supports automatic detection of latest checkpoint or explicit path specification.

##### _find_latest_checkpoint(checkpoint_folder: str) -> str | None
```python
def _find_latest_checkpoint(self, checkpoint_folder: str) -> str | None
```

Finds the latest checkpoint in the given folder by parsing `global_step_*` directories.

##### _run_validate(model_tag: ModelTag)
```python
def _run_validate(self, model_tag: ModelTag)
```

Runs validation cycle after a weight sync event. Switches workers to validation mode, waits for completion, computes metrics, and switches back to training mode.

##### _start_validate(model_tag: ModelTag)
```python
def _start_validate(self, model_tag: ModelTag)
```

Starts validation by switching rollout workers to validation mode.

##### _end_validate(model_tag: ModelTag)
```python
def _end_validate(self, model_tag: ModelTag)
```

Ends validation by computing metrics, logging results, switching workers back to training mode, and notifying weight sync controller.

## Data Loading

### BaseDataLoader

Abstract base class for data input components.

#### Constructor
```python
def __init__(self, config: DictConfig, is_validate: bool = False)
```

**Parameters:**
- `config`: Configuration for the data loader
- `is_validate`: Whether this dataloader is for validation (affects behavior and tracking)

#### Abstract Methods

##### __len__() -> int
```python
def __len__(self) -> int
```

**Returns:** Number of remaining data items

##### __getitem__(index: int) -> dict[str, Any]
```python
def __getitem__(self, index: int) -> dict[str, Any]
```

**Parameters:**
- `index`: Index of data item to retrieve

**Returns:** Single data item as dictionary

##### get_next_item() -> dict[str, Any] | None
```python
def get_next_item(self) -> dict[str, Any] | None
```

**Returns:** Next data item in sequence, or None if exhausted

##### is_finished() -> bool
```python
def is_finished(self) -> bool
```

**Returns:** True if no more data available

##### can_return_item() -> bool
```python
def can_return_item(self) -> bool
```

**Returns:** True if the data loader can return an item currently

##### reset()
```python
def reset() -> None
```

Resets the data loader to initial state (used for validation cycles)

#### Data Management Methods

##### add_item(item: dict[str, Any])
```python
def add_item(self, item: dict[str, Any]) -> None
```

**Parameters:**
- `item`: Data item to add (added to end by default)

##### add_item_front(item: dict[str, Any])
```python
def add_item_front(self, item: dict[str, Any]) -> None
```

**Parameters:**
- `item`: Data item to add to beginning of queue

##### add_item_back(item: dict[str, Any])
```python
def add_item_back(self, item: dict[str, Any]) -> None
```

**Parameters:**
- `item`: Data item to add to end of queue

## LLM Service Integration

### LLMServiceClient

Service client for interacting with LLM APIs, encapsulating OpenAI client functionality.

#### Constructor
```python
def __init__(self, config: DictConfig)
```

**Parameters:**
- `config`: Configuration containing LLM settings

**Initializes:**
- OpenAI client with API key and base URL
- Model tag and weight sync coordination settings

#### Methods

##### completion(prompt: str, **kwargs) -> dict[str, Any]
```python
def completion(self, prompt: str, **kwargs) -> dict[str, Any]
```

**Parameters:**
- `prompt`: Input text prompt
- `**kwargs`: Additional completion parameters

**Returns:** Dictionary containing:
- `prompt`: Original input prompt
- `response`: LLM generated text
- `finish_reason`: Completion status
- Additional passed kwargs

**Features:**
- Automatic retry logic with configurable max_retries
- Weight sync coordination (blocks if weight sync in progress)
- Error handling and logging

##### generate(messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]
```python
def generate(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]
```

**Parameters:**
- `messages`: List of message dictionaries for chat completion
- `**kwargs`: Additional generation parameters

**Returns:** Dictionary containing:
- `messages`: Original input messages
- `response`: Generated response text
- `tool_calls`: Any tool calls made
- `finish_reason`: Completion status
- Additional passed kwargs

##### set_weight_sync_controller(controller)
```python
def set_weight_sync_controller(self, controller: WeightSyncController)
```

**Parameters:**
- `controller`: Weight synchronization controller reference

## Rollout Workers

### BaseRolloutWorker

Abstract base class for rollout execution workers that interact with LLM services.

#### Constructor
```python
def __init__(self, config: DictConfig)
```

**Parameters:**
- `config`: Worker configuration including LLM settings

**Initializes:**
- LLMServiceClient for LLM interactions
- Threading components for async execution
- Module references (set via `set_module_references`)

#### Setup Methods

##### set_module_references(trajectory_pool, dataloader, weight_sync_controller, validate_dataloader, validator)
```python
def set_module_references(self, trajectory_pool: TrajectoryPool, dataloader: BaseDataLoader, weight_sync_controller: WeightSyncController, validate_dataloader: BaseDataLoader, validator: Validator)
```

**Parameters:**
- `trajectory_pool`: Reference to trajectory collection pool
- `dataloader`: Reference to data source
- `weight_sync_controller`: Reference to weight synchronization controller
- `validate_dataloader`: Reference to validation data source
- `validator`: Reference to validation trajectory collector

##### set_activity_tracker(tracker)
```python
def set_activity_tracker(self, tracker: ActivityTrackerProxy)
```

**Parameters:**
- `tracker`: Activity monitoring proxy

#### Execution Methods

##### run()
```python
def run()
```

Starts the worker thread and begins the main processing loop.

**Preconditions:**
- Module references must be set
- Activity tracker must be set

##### stop()
```python
def stop()
```

Gracefully stops the worker and waits for thread completion.

##### begin_validate()
```python
def begin_validate()
```

Switches the worker to validation mode. The worker will use the validation dataloader and send trajectories to the validator.

##### end_validate()
```python
def end_validate()
```

Switches the worker back to training mode. The worker will use the training dataloader and send trajectories to the trajectory pool.

##### step(task: dict[str, Any]) -> str | None
```python
def step(self, task: dict[str, Any]) -> str | None
```

**Parameters:**
- `task`: Single task to process

**Returns:**
- `"success"`: Trajectory processed and added successfully
- `"fail"`: Failed to process trajectory
- `"re-rollout"`: Should retry processing (weight sync in progress)
- `None`: Processing failed before trajectory creation

**Abstract method** - Must be implemented by derived classes to define specific worker behavior.

#### LLM Interface Methods

Workers access LLM functionality through the `_llm_client` (LLMServiceClient instance):

##### _llm_client.completion(prompt: str, **kwargs) -> dict[str, Any]
```python
def _llm_client.completion(self, prompt: str, **kwargs) -> dict[str, Any]
```

**Parameters:**
- `prompt`: Input text prompt
- `**kwargs`: Additional completion parameters (model, max_tokens, temperature, etc.)

**Returns:** Dictionary containing:
- `prompt`: Original input prompt
- `response`: LLM generated text
- `finish_reason`: Completion status
- Additional passed kwargs

**Features:**
- Automatic retry logic with configurable max_retries
- Weight sync coordination (blocks during sync)
- Error handling and logging

##### _llm_client.generate(messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]
```python
def _llm_client.generate(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]
```

**Parameters:**
- `messages`: List of message dictionaries for chat completion
- `**kwargs`: Additional generation parameters

**Returns:** Dictionary containing:
- `messages`: Original input messages
- `response`: Generated response text
- `tool_calls`: Any tool calls made
- `finish_reason`: Completion status
- Additional passed kwargs

#### Data Flow Methods

##### _put_trajectory(trajectory: Trajectory) -> str
```python
def _put_trajectory(self, trajectory: Trajectory) -> str
```

**Parameters:**
- `trajectory`: Completed trajectory to submit

**Returns:**
- `"success"`: Trajectory submitted successfully
- `"fail"`: Failed to submit trajectory
- `"re-rollout"`: Should retry (weight sync in progress)

##### _get_rollout_task() -> dict[str, Any] | None
```python
def _get_rollout_task(self) -> dict[str, Any] | None
```

**Returns:** Next task from dataloader, or None if none available

**Features:**
- Automatic sleep to prevent busy waiting
- Non-blocking operation

##### _put_rollout_task(task: dict[str, Any]) -> bool
```python
def _put_rollout_task(self, task: dict[str, Any]) -> bool
```

**Parameters:**
- `task`: Task to return to dataloader for reprocessing

**Returns:** True if successfully returned, False otherwise

### SimpleRolloutWorker

Concrete implementation of BaseRolloutWorker with basic LLM completion functionality.

#### Constructor
```python
def __init__(self, config: DictConfig)
```

Inherits from BaseRolloutWorker.

#### step(task: dict[str, Any]) -> str | None
```python
def step(self, task: dict[str, Any]) -> str | None
```

**Parameters:**
- `task`: Task dictionary containing `prompt` field

**Returns:**
- `"success"`: Trajectory processed and submitted successfully
- `"fail"`: Failed to submit trajectory
- `"re-rollout"`: Should retry processing
- `None`: Processing failed (missing prompt)

**Process:**
1. Extracts prompt from task
2. Calls LLMServiceClient completion
3. Creates trajectory with prompt, response, and task metadata
4. Submits trajectory to trajectory pool
5. Returns submission result

**Error Handling:**
- Returns None if prompt missing
- Propagates result from trajectory submission

### AgentRolloutWorker

Advanced rollout worker implementation for agent-based tasks with tool calling and multi-turn interaction support.

#### Constructor
```python
def __init__(self, config: DictConfig)
```

Inherits from BaseRolloutWorker and adds agent-specific functionality.

#### Key Features
- Supports chat-based interactions with message history
- Tool calling capabilities through LLM generate method
- Multi-turn conversation management
- Agent-specific trajectory formatting

This worker type is designed for more complex agent tasks that require stateful interactions and tool usage.

### BaseNexAURolloutWorker & DefaultNexAURolloutWorker

NexAU-based rollout worker for agent tasks using the NexAU agent framework. Provides comprehensive agent execution, trace processing, and evaluation capabilities.

#### Constructor
```python
def __init__(self, config: DictConfig)
```

**Parameters:**
- `config`: Worker configuration including:
  - `nexau_agent_config_path`: Path to NexAU agent configuration YAML
  - `evaluator_module_path`: Path to evaluator module in format "path/to/file.py:ClassName"
  - `nexau_agent_workspace`: Optional workspace path to add to sys.path for local imports
  - `task_name`: Task identifier for logging

**Initializes:**
- NexAU agent from configuration
- Task-specific evaluator
- Tokenizer for trajectory processing
- Agent workspace for local module imports

#### Key Features

##### Agent Integration
- Loads NexAU agent from YAML configuration
- Supports system prompts, tool bindings, and LLM config
- Automatic workspace path management for local imports
- Compatible with NexAU tracer system

##### Evaluation System
The worker includes an integrated evaluation system with the following components:

**Evaluation Target Classes:**
- `BaseEvaluationTarget`: Base class with `final_answer` field
- `NexAUEvaluationTarget`: Extends base with `observation` field containing execution trajectory

**Evaluation Result:**
- `EvaluationRunResult`: Contains `reward`, `ground_truth`, `metrics` dict, and `extra_info` dict
- Metrics must be scalar floats for aggregation
- Extra info can contain any additional data

**Evaluator Base Class:**
```python
class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, data: dict[str, Any], evaluation_target: BaseEvaluationTarget) -> EvaluationRunResult:
        """Evaluate agent output against data."""
```

##### Trace Processing
- Extracts trajectories from NexAU trace trees
- Processes LLM calls recursively through trace hierarchy
- Collects prompt messages, tools, responses, and token IDs
- Handles both `nexrl_train` response tokens and logprob extraction

##### Token Management
- Generates token sequences for training
- Creates loss masks based on response boundaries
- Supports customizable masking strategies via `get_train_loss_mask`
- Handles padding and sequence alignment

#### Default Implementation

`DefaultNexAURolloutWorker` is an alias for `BaseNexAURolloutWorker`, indicating it provides a complete, usable implementation.

The default `run_agent` method:
1. Formats task query (can be overridden via `format_task_query`)
2. Executes NexAU agent with tracer
3. Processes trace to extract trajectories
4. Runs evaluation to compute rewards
5. Generates tokens and loss masks
6. Creates final trajectory with all training data
7. Submits to trajectory pool

#### Customization Points

Subclasses can override specific methods:

##### format_task_query(data_item: dict[str, Any]) -> str
```python
def format_task_query(self, data_item: dict[str, Any]) -> str
```

**Parameters:**
- `data_item`: Raw data from dataloader

**Returns:** Formatted query string for agent

**Purpose:** Convert raw data into agent-compatible query format

##### get_train_loss_mask(trajectory_infos: list[dict]) -> list[bool]
```python
def get_train_loss_mask(self, trajectory_infos: list[dict]) -> list[bool]
```

**Parameters:**
- `trajectory_infos`: List of trajectory dictionaries from trace processing

**Returns:** Boolean mask indicating which tokens to include in loss computation

**Purpose:** Customize which parts of the response contribute to training loss

**Default behavior:** All response tokens contribute to loss (all True)

##### run_agent(task: dict[str, Any]) -> str | None
```python
def run_agent(self, task: dict[str, Any]) -> str | None
```

**Parameters:**
- `task`: Task dictionary from dataloader

**Returns:**
- `"success"`: Trajectory created and submitted
- `"fail"`: Submission failed
- `"re-rollout"`: Should retry (weight sync in progress)
- `None`: Processing failed

**Purpose:** Complete agent execution pipeline (rarely needs override)

#### Custom Rollout Worker Loading

NexRL supports loading custom rollout workers from recipe directories:

**Configuration:**
```yaml
rollout_worker:
  type: "nexau"  # Use base NexAU worker

  # Optional: load custom worker from recipe
  custom_rollout_worker_module_path: "recipe/my_task/agent_workspace/my_worker.py"
  custom_rollout_worker_class_name: "MyCustomWorker"

  # NexAU-specific config
  nexau_agent_config_path: "recipe/my_task/agent_workspace/agent_config.yaml"
  evaluator_module_path: "recipe/my_task/agent_workspace/evaluator.py:MyEvaluator"
  nexau_agent_workspace: "recipe/my_task/agent_workspace"
  task_name: "my_task"
```

**Loading Process:**
1. If `custom_rollout_worker_module_path` and `custom_rollout_worker_class_name` are specified, loads custom worker
2. Otherwise, uses registered worker type from controller registry
3. Custom workers are loaded dynamically at runtime
4. Supports both absolute and relative paths (relative to NEXRL_PATH)

#### Recipe Structure

NexAU-based tasks typically follow this recipe structure:

```
recipe/
└── my_task_name/
    ├── my_task_name.yaml          # Main recipe configuration
    ├── my_task_name.env.sh        # Environment setup script
    └── agent_workspace/           # Agent-specific files
        ├── agent_config.yaml      # NexAU agent configuration
        ├── evaluator.py          # Task-specific evaluator
        ├── my_worker.py          # Optional: custom worker
        └── custom_tools.py       # Optional: task-specific tools
```

**Agent Workspace Benefits:**
- Self-contained task configuration
- Easy to version control and share
- Local imports for task-specific modules
- Decoupled from core framework code

## Trajectory Management

### TrajectoryPool

Multi-store trajectory pool that manages separate TrajectoryPoolInstance objects for different models, providing flexible batching strategies and weight synchronization coordination.

#### Constructor
```python
def __init__(self, config: DictConfig)
```

**Parameters:**
- `config`: Pool configuration including grouping and batching settings

**Configuration Options:**
- `key_list`: List of keys for grouping trajectories
- `group_size`: Number of trajectories per group
- `batch_size`: Default batch size for retrieval
- `check_batch_ready_function`: Batch readiness criteria ("batch_size", "loaded_batch_finished")

#### Core Methods

##### put_trajectory(trajectory: Trajectory) -> str
```python
def put_trajectory(self, trajectory: Trajectory) -> str
```

**Parameters:**
- `trajectory`: Trajectory data to store

**Returns:**
- `"success"`: Trajectory stored successfully
- `"fail"`: Failed to store trajectory
- `"re-rollout"`: Should retry (weight sync in progress)

**Process:**
1. Extracts ModelTag from trajectory (defaults to "default")
2. Creates or retrieves appropriate TrajectoryPoolInstance
3. Adds trajectory to instance (may block during weight sync)

##### get_batch(batch_size: int | None = None, model_tag: ModelTag | None = None) -> Batch | None
```python
def get_batch(self, batch_size: int | None = None, model_tag: ModelTag | None = None) -> Batch | None
```

**Parameters:**
- `batch_size`: Number of trajectories to include
- `model_tag`: Specific model to get batch from

**Returns:** Batch of trajectories, or None if insufficient samples

**Behavior:**
- If `model_tag` is None, tries any available store
- If specified model_tag has no store, returns None

##### get_batch_any(batch_size: int | None = None) -> Batch | None
```python
def get_batch_any(self, batch_size: int | None = None) -> Batch | None
```

**Parameters:**
- `batch_size`: Number of trajectories to retrieve

**Returns:** Batch from any store with sufficient samples, or None

##### is_empty(model_tag: ModelTag | None = None) -> bool
```python
def is_empty(self, model_tag: ModelTag | None = None) -> bool
```

**Parameters:**
- `model_tag`: Specific model to check, or None for all models

**Returns:** True if specified store (or all stores) is empty

##### get_model_tags() -> list[ModelTag]
```python
def get_model_tags(self) -> list[ModelTag]
```

**Returns:** List of all ModelTags with active stores

### TrajectoryPoolInstance

Individual pool instance managing trajectories for a single model with weight synchronization coordination.

#### Methods

##### set_module_references(dataloader, weight_sync_controller, activity_tracker)
```python
def set_module_references(self, dataloader: BaseDataLoader, weight_sync_controller: WeightSyncController, activity_tracker: ActivityTrackerProxy)
```

**Parameters:**
- `dataloader`: Reference to data loader
- `weight_sync_controller`: Weight synchronization controller
- `activity_tracker`: Activity tracking proxy

##### put_trajectory(trajectory: Trajectory) -> str
```python
def put_trajectory(self, trajectory: Trajectory) -> str
```

**Parameters:**
- `trajectory`: Trajectory to add

**Returns:**
- `"success"`: Added successfully
- `"fail"`: Failed to add
- `"re-rollout"`: Weight sync in progress, should retry

##### notify_weight_sync_starting()
```python
def notify_weight_sync_starting()
```

Blocks new trajectory additions during weight synchronization.

##### unlock_for_weight_sync()
```python
def unlock_for_weight_sync()
```

Unblocks trajectory additions after weight synchronization completes.

### Trajectory Store Types

TrajectoryPoolInstance automatically creates appropriate stores based on configuration:

#### SimpleTrajectoryStore
- **Use Case**: No grouping required
- **Behavior**: Directly adds trajectories to finished samples
- **Configuration**: Empty `key_list`

#### GroupedTrajectoryStore
- **Use Case**: Single-level grouping (e.g., by user ID)
- **Behavior**: Groups trajectories by specified key, releases when group reaches target size
- **Configuration**: Single item in `key_list`

#### HierarchicalTrajectoryStore
- **Use Case**: Multi-level grouping (e.g., by user ID then session ID)
- **Behavior**: Creates nested hierarchy, releases leaf groups when complete
- **Configuration**: Multiple items in `key_list`

## Trainer Architecture

NexRL provides a hierarchical trainer architecture where algorithm-specific logic is integrated directly into trainer classes.

### Training Architecture Overview

#### Self-Hosted Training (NexTrainer Backend)

```
                    BaseTrainer
                        ↓
                SelfHostedTrainer
          (abstract _prepare_batch method)
                        ↓
            ┌──────────────────────┐
            ↓                      ↓
  SelfHostedGrpoTrainer    Custom Trainer
   (implements GRPO)      (custom algorithm)
```

**Pipeline:**
```
Trajectories → Process → _prepare_batch (algorithm) → Train Service → NexTrainer
```

- **`SelfHostedTrainer`**: Base class for self-hosted training
  - Handles trajectory processing (padding, tokenization)
  - Implements main training loop with checkpointing
  - Defines abstract `_prepare_batch(batch) -> (batch, metrics)` method
- **`SelfHostedGrpoTrainer`**: GRPO implementation
  - Overrides `_prepare_batch` with GRPO logic
  - Computes group-relative advantages
  - Applies KL penalty and computes metrics

#### Remote API Training (Tinker/Weaver Backend)

```
                     BaseTrainer
                         ↓
                 RemoteApiTrainer
        (abstract _prepare_trajectories method)
                         ↓
        ┌────────────────────────────────┐
        ↓                                ↓
RemoteApiGrpoTrainer          RemoteApiCrossEntropyTrainer
  (GRPO algorithm)              (supervised learning)
```

**Pipeline:**
```
Trajectories → _prepare_trajectories (algorithm) → Service Datums → Tinker/Weaver API
```

- **`RemoteApiTrainer`**: Base class for remote API training
  - Handles trajectory-to-datum conversion
  - Implements training loop with remote API calls
  - Defines abstract `_prepare_trajectories(trajectories, metrics) -> trajectories` method
- **`RemoteApiGrpoTrainer`**: GRPO for remote APIs
  - Overrides `_prepare_trajectories` with GRPO advantage computation
  - Groups by `run_id` and computes group-relative advantages
- **`RemoteApiCrossEntropyTrainer`**: Supervised learning for remote APIs
  - Implements cross-entropy loss preparation

### BaseTrainer

Abstract base class for all trainers.

#### Constructor
```python
def __init__(self, config: DictConfig)
```

**Parameters:**
- `config`: Configuration dictionary with training settings

#### Core Methods

##### train(trajectories: list[Trajectory]) -> dict
```python
@abstractmethod
def train(self, trajectories: list[Trajectory]) -> dict
```

**Parameters:**
- `trajectories`: List of trajectories to train on

**Returns:** Dictionary of training metrics

**Purpose:** Main training method that derived classes must implement. Should handle:
1. Trajectory processing
2. Algorithm-specific preparation
3. Training step execution
4. Metrics collection

##### initialize_workers() -> None
```python
def initialize_workers() -> None
```

Initialize backend training workers. Override in derived classes if needed.

##### run() -> None
```python
def run() -> None
```

Start the training loop in a background thread.

##### stop() -> None
```python
def stop() -> None
```

Stop the trainer gracefully.

### SelfHostedTrainer

Base trainer for self-hosted training backends (NexTrainer).

#### Architecture

```python
class SelfHostedTrainer(BaseTrainer):
    """Base trainer for self-hosted backends with extensible batch preparation."""

    @abstractmethod
    def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
        """Prepare batch (algorithm-specific, override in subclasses)."""
        pass

    def train(self, trajectories: list[Trajectory]) -> dict:
        """Main training flow."""
        # 1. Process trajectories (padding, tokenization)
        trajectories = self._process_trajectories(trajectories)

        # 2. Convert to batch
        batch = Batch.from_trajectories(trajectories, model_tag=self._model_tag)
        batch = batch.pad_to_world_size(world_size=self.world_size)

        # 3. Prepare batch (algorithm-specific)
        batch, preparation_metrics = self._prepare_batch(batch)

        # 4. Execute training step
        # ... training service calls, checkpointing ...

        return train_metrics
```

#### Key Methods

##### _prepare_batch(batch: Batch) -> tuple[Batch, dict]
```python
@abstractmethod
def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]
```

**Parameters:**
- `batch`: Batch of trajectory data from rollout

**Returns:** Tuple of (prepared_batch, metrics_dict)

**Purpose:** Algorithm-specific batch preparation. Subclasses must implement this to define their algorithm logic (advantage computation, KL penalty, etc.).

##### _process_trajectories(trajectories: list[Trajectory]) -> list[Trajectory]
```python
def _process_trajectories(self, trajectories: list[Trajectory]) -> list[Trajectory]
```

**Parameters:**
- `trajectories`: Raw trajectories from rollout workers

**Returns:** Processed trajectories with padding and tensor fields

**Process:**
1. Separates prompt and response tokens based on `loss_mask`
2. Applies left padding to prompts, right padding to responses
3. Computes position IDs from attention masks
4. Creates tensor fields: `input_ids`, `attention_mask`, `position_ids`, `loss_mask`, `prompts`, `responses`

### SelfHostedGrpoTrainer

Self-hosted trainer with integrated GRPO (Group Relative Policy Optimization) algorithm.

#### Architecture

Extends `SelfHostedTrainer` and implements `_prepare_batch` with GRPO-specific logic.

#### Key Methods

##### _prepare_batch(batch: Batch) -> tuple[Batch, dict]
```python
def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
    """Prepare batch using GRPO algorithm."""
    metrics = {}

    # 1. Log rollout metrics
    self._log_rollout_metrics(batch)

    # 2. Remove redundant padding
    batch = Batch.remove_redundant_left_padding(batch, ...)
    batch = Batch.remove_redundant_right_padding(batch, ...)

    # 3. Recompute old log probabilities
    old_log_probs = self._compute_old_log_probs(batch)
    batch.values["old_log_probs"] = old_log_probs

    # 4. Compute token-level rewards
    reward_tensor = self._reward_fn(batch)
    batch.values["token_level_scores"] = reward_tensor

    # 5. Apply KL penalty (optional)
    if self._use_kl_in_reward:
        batch, kl_metrics = self._apply_kl_penalty(batch, ...)
        metrics.update(kl_metrics)

    # 6. Compute GRPO advantages
    batch = self._compute_advantage(batch)

    # 7. Compute metrics
    metrics.update(self._compute_data_metrics(batch))

    return batch, metrics
```

#### GRPO Algorithm Details

##### _compute_advantage(batch: Batch) -> Batch
```python
def _compute_advantage(self, batch: Batch) -> Batch
```

**Purpose:** Compute group-relative policy optimization advantages.

**Process:**
1. Extract group IDs from batch (`uid` or `group_id`)
2. Extract run IDs for trajectory grouping
3. Call `core_algos.compute_grpo_outcome_advantage`:
   - Groups trajectories by (group_id, run_id)
   - Computes group mean and std of rewards
   - Normalizes advantages: `(reward - mean) / (std + 1e-8)`
4. Store advantages and returns in batch
5. Store group std statistics in metadata

##### _reward_fn(batch: Batch) -> torch.Tensor
```python
def _reward_fn(self, batch: Batch) -> torch.Tensor
```

**Purpose:** Convert scalar rewards to token-level reward tensors.

**Process:**
1. Creates zero tensor matching response shape
2. For each trajectory, assigns reward to last valid token position
3. Uses `loss_mask` to identify valid tokens
4. Returns token-level reward tensor

##### _apply_kl_penalty(batch, kl_ctrl, kl_penalty) -> tuple[Batch, dict]
```python
def _apply_kl_penalty(self, batch, kl_ctrl, kl_penalty) -> tuple[Batch, dict]
```

**Purpose:** Apply KL divergence penalty to rewards.

**Process:**
1. Computes KL divergence between current and reference policy
2. Scales by KL controller coefficient
3. Subtracts from token-level scores: `rewards = scores - beta * kld`
4. Updates KL controller based on current KL
5. Returns modified batch and KL metrics

### RemoteApiTrainer

Base trainer for remote API training backends (Tinker, Weaver).

#### Architecture

```python
class RemoteApiTrainer(BaseTrainer):
    """Base trainer for remote API backends."""

    @abstractmethod
    def _prepare_trajectories(
        self,
        trajectories: list[Trajectory],
        metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """Prepare trajectories (algorithm-specific, override in subclasses)."""
        pass

    def train(self, trajectories: list[Trajectory]) -> dict:
        """Main training flow."""
        metrics = {}

        # 1. Prepare trajectories (algorithm-specific)
        trajectories = self._prepare_trajectories(trajectories, metrics)

        # 2. Convert to service format
        datums = self._convert_trajectories_to_datums(trajectories)

        # 3. Send to remote API
        response = self._service_holder.train(datums, config=train_config)

        return metrics
```

#### Key Methods

##### _prepare_trajectories(trajectories, metrics) -> list[Trajectory]
```python
@abstractmethod
def _prepare_trajectories(
    self,
    trajectories: list[Trajectory],
    metrics: dict[str, Any]
) -> list[Trajectory]
```

**Parameters:**
- `trajectories`: List of trajectories from rollout
- `metrics`: Dictionary to populate with metrics

**Returns:** Processed trajectories

**Purpose:** Algorithm-specific trajectory preparation. Subclasses implement this to add algorithm-specific fields (advantages, etc.).

##### set_service_holder(service_holder)
```python
def set_service_holder(self, service_holder: TinkerServiceHolder | WeaverServiceHolder)
```

**Parameters:**
- `service_holder`: Service holder instance for API communication

**Purpose:** Set the remote API service holder (Tinker or Weaver).

### RemoteApiGrpoTrainer

Remote API trainer with GRPO algorithm for Tinker/Weaver backends.

#### Key Methods

##### _prepare_trajectories(trajectories, metrics) -> list[Trajectory]
```python
def _prepare_trajectories(self, trajectories, metrics) -> list[Trajectory]:
    """Prepare trajectories with GRPO advantage computation."""
    from ..algorithm.core_algos import compute_grpo_advantage_for_trajectories
    from ..utils.logging_utils import log_grpo_metrics

    # Compute GRPO advantages (groups by run_id)
    trajectories = compute_grpo_advantage_for_trajectories(
        trajectories, logger=logger, use_run_ids=True
    )

    # Log GRPO statistics
    log_grpo_metrics(trajectories, metrics)

    return trajectories
```

**Process:**
1. Groups trajectories by `run_id` (same prompt)
2. Computes group-relative advantages within each group
3. Normalizes by group std deviation
4. Logs GRPO statistics to metrics dictionary

### RemoteApiCrossEntropyTrainer

Remote API trainer with cross-entropy loss for supervised learning.

#### Purpose

Used for supervised fine-tuning or imitation learning tasks where trajectories contain reference outputs.

## Validation Components

### Validator

Collects validation trajectories, computes metrics, and logs results. Unlike TrajectoryPool, this component focuses on simple collection without batching logic.

#### Constructor
```python
def __init__(self, config: DictConfig)
```

**Parameters:**
- `config`: Validator configuration

#### Setup Methods

##### set_module_references(validate_dataloader)
```python
def set_module_references(self, validate_dataloader: BaseDataLoader)
```

**Parameters:**
- `validate_dataloader`: Reference to validation data loader

#### Methods

##### put_trajectory(trajectory: Trajectory) -> str
```python
def put_trajectory(self, trajectory: Trajectory) -> str
```

**Parameters:**
- `trajectory`: Validation trajectory to store

**Returns:** "success" to match TrajectoryPool.put_trajectory signature

Stores a validation trajectory for later metric computation.

##### is_complete() -> bool
```python
def is_complete(self) -> bool
```

**Returns:** True if all validation trajectories have been collected

Checks if validation dataloader is drained and rollout workers are quiescent.

##### compute_and_log_metrics() -> dict[str, float]
```python
def compute_and_log_metrics(self) -> dict[str, float]
```

**Returns:** Dictionary of computed metrics with "val/" prefix

Computes mean of each score key across all trajectories and logs results via activity tracker.

##### clear()
```python
def clear()
```

Clears all stored validation trajectories.

## Training Service Integration

### TrainServiceClient

Base client interface for communicating with training services. Different backends (self-hosted, Tinker, Weaver) provide their own implementations.

Training services provide the actual model training execution through standardized APIs. NexRL trainers communicate with these services to perform forward passes, backward passes, and optimization steps.

**Available Service Clients:**
- Self-hosted backend: Direct API calls to training workers
- Tinker: Uses TinkerServiceHolder for managed training
- Weaver: Uses WeaverServiceHolder for managed training

**Key Operations:**
- `forward()`: Forward pass through the model
- `forward_backward()`: Combined forward and backward pass
- `optim_step()`: Optimizer step
- `save_checkpoint()`: Save model checkpoint
- `load_checkpoint()`: Load model checkpoint

See trainer implementations (`SelfHostedGrpoTrainer`, `RemoteApiGrpoTrainer`) for usage examples.

### WeightSyncController

Manages model weights and synchronization coordination across the system. Supports multiple synchronization modes and coordinates with trajectory pools and rollout services.

#### Constructor
```python
def __init__(self, config: DictConfig)
```

**Parameters:**
- `config`: Weight manager configuration

**Configuration Options:**
- `sync_mode`: Synchronization mode ("sync", "fully-async", "batch-async")
- `staleness_threshold`: Maximum staleness allowed in async modes
- `checkpoint_manager`: Checkpoint manager configuration

#### Setup Methods

##### set_module_references(dataloader, trajectory_pool)
```python
def set_module_references(self, dataloader: BaseDataLoader, trajectory_pool: TrajectoryPool) -> None
```

**Parameters:**
- `dataloader`: Reference to data loader
- `trajectory_pool`: Reference to trajectory pool

#### Coordination Methods

##### check_rollout_service_status(model_tag: ModelTag) -> Literal["continue", "block"]
```python
def check_rollout_service_status(self, model_tag: ModelTag) -> Literal["continue", "block"]
```

**Parameters:**
- `model_tag`: Model to check status for

**Returns:**
- `"continue"`: Rollout service can continue processing
- `"block"`: Rollout service should block for weight sync

##### trajectory_pool_notify_batch_ready(model_tag: ModelTag)
```python
def trajectory_pool_notify_batch_ready(self, model_tag: ModelTag) -> None
```

**Parameters:**
- `model_tag`: Model with ready batch

Coordinates weight synchronization when training batches are ready.

##### train_worker_notify_weight_update(worker_name: str, model_tag: ModelTag)
```python
def train_worker_notify_weight_update(self, worker_name: str, model_tag: ModelTag) -> None
```

**Parameters:**
- `worker_name`: Name of the training worker
- `model_tag`: Model that was trained

Handles training completion and performs synchronous weight synchronization. May trigger validation if configured.

##### sync_weight_to_rollout_service(model_tag: ModelTag)
```python
def sync_weight_to_rollout_service(self, model_tag: ModelTag) -> None
```

**Parameters:**
- `model_tag`: Model to sync weights for

Synchronizes model weights from training service to rollout/inference service.

##### get_rollout_model_version(model_tag: ModelTag) -> int
```python
def get_rollout_model_version(self, model_tag: ModelTag) -> int
```

**Parameters:**
- `model_tag`: Model to get version for

**Returns:** Current rollout model version number

##### is_waiting_for_validation() -> bool
```python
def is_waiting_for_validation(self) -> bool
```

**Returns:** True if weight sync is waiting for validation to complete

Used by controller to trigger validation cycles after weight updates.

##### end_validate(model_tag: ModelTag)
```python
def end_validate(self, model_tag: ModelTag) -> None
```

**Parameters:**
- `model_tag`: Model that completed validation

Called after validation completes to unlock weight synchronization.

#### Synchronization Modes

##### sync
- Blocks all workers until all sync to newest version
- Ensures strict consistency across all components

##### fully-async
- No blocking, workers sync opportunistically
- Allows maximum throughput with potential staleness

##### batch-async
- Blocks individual workers when staleness exceeds threshold
- Balances consistency and throughput

#### Checkpoint Management

Checkpoint management is handled through the TrainServiceClient interface, which provides methods for saving and loading checkpoints. The WeightSyncController coordinates checkpoint operations but delegates actual checkpoint I/O to the training service.

**Key Operations:**
- Checkpoints are saved via `TrainServiceClient.save_checkpoint()` during training
- Initial checkpoint loading is handled by the controller during startup
- Resume functionality supports automatic checkpoint discovery or explicit path specification
- Weight synchronization uses a dedicated sync weight buffer path

## Activity Monitoring

### ActivityTracker

Centralized tracker for monitoring in-flight work across all modules and coordinating experiment logging.

#### Constructor
```python
def __init__(self, config: DictConfig, max_errors: int = 1000) -> None
```

**Parameters:**
- `config`: Configuration containing project and experiment names for logging
- `max_errors`: Maximum errors to retain in memory

**Attributes:**
- `experiment_logger`: Tracking instance for logging metrics to wandb/etc.

#### Activity Tracking

##### start(module: str, work: str) -> str
```python
def start(self, module: str, work: str) -> str
```

**Parameters:**
- `module`: Module name starting work
- `work`: Description of work being performed

**Returns:** Unique token for this work item

##### end(token: str) -> None
```python
def end(self, token: str) -> None
```

**Parameters:**
- `token`: Token from corresponding start() call

#### Status Monitoring

##### is_quiescent() -> bool
```python
def is_quiescent(self) -> bool
```

**Returns:** True if no work is currently in progress

##### wait_quiescent(timeout: float | None = None) -> bool
```python
def wait_quiescent(self, timeout: float | None = None) -> bool
```

**Parameters:**
- `timeout`: Maximum time to wait, or None for indefinite

**Returns:** True if quiescence achieved, False if timeout

##### get_running_status_summary() -> str
```python
def get_running_status_summary(self) -> str
```

**Returns:** Human-readable summary of current activity

#### Module Health

##### register_module(module_name: str, module_ref: Any, is_rollout_worker: bool = False) -> None
```python
def register_module(self, module_name: str, module_ref: Any, is_rollout_worker: bool = False) -> None
```

**Parameters:**
- `module_name`: Name for health checking
- `module_ref`: Reference to module (local object or Ray actor)
- `is_rollout_worker`: Whether this module is a rollout worker (for specialized monitoring)

##### is_rollout_worker_quiescent() -> bool
```python
def is_rollout_worker_quiescent(self) -> bool
```

**Returns:** True if all registered rollout workers are idle

##### check_module_liveness(timeout: float = 5.0) -> bool
```python
def check_module_liveness(self, timeout: float = 5.0) -> bool
```

**Parameters:**
- `timeout`: Timeout for Ray operations

**Returns:** True if all registered modules are alive

#### Error Reporting

##### report_exception(module: str, work: str, exception: Exception, severity: ErrorSeverity | None = None) -> str
```python
def report_exception(self, module: str, work: str, exception: Exception, severity: ErrorSeverity | None = None) -> str
```

**Parameters:**
- `module`: Module where exception occurred
- `work`: Work context
- `exception`: Exception that was raised
- `severity`: Error severity (defaults to ERROR)

**Returns:** Unique error ID

##### get_error_health_status() -> dict[str, Any]
```python
def get_error_health_status(self) -> dict[str, Any]
```

**Returns:** Dictionary with health status information

#### Training Step Tracking

##### set_training_step(step: int) -> None
```python
def set_training_step(self, step: int) -> None
```

**Parameters:**
- `step`: Current training step

Updates the current training step for logging and monitoring purposes.

##### get_training_step() -> int
```python
def get_training_step(self) -> int
```

**Returns:** Current training step

#### Experiment Logging

##### experiment_logger_post(backend: str, **kwargs)
```python
def experiment_logger_post(self, backend: str, **kwargs)
```

**Parameters:**
- `backend`: Logging backend ("wandb", etc.)
- `**kwargs`: Backend-specific parameters (e.g., data, step, content, title)

Posts metrics or messages to the specified logging backend through the experiment_logger.

### ActivityTrackerProxy

Local proxy that forwards activity tracking calls to a central ActivityTracker.

#### Constructor
```python
def __init__(self, central_tracker: Any)
```

**Parameters:**
- `central_tracker`: Reference to central ActivityTracker (local or Ray actor)

#### Usage

##### track(module: str, work: str, auto_report_errors: bool = True)
```python
def track(self, module: str, work: str, auto_report_errors: bool = True) -> "_ProxyTrackCtx"
```

**Parameters:**
- `module`: Module name performing work
- `work`: Description of work
- `auto_report_errors`: Whether to automatically report exceptions

**Returns:** Context manager for activity tracking

##### is_rollout_worker_quiescent() -> bool
```python
def is_rollout_worker_quiescent(self) -> bool
```

**Returns:** True if all registered rollout workers are idle

##### set_training_step(step: int) -> None
```python
def set_training_step(self, step: int) -> None
```

**Parameters:**
- `step`: Current training step

Forwards training step update to central tracker.

##### get_training_step() -> int
```python
def get_training_step(self) -> int
```

**Returns:** Current training step from central tracker

##### experiment_logger_post(backend: str, **kwargs)
```python
def experiment_logger_post(self, backend: str, **kwargs)
```

**Parameters:**
- `backend`: Logging backend ("wandb", etc.)
- `**kwargs`: Backend-specific parameters

Forwards logging request to central tracker's experiment_logger.

**Usage Example:**
```python
with activity_tracker.track("MyModule", "processing_batch"):
    # Perform work here
    process_batch(batch)
    # Activity automatically tracked and errors reported
```

## Error Management

### ErrorReporter

Centralized error reporting and aggregation system.

#### Constructor
```python
def __init__(self, max_errors: int = 1000)
```

**Parameters:**
- `max_errors`: Maximum errors to keep in memory

#### Error Reporting

##### report_exception(module_name: str, work_context: str, exception: Exception, severity: ErrorSeverity = ErrorSeverity.ERROR, details: dict[str, Any] | None = None) -> str
```python
def report_exception(self, module_name: str, work_context: str, exception: Exception, severity: ErrorSeverity = ErrorSeverity.ERROR, details: dict[str, Any] | None = None) -> str
```

**Parameters:**
- `module_name`: Module where error occurred
- `work_context`: Context of work being performed
- `exception`: Exception instance
- `severity`: Error severity level
- `details`: Additional error details

**Returns:** Unique error ID

##### report_error(module_name: str, work_context: str, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, details: dict[str, Any] | None = None) -> str
```python
def report_error(self, module_name: str, work_context: str, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, details: dict[str, Any] | None = None) -> str
```

**Parameters:**
- `module_name`: Module where error occurred
- `work_context`: Context of work being performed
- `message`: Error message
- `severity`: Error severity level
- `details`: Additional error details

**Returns:** Unique error ID

#### Health Assessment

##### get_health_status() -> dict[str, Any]
```python
def get_health_status(self) -> dict[str, Any]
```

**Returns:** Dictionary containing:
- `status`: Overall health status ("healthy", "warning", "error")
- `message`: Summary message
- `recent_error_count`: Total recent errors
- `error_level_count`: Recent error-level issues
- `warning_level_count`: Recent warning-level issues

### ErrorSeverity

Enumeration of error severity levels.

```python
class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
```

### ErrorInfo

Data class containing error information.

```python
@dataclass
class ErrorInfo:
    error_id: str                    # Unique error identifier
    timestamp: float                 # Error occurrence time
    module_name: str                 # Module where error occurred
    work_context: str               # Work context
    severity: ErrorSeverity         # Severity level
    message: str                    # Error message
    details: dict[str, Any]         # Additional details
    exception_type: str | None      # Exception type name
    exception_traceback: str | None # Full traceback
```

## Resource Management

### RayResourceManager

Manages Ray actors and handles actor creation with co-location support.

#### Constructor
```python
def __init__(self)
```

Initializes the resource manager with empty role registrations.

#### Setup Methods

##### register_role(role: NexRLRole, cls: type, config: DictConfig, count: int, colocation_group: str | None)
```python
def register_role(self, role: NexRLRole, cls: type, config: DictConfig, count: int, colocation_group: str | None = None)
```

**Parameters:**
- `role`: NexRL role being registered
- `cls`: Class to instantiate for this role
- `config`: Configuration for the role
- `count`: Number of instances to create
- `colocation_group`: Optional group name for co-location (None = standalone actor)

**Colocation Behavior:**
- Roles with the same `colocation_group` will share a single Ray actor
- Roles with `colocation_group=None` get dedicated actors
- Methods are prefixed with role name for co-located actors

##### create_all_actors()
```python
def create_all_actors()
```

Creates and deploys all registered actors based on role registrations and colocation groups.

#### Access Methods

##### get_actor_wrapper(role: NexRLRole) -> list[RayActorWrapper]
```python
def get_actor_wrapper(self, role: NexRLRole) -> list[Any]
```

**Parameters:**
- `role`: Role to get actor wrappers for

**Returns:** List of actor wrappers for the role

### RayActorWrapper

Wrapper that enables elegant access to co-located Ray actors.

#### Constructor
```python
def __init__(self, actor: ActorHandle, actor_class: type, role: NexRLRole, is_colocated: bool = True)
```

**Parameters:**
- `actor`: Ray actor handle
- `actor_class`: Original actor class
- `role`: NexRL role of this component
- `is_colocated`: Whether actor is co-located with others

**Functionality:**
- Automatically rebinds methods from actor to wrapper
- Handles role-prefixed method names for co-located actors
- Provides transparent access to actor methods

## Utility Functions

### Execution Utilities

#### execute(func, *args, **kwargs) -> Any
```python
def execute(func: Any, *args, **kwargs) -> Any
```

**Parameters:**
- `func`: Function to execute (local or Ray remote)
- `*args, **kwargs`: Function arguments

**Returns:** Function result

**Behavior:**
- Local mode: Always executes locally
- Ray mode: Auto-detects Ray remote methods and uses ray.get()

#### execute_async(func, *args, **kwargs) -> Any
```python
def execute_async(func: Any, *args, **kwargs) -> Any
```

**Parameters:**
- `func`: Function to execute
- `*args, **kwargs`: Function arguments

**Returns:** Immediate result (local) or ObjectRef (Ray)

**Purpose:** Enables asynchronous execution patterns

### Logging Utilities

#### set_logging_basic_config(level)
```python
def set_logging_basic_config(level)
```

**Parameters:**
- `level`: Logging level (e.g., logging.DEBUG)

**Purpose:** Sets up consistent logging format across the framework

## Configuration

NexRL uses Hydra for configuration management. The main configuration structure includes:

### Main Configuration
- `launch_mode`: "local" or "ray"
- `project_name`: Project name for experiment tracking
- `experiment_name`: Experiment name for logging
- `data`: DataLoader configuration
- `rollout_worker`: RolloutWorker configuration
- `trajectory_pool`: TrajectoryPool configuration
- `trainer`: Trainer configuration
- `algorithm`: Algorithm configuration (only for self_hosted_grpo trainer)
- `weight`: WeightSyncController configuration
- `service`: Service configurations (train_service, inference_service)
- `validate`: Validation configuration
- `resume`: Resume configuration
- `logger`: Logging backend configuration
- `runtime_monitor`: Runtime monitoring configuration

### Data Loader Configuration
- `type`: Loader type ("mock", "torch")
- `seed`: Random seed for data loading
- Additional configuration depends on loader type

### Rollout Worker Configuration
- `type`: Worker type ("mock", "simple", "single_turn_math")
- `num_workers`: Total number of rollout workers

### Trajectory Pool Configuration
- `type`: Pool type ("default")
- `batch_size`: Batch size for trajectory processing
- `group_size`: Size of trajectory groups
- `key_list`: Keys for hierarchical grouping
- `check_batch_ready_function`: Batch readiness criteria

### Trainer Configuration
- `type`: Trainer type ("self_hosted_grpo", "remote_api_grpo", "remote_api_cross_entropy")
- `total_train_steps`: Maximum training steps
- `checkpoint_path`: Path to save checkpoints (self-hosted only)
- `sync_weight_path`: Path for weight synchronization buffer (self-hosted only)
- `save_freq`: Checkpoint save frequency in steps (self-hosted only)
- `remove_previous_ckpt`: Whether to remove previous checkpoints (self-hosted only)

### Algorithm Configuration (Self-Hosted GRPO Only)
- `type`: Algorithm type ("grpo")
- `batch_size`: Batch size for training
- `do_old_log_prob_compute`: Whether to recompute old log probabilities
- `use_kl_in_reward`: Whether to include KL penalty in rewards
- `critic.kl_ctrl`: KL controller configuration (adaptive or fixed)
- `inference_service`: Reference model configuration for old log probs

### Weight Synchronization Configuration
- `type`: Controller type ("default")
- `sync_mode`: Synchronization mode ("sync", "fully-async", "batch-async")
- `staleness_threshold`: Maximum staleness in async modes

### Service Configuration
- `train_service`: Training service configuration
  - `backend`: Service backend ("mock", "nextrainer")
  - `url`: Service URL
  - `model_tag`: Model identifier
  - `identifier`: Optional service identifier
- `inference_service`: Inference service configuration
  - `backend`: Service backend ("vllm", etc.)
  - `url`: Service URL
  - `model_tag`: Model identifier
  - `api_key`: API key for service
  - `max_retries`: Retry attempts
  - `freeze_for_weight_sync`: Whether to block during weight sync

### Validation Configuration
- `validate_before_train`: Run validation before starting training
- `data`: Validation dataloader configuration (same structure as main data config)
- `eval`: Validator configuration
  - `type`: Validator type ("default")

### Resume Configuration
- `mode`: Resume mode ("disable", "auto", "from_path")
- `resume_path`: Path to checkpoint for "from_path" mode

### Logger Configuration
- `backend`: Logging backend ("wandb", etc.)

### Runtime Monitoring Configuration
- `runtime_monitor`:
  - `exception_handling`:
    - `enabled`: Enable exception monitoring
    - `check_interval`: Exception check interval (seconds)
    - `policy`: Error handling policy ("stop_on_error", "continue", "stop_on_critical")
  - `health_check`:
    - `enabled`: Enable module liveness monitoring
    - `check_interval`: Health check interval (seconds)
    - `timeout`: Health check timeout (Ray mode only)

## Usage Patterns

### Basic Local Usage
```python
import hydra
from omegaconf import DictConfig
from nexrl import NexRLController

@hydra.main(config_path="config", config_name="rl_train")
def main(config: DictConfig):
    config.launch_mode = "local"
    controller = NexRLController(config)
    controller.run()
```

### Ray Distributed Usage
```python
import ray
import hydra
from omegaconf import DictConfig
from nexrl import NexRLController

@hydra.main(config_path="config", config_name="rl_train")
def main(config: DictConfig):
    config.launch_mode = "ray"
    ray.init()

    # Create controller as Ray actor
    ControllerActor = ray.remote(NexRLController)
    controller_actor = ControllerActor.remote(config)
    ray.get(controller_actor.run.remote())

    ray.shutdown()
```

### Custom RolloutWorker Implementation

#### Simple Worker Example
```python
from nexrl import BaseRolloutWorker
from typing import Any

class MyRolloutWorker(BaseRolloutWorker):
    def step(self, task: dict[str, Any]) -> str | None:
        # Extract task data
        if "prompt" not in task:
            return None

        prompt = task["prompt"]

        # Custom processing logic using LLMServiceClient
        result = self._llm_client.completion(prompt, temperature=0.7)

        # Create trajectory
        trajectory = {
            "prompt": prompt,
            "response": result["response"],
            "finish_reason": result["finish_reason"],
            "custom_metadata": task.get("metadata", {}),
            "model_tag": self._llm_client._model_tag
        }

        # Submit to trajectory pool and return result
        return self._put_trajectory(trajectory)
```

#### NexAU Worker with Custom Query Formatting
```python
from nexrl.rollout_worker import BaseNexAURolloutWorker
from typing import Any

class MyNexAUWorker(BaseNexAURolloutWorker):
    """Custom NexAU worker with specialized query formatting."""

    def format_task_query(self, data_item: dict[str, Any]) -> str:
        """
        Format task-specific data into agent query.

        This method is called by run_agent to convert raw data
        into a format the agent can process.
        """
        # Extract fields from your data
        context = data_item.get("context", "")
        question = data_item.get("question", "")

        # Format into task-specific template
        query = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        return query

    def get_train_loss_mask(self, trajectory_infos: list[dict]) -> list[bool]:
        """
        Optional: Customize which tokens contribute to loss.

        By default, all response tokens are used for training.
        Override this to mask certain tokens (e.g., system messages, tools).
        """
        # Example: Use all response tokens (default behavior)
        return [True] * len(trajectory_infos)
```

#### Custom Evaluator Implementation
```python
from nexrl.rollout_worker import (
    Evaluator,
    BaseEvaluationTarget,
    EvaluationRunResult
)
from typing import Any

class MyTaskEvaluator(Evaluator):
    """Custom evaluator for specific task metrics."""

    def evaluate(
        self,
        data: dict[str, Any],
        evaluation_target: BaseEvaluationTarget
    ) -> EvaluationRunResult:
        """
        Evaluate agent output and compute reward/metrics.

        Args:
            data: Original task data (may contain ground truth)
            evaluation_target: Agent output with final_answer field

        Returns:
            EvaluationRunResult with reward, ground_truth, metrics, extra_info
        """
        # Extract agent's answer
        agent_answer = evaluation_target.final_answer

        # Get ground truth from data
        ground_truth = data.get("ground_truth", "")

        # Compute reward (primary signal for RL)
        reward = 1.0 if agent_answer.strip() == ground_truth.strip() else 0.0

        # Compute additional metrics (must be scalar floats)
        metrics = {
            "answer_length": float(len(agent_answer)),
            "exact_match": reward,
        }

        # Store extra info (can be any type)
        extra_info = {
            "agent_answer": agent_answer,
            "parsed_output": self._parse_answer(agent_answer),
        }

        return EvaluationRunResult(
            reward=reward,
            ground_truth=ground_truth,
            metrics=metrics,
            extra_info=extra_info
        )

    def _parse_answer(self, answer: str) -> dict:
        """Helper method for answer parsing."""
        # Your parsing logic here
        return {"parsed": answer}
```

### Custom Trainer Implementation

For self-hosted training, extend `SelfHostedTrainer`:

```python
from nexrl.trainer import SelfHostedTrainer
from nexrl.nexrl_types import Batch

class MyCustomTrainer(SelfHostedTrainer):
    """Custom trainer with custom algorithm logic."""

    def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
        """
        Implement custom batch preparation logic.

        Args:
            batch: Batch from trajectory pool

        Returns:
            Tuple of (prepared_batch, metrics_dict)
        """
        metrics = {}

        # 1. Your preprocessing
        # ... compute custom features ...

        # 2. Compute algorithm-specific values (e.g., advantages)
        batch.values["advantages"] = self._compute_advantages(batch)

        # 3. Log metrics
        metrics["custom/mean_advantage"] = batch.values["advantages"].mean().item()

        return batch, metrics

    def _compute_advantages(self, batch: Batch):
        # Your advantage computation logic
        pass
```

For remote API training, extend `RemoteApiTrainer`:

```python
from nexrl.trainer import RemoteApiTrainer
from nexrl.nexrl_types import Trajectory
from typing import Any

class MyRemoteTrainer(RemoteApiTrainer):
    """Custom remote API trainer."""

    def _prepare_trajectories(
        self,
        trajectories: list[Trajectory],
        metrics: dict[str, Any]
    ) -> list[Trajectory]:
        """
        Prepare trajectories for remote API.

        Args:
            trajectories: List of trajectories from rollout
            metrics: Dictionary to populate with metrics

        Returns:
            Processed trajectories
        """
        # Your trajectory processing logic
        for traj in trajectories:
            traj["custom_field"] = self._compute_custom_field(traj)

        # Log metrics
        metrics["custom_metric"] = self._compute_metric(trajectories)

        return trajectories
```

### Activity Tracking Usage
```python
# In any module with activity tracker
with self._activity_tracker.track("MyModule", "processing_data"):
    # Perform work that should be tracked
    result = process_data(data)
    # Automatic activity tracking and error reporting

# Check rollout worker specific status
if self._activity_tracker.is_rollout_worker_quiescent():
    logger.info("All rollout workers are idle")

# Log metrics to experiment tracking
self._activity_tracker.experiment_logger_post(
    backend="wandb",
    data={"metric_name": value},
    step=training_step
)
```

### Validation Usage
```python
# Validation is automatically triggered by the controller when:
# 1. validate_before_train is enabled (runs before training starts)
# 2. Weight sync completes and validation frequency is configured

# Custom rollout workers should handle validation mode:
def step(self, task: dict[str, Any]) -> str | None:
    # Check if in validation mode
    if self._is_running_validate:
        # Use validation dataloader and validator
        validate_task = self._get_validate_task()
        trajectory = self._process_task(validate_task)
        return self._put_validate_trajectory(trajectory)
    else:
        # Normal training mode
        trajectory = self._process_task(task)
        return self._put_trajectory(trajectory)
```

## Best Practices

### Module Development
1. Always inherit from NexRLModule for Ray compatibility
2. Implement proper cleanup in stop() methods
3. Use activity tracking for long-running operations
4. Handle exceptions gracefully and report through activity tracker

### Resource Management
1. Define clear resource pool mappings based on workload
2. Use co-location for related services to reduce communication overhead
3. Monitor resource usage through activity tracker
4. Plan GPU allocation based on model requirements

### Error Handling
1. Use structured error reporting through ErrorReporter
2. Implement proper retry logic for transient failures
3. Monitor system health through activity tracker
4. Define clear error policies for different scenarios

### Configuration Management
1. Use Hydra for configuration management
2. Define environment-specific overrides
3. Validate configurations before deployment
4. Document configuration options clearly

### Performance Optimization
1. Use appropriate batching strategies for trajectory collection
2. Monitor activity tracker for bottlenecks
3. Optimize resource pool configurations
4. Use async execution patterns where appropriate

## Troubleshooting

### Common Issues

#### Ray Connection Problems
- Check Ray cluster status with `ray status`
- Verify placement group creation
- Monitor resource allocation

#### Module Communication Failures
- Check activity tracker health status
- Verify module references are set correctly
- Monitor error reporter for communication errors

#### Performance Issues
- Check activity tracker for bottlenecks
- Monitor resource utilization
- Verify batching configurations

#### Configuration Problems
- Validate configuration syntax
- Check module type compatibility
- Verify resource specifications

#### Validation Issues
- Ensure validation dataloader is properly configured
- Check that rollout workers support validation mode (begin_validate/end_validate)
- Verify validator is receiving trajectories
- Check validation frequency configuration in weight sync controller

#### Checkpoint Issues
- Verify checkpoint paths are accessible
- Check resume mode configuration (disable/auto/from_path)
- Ensure checkpoint directory structure matches expected format (global_step_*)
- Verify train service has proper checkpoint save/load permissions

### Debugging Tools

#### Activity Monitoring
```python
# Get current system status
status = controller.activity_tracker.get_running_status_summary()
print(f"System status: {status}")

# Check health
health = controller.activity_tracker.get_error_health_status()
print(f"Health status: {health}")
```

#### Resource Monitoring
```python
# Check actor wrappers for each role
for role in NexRLRole:
    wrappers = resource_manager.get_actor_wrapper(role)
    print(f"{role}: {len(wrappers)} actors")

# Check module health
for module_name, module_ref in controller.activity_tracker._module_refs.items():
    is_alive = controller.activity_tracker.check_module_liveness()
    print(f"{module_name}: {'alive' if is_alive else 'dead'}")
```

#### Error Analysis
```python
# Get recent errors
health_status = error_reporter.get_health_status()
if health_status["status"] != "healthy":
    print(f"System unhealthy: {health_status['message']}")
```

This developer guide provides comprehensive coverage of the NexRL framework's architecture, components, and usage patterns. For specific implementation details, refer to the source code and configuration examples.
