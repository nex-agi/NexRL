# NexRL Test Suite

This directory contains the test suite for the NexRL framework.

## Structure

The test directory is organized into two main categories:

```
tests/
├── __init__.py
├── conftest.py                    # Top-level pytest configuration
├── README.md                      # This file
├── unittests/                     # Unit tests for framework components
│   ├── __init__.py
│   ├── conftest.py               # Shared pytest fixtures
│   ├── test_base_module.py       # Tests for base_module
│   ├── test_nexrl_types.py       # Tests for nexrl_types
│   ├── test_train_batch_pool.py  # Tests for train_batch_pool
│   ├── mock/                     # Tests for mock components
│   │   ├── __init__.py
│   │   ├── test_mock_data_loader.py
│   │   ├── test_mock_rollout_worker.py
│   │   ├── test_mock_algorithm_processor.py
│   │   └── test_mock_llm_service_client.py
│   ├── data_loader/              # Tests for data_loader module
│   │   ├── __init__.py
│   │   ├── test_base_data_loader.py
│   │   ├── test_torch_data_loader.py
│   │   └── test_data/
│   │       └── test_dataset.parquet
│   ├── rollout_worker/           # Tests for rollout_worker module
│   │   ├── __init__.py
│   │   ├── test_base_rollout_worker.py
│   │   └── test_simple_rollout_worker.py
│   ├── train_worker/             # Tests for train_worker module
│   │   └── __init__.py
│   ├── trajectory_pool/          # Tests for trajectory_pool module
│   │   └── __init__.py
│   ├── algorithm_processor/      # Tests for algorithm_processor module
│   │   └── __init__.py
│   └── utils/                    # Tests for utils module
│       ├── __init__.py
│       └── test_config_utils.py
└── lint/                         # Lint and code quality check scripts
    ├── __init__.py
    └── check_license_header.py   # License header compliance checker (pre-commit hook)
```

### Test Categories

- **unittests/**: Contains unit tests for all framework components. The structure mirrors the `nexrl` module hierarchy.
- **lint/**: Contains scripts for code quality checks and linting (used by pre-commit hooks and CI/CD).

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run only unit tests
```bash
pytest tests/unittests/
```

### Run specific test file
```bash
pytest tests/unittests/test_nexrl_types.py
```

### Run with verbose output
```bash
pytest -v tests/
```

### Run tests with markers
```bash
pytest -m unit tests/
pytest -m integration tests/
```

## Writing Tests

### Unit Tests

1. Place test files in `tests/unittests/` in a directory that mirrors the module structure
2. Name test files with the `test_` prefix
3. Name test functions with the `test_` prefix
4. Use the provided fixtures from `conftest.py`
5. Keep tests simple and focused on a single behavior
6. Use mock components from `nexrl.mock` instead of `unittest.mock` when possible

### Lint Scripts

1. Place lint scripts in `tests/lint/`
2. Lint scripts check code quality, formatting, and compliance
3. These scripts are typically called by pre-commit hooks and CI/CD pipelines

## Example Test

```python
def test_example_function(basic_config):
    """Test description"""
    # Arrange
    obj = SomeClass(basic_config)

    # Act
    result = obj.some_method()

    # Assert
    assert result is not None
```

## Available Fixtures

- `basic_config`: Basic configuration for general testing
- `data_loader_config`: Configuration for data loader testing
- `rollout_worker_config`: Configuration for rollout worker testing
- `train_batch_pool_config`: Configuration for train batch pool testing
