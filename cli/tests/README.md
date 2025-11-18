# NexRL CLI Tests

This directory contains unit tests for the NexRL CLI tool.

## Running Tests

Run all tests:
```bash
cd NexRL
python -m pytest cli/tests/
```

Run specific test file:
```bash
python -m pytest cli/tests/test_k8s_utils.py
```

Run with coverage:
```bash
python -m pytest --cov=cli cli/tests/
```

## Test Structure

- `test_k8s_utils.py`: Tests for Kubernetes utilities
- `test_validation.py`: Tests for validation functions
- `test_yaml_builder.py`: Tests for YAML template rendering

## Requirements

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-mock
```
