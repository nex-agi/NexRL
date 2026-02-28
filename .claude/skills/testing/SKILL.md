---
name: testing
description: Run tests and linting for NexRL Python project.
---

# Testing Skill

## Test Suite

```bash
# All tests
python -m pytest tests/ -v --tb=short

# Unit tests only
python -m pytest tests/unittests/ -v

# Specific test file
python -m pytest tests/unittests/test_foo.py -v

# By marker
python -m pytest tests/ -m unit -v
python -m pytest tests/ -m "not slow" -v
```

## Linting

```bash
# pylint (non-test files)
python -m pylint nexrl/ --rcfile=.pylintrc -rn -sn

# mypy
python -m mypy nexrl/ --ignore-missing-imports

# License header check
python tests/lint/check_license_header.py <files>
```

## Formatting Check

```bash
# Check (don't fix)
python -m black --check nexrl/ tests/
python -m isort --check nexrl/ tests/

# Fix
python -m black nexrl/ tests/
python -m isort nexrl/ tests/
```

## Pre-commit (all checks)

```bash
pre-commit run --all-files
```

## Worktree Usage

When running in a worktree, ensure you're in the worktree directory:
```bash
# Ensure you're in the worktree root
cd "$(git rev-parse --show-toplevel)"
python -m pytest tests/ -v --tb=short
```

## Test Markers

- `unit` — Unit tests
- `integration` — Integration tests
- `slow` — Slow running tests

## Notes

- Tests belong in `tests/` only (per `.ai-instructions/developing/testing-and-examples.md`)
- Never create temporary test files or ad-hoc example scripts
- Python 3.12+ required
