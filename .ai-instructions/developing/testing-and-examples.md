# Testing and Examples Policy

## Core Principles

**DO NOT** write examples or temporary test scripts unless explicitly requested by the user.

## Testing Guidelines

### Test Location
- All tests belong in the `tests/` directory
- If a test is needed for a feature, add it to the appropriate subfolder in `tests/`
- **NEVER** create temporary test files or example scripts outside the `tests/` directory

### Test Structure
The project has the following test structure:
- `tests/unittests/` - Core functionality tests
- `tests/lint/` - Linting and code quality checks

### When to Add Tests
- When implementing new features that require validation
- When fixing bugs to ensure the fix works and prevent regression
- When the user explicitly requests tests

### When NOT to Create Tests
- Do not create temporary "proof of concept" test files
- Do not create ad-hoc example scripts to demonstrate functionality
- Do not create test files just to show how something works unless requested

## Examples Policy

### Do NOT Write Examples Unless Asked
- Examples should only be created when explicitly requested by the user
- If you need to demonstrate something, explain it in comments or documentation instead

## Summary

- ❌ No temporary test files or examples
- ✅ Add tests to `tests/` when needed
- ❌ No examples unless user asks
- ❌ No standalone docs for dev tasks
- ✅ Check and update docs in `docs/` for core features
