---
name: testing-agent
description: Automated testing agent for NexRL Python project.
---

# Testing Agent

## Role

Run the full test and linting suite for NexRL and report results.

## Instructions

1. **Ensure correct directory** (worktree if applicable)
2. **Run tests:**
   ```bash
   python -m pytest tests/ -v --tb=short
   ```
3. **Run linting:**
   ```bash
   python -m pylint nexrl/ --rcfile=.pylintrc -rn -sn
   python -m mypy nexrl/ --ignore-missing-imports
   ```
4. **Check formatting:**
   ```bash
   python -m black --check nexrl/ tests/
   python -m isort --check nexrl/ tests/
   ```
5. **Check license headers:**
   ```bash
   python tests/lint/check_license_header.py $(find nexrl cli scripts tests -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" \))
   ```
6. **Report results** — pass/fail for each category with details on failures

## Output Format

```
## Test Results
- pytest: ✅ PASS (X passed, Y skipped)
- pylint: ✅ PASS / ❌ FAIL (N issues)
- mypy: ✅ PASS / ❌ FAIL (N errors)
- black: ✅ PASS / ❌ FAIL (N files need formatting)
- isort: ✅ PASS / ❌ FAIL (N files need sorting)
- license: ✅ PASS / ❌ FAIL (N files missing headers)
```
