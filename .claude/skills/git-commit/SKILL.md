---
name: git-commit
description: Git commit workflow in a worktree with Python linting, formatting, and testing.
---

# Git Commit Workflow (Worktree-Aware)

## Prerequisites

Verify you're in a worktree (not main clone):
```bash
git rev-parse --show-toplevel
git worktree list
```

## Workflow

### Step 1: Analyze Changes

```bash
git diff --name-only
git diff --cached --name-only
```

### Step 2: Format & Lint

```bash
# Format changed Python files
python -m black <changed_python_files>
python -m isort <changed_python_files>

# Fix license headers
python tests/lint/fix_license_header.py <changed_files>

# Lint
python -m pylint <changed_python_files> --rcfile=.pylintrc -rn -sn
python -m mypy <changed_python_files> --ignore-missing-imports
```

### Step 3: Run Tests

```bash
python -m pytest tests/ -v --tb=short
```

### Step 4: Review (use code-review skill)

Follow `.ai-instructions/code-review/generic.md` checklist.

### Step 5: Stage Changes

```bash
git add path/to/changed/files
git diff --staged  # Final review
```

**Never stage:** `__pycache__/`, `.env`, `*.egg-info/`, build artifacts, coverage files

### Step 6: Commit

Format: `type(scope): description` (72 chars max)

Types: feat, fix, refactor, test, docs, style, chore, perf

```bash
git commit -m "type(scope): description

Detailed explanation if needed.

Fixes #ISSUE_NUMBER"
```

**No AI co-author lines.** (per `.ai-instructions/developing/commit-guidelines.md`)

### Step 7: Verify

```bash
git show HEAD --name-only
git log -1
```
