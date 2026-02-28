---
name: code-review
description: Code review for Python changes in NexRL, following project conventions.
---

# Code Review Skill

## Overview

Review Python code changes following NexRL project conventions and `.ai-instructions/code-review/generic.md`.

## Checklist

### Code Quality
- [ ] Follows project style (black, isort formatting)
- [ ] No debug code or commented-out sections
- [ ] No TODOs/FIXMEs unless documented
- [ ] License headers present on Python/shell/YAML/TOML files
- [ ] pylint and mypy pass without new warnings

### Architecture & Design
- [ ] Consistent with NexRL's loosely-coupled architecture
- [ ] Proper use of existing abstractions (workers, trainers, services)
- [ ] No circular imports
- [ ] Configuration changes follow existing patterns (Hydra/OmegaConf)

### Documentation Alignment
- [ ] Documentation in `docs/` reflects code changes
- [ ] Docstrings present for public APIs
- [ ] Configuration reference updated if parameters changed
- [ ] Examples still work

### Testing
- [ ] Tests added for new functionality (in `tests/`)
- [ ] No ad-hoc test files outside `tests/`
- [ ] Existing tests still pass

### Commit Content
- [ ] Only relevant changes included
- [ ] No build artifacts or sensitive information
- [ ] Commit message follows conventions (present tense, no AI co-author)

## How to Run

```bash
# Review staged changes
git diff --staged

# Review against base branch
git diff origin/main...HEAD

# Check specific files
python -m pylint <files> --rcfile=.pylintrc -rn -sn
python -m mypy <files> --ignore-missing-imports
```
