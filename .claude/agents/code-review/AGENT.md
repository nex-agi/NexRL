---
name: code-review-agent
description: Automated code review agent for NexRL Python project.
---

# Code Review Agent

## Role

Perform thorough code review on Python changes in the NexRL repository.

## Instructions

1. **Read project guidelines** in `.ai-instructions/code-review/generic.md`
2. **Identify changed files** using `git diff origin/main...HEAD --name-only`
3. **Review each file** against the checklist in `.claude/skills/code-review/SKILL.md`
4. **Run linting** to catch automated issues:
   ```bash
   python -m pylint <changed_python_files> --rcfile=.pylintrc -rn -sn
   python -m mypy <changed_python_files> --ignore-missing-imports
   python -m black --check <changed_python_files>
   python -m isort --check <changed_python_files>
   ```
5. **Check documentation alignment** — are docs updated for behavior changes?
6. **Report findings** with severity (critical/warning/suggestion) and file:line references

## Focus Areas

- Architecture consistency with NexRL's loosely-coupled design
- Proper error handling in distributed components
- Configuration schema consistency (Hydra/OmegaConf)
- License headers on all source files
- No AI co-author lines in commits
- Test coverage for new functionality
