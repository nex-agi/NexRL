# Claude Code Instructions for NexRL

## Project Overview

NexRL is an ultra-loosely-coupled LLM post-training (reinforcement learning) framework by Nex-AGI Team. Python 3.12+, Apache 2.0 licensed.

## Existing Guidelines

**Always read and follow** the instructions in `.ai-instructions/` first:
- `.ai-instructions/developing/documentation.md` — Documentation-first workflow
- `.ai-instructions/developing/testing-and-examples.md` — Testing policy (tests in `tests/` only, no ad-hoc examples)
- `.ai-instructions/developing/commit-guidelines.md` — Commit message format, no AI co-author
- `.ai-instructions/code-review/generic.md` — Pre-commit review checklist

## Project Structure

```
nexrl/              # Main package (algorithm, agent, trainer, rollout_worker, etc.)
cli/                # CLI tools
tests/              # Tests (unittests/, lint/)
recipe/             # Training recipes
scripts/            # Utility scripts
docs/               # Documentation (developer-guide/)
docker/             # Docker configs
```

## Build & Tooling

- **Package manager:** setuptools (see `pyproject.toml`)
- **Formatter:** black (Python 3.12), isort
- **Linter:** pylint (`.pylintrc`), mypy
- **Tests:** pytest (`pytest.ini` — `tests/` dir, `-v --tb=short --strict-markers`)
- **Pre-commit:** `.pre-commit-config.yaml` (black, isort, license header, pylint, mypy)

## Key Commands

```bash
# Run tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/unittests/test_foo.py -v

# Lint
python -m pylint nexrl/ --rcfile=.pylintrc -rn -sn
python -m mypy nexrl/ --ignore-missing-imports

# Format
python -m black nexrl/ tests/
python -m isort nexrl/ tests/

# License header fix
python tests/lint/fix_license_header.py <files>
```

## Skills

Available Claude skills in `.claude/skills/`:
- **fix-issue** — Worktree-based GitHub issue fixing
- **github-pr** — PR creation from worktree branches
- **git-commit** — Commit workflow with linting/testing
- **code-review** — Python code review
- **testing** — pytest and linting
- **address-pr-comments** — Handle PR review comments

## Agents

- `.claude/agents/code-review/AGENT.md` — Automated code review
- `.claude/agents/testing/AGENT.md` — Automated testing

## Rules

1. **Never add AI co-author lines** to commits
2. **Never create example/test files** outside `tests/` unless asked
3. **Always update docs** when changing documented behavior
4. **Follow documentation-first** approach — read docs before coding
5. **Use git worktree** for all issue work (multi-agent isolation)
6. **License headers** required on Python, shell, YAML, TOML files
