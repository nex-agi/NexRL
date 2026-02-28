---
name: fix-issue
description: Fix a GitHub issue using git worktree for isolation. Fetches issue, creates worktree, implements fix, then creates PR.
---

# Fix Issue Workflow (Worktree-Based)

## Workflow

### Step 1: Verify Prerequisites

```bash
gh auth status
```

### Step 2: Fetch Issue

```bash
ISSUE_NUM=<number>
gh issue view $ISSUE_NUM --repo "${REPO_OWNER}/${REPO_NAME}" --json number,title,body,state,labels
```

If closed, confirm with user before proceeding.

### Step 3: Update Issue Status

```bash
gh issue comment $ISSUE_NUM --repo china-qijizhifeng/NexRL \
  --body "🤖 Agent picking up this issue. Creating worktree branch \`agent/issue-${ISSUE_NUM}\`."

gh issue edit $ISSUE_NUM --repo china-qijizhifeng/NexRL \
  --add-label "status:in-progress" 2>/dev/null || true
```

### Fetch Shared Knowledge

Clone or update the shared knowledge repo for common scripts and docs.

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
SHARED_KNOWLEDGE="${REPO_ROOT}/../nex-taas-shared-knowledge"
if [ ! -d "$SHARED_KNOWLEDGE" ]; then
  git clone https://github.com/china-qijizhifeng/nex-taas-shared-knowledge.git "$SHARED_KNOWLEDGE"
else
  git -C "$SHARED_KNOWLEDGE" pull --ff-only 2>/dev/null || true
fi
```

### Step 4: Create Worktree

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
git fetch origin

BRANCH="agent/issue-${ISSUE_NUM}"
WORKTREE_DIR="${REPO_ROOT}/../worktrees/NexRL-issue-${ISSUE_NUM}"

git worktree add "$WORKTREE_DIR" -b "$BRANCH" "origin/main"
cd "$WORKTREE_DIR"
```

### Step 5: Read Documentation First

Before coding, read relevant docs per `.ai-instructions/developing/documentation.md`:
- `docs/developer-guide/` for architecture understanding
- Component-specific docs for the area being changed

### Step 6: Plan the Fix

Enter plan mode. Consider:
- Root cause analysis (for bugs)
- Files that need changes
- Impact on other components
- Documentation updates needed

### Step 7: Implement

Work in the worktree directory. Follow:
- `.ai-instructions/developing/` conventions
- Run `python -m black` and `python -m isort` on changed files
- Ensure license headers present

### Step 8: Test

Run project tests (use `testing` skill):
```bash
python -m pytest tests/ -v
python -m pylint nexrl/ --rcfile=.pylintrc -rn -sn
```

### Step 9: Commit and PR

Use `git-commit` skill, then `github-pr` skill.
PR body should include: `Fixes #ISSUE_NUM`

### Step 10: Handle Blockers

If blocked:
```bash
gh issue edit $ISSUE_NUM --repo china-qijizhifeng/NexRL \
  --remove-label "status:in-progress" --add-label "status:blocked"

gh issue comment $ISSUE_NUM --repo china-qijizhifeng/NexRL --body "
🚧 Blocked: <describe the problem>
Need guidance on: <specific question>
Progress so far: <what's been done>"
```

### Step 11: Cleanup (after PR merged)

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
git worktree remove "$WORKTREE_DIR"
git branch -D "$BRANCH"
```
