---
name: address-pr-comments
description: Address GitHub PR review comments. Navigate to correct worktree, make fixes, push updates.
---

# Address PR Comments Workflow (Worktree-Aware)

## Workflow

### Step 1: Find the Worktree

```bash
PR_NUM=<number>
BRANCH=$(gh pr view $PR_NUM --repo "$REPO" --json headRefName -q '.headRefName')

# Find and enter the worktree
git worktree list | grep "$BRANCH"
cd <worktree-path>
```

### Step 2: Fetch Unresolved Comments

```bash
gh api graphql -f query='
query {
  repository(owner: "china-qijizhifeng", name: "NexRL") {
    pullRequest(number: '$PR_NUM') {
      reviewThreads(first: 50) {
        nodes {
          id isResolved
          comments(first: 1) {
            nodes { id databaseId body path line }
          }
        }
      }
    }
  }
}'
```

### Step 3: Classify & Address

- **A: Actionable** → Make code changes
- **B: Discussable** → Ask user for guidance
- **C: Informational** → Resolve with acknowledgment

### Step 4: Format, Lint & Test

```bash
python -m black <changed_files>
python -m isort <changed_files>
python -m pylint <changed_files> --rcfile=.pylintrc -rn -sn
python -m pytest tests/ -v --tb=short
```

### Step 5: Commit & Push

```bash
git add -A
git commit -m "chore(pr): address review comments for #${PR_NUM}"
git push origin "$BRANCH"
```

### Step 6: Reply & Resolve Threads

Reply to each addressed comment via gh API, then resolve the thread.

### Step 7: Update Issue Status

```bash
ISSUE_NUM=$(echo "$BRANCH" | grep -oP 'issue-\K\d+')
if [ -n "$ISSUE_NUM" ]; then
  gh issue comment "$ISSUE_NUM" --repo china-qijizhifeng/NexRL \
    --body "📝 PR comments addressed and pushed."
fi
```
