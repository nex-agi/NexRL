# Commit Guidelines

## Overview

This document defines commit message standards and authorship practices for the NexRL project to maintain clear project history and proper attribution.

## Commit Message Format

Follow this structure for commit messages:

```
Brief summary in present tense (max 72 characters)

Optional detailed explanation of:
- Why this change is needed
- What problem it solves
- Any important context or trade-offs

Optional references to issues or docs
```

## Summary Line Rules

1. **Use present tense** - "Add feature" not "Added feature"
2. **Be concise** - Keep under 72 characters
3. **Start with verb** - Add, Update, Fix, Remove, Refactor, etc.
4. **No period at end** - Don't end with punctuation
5. **Focus on "why"** - Not just "what" was changed

## Good Summary Line Examples

```
Add support for custom rollout workers with distributed execution

Update trainer configuration docs with new parameters

Fix inconsistent trajectory batching in data loader

Refactor service holder to support dynamic worker allocation

Remove deprecated checkpoint format handling code

Optimize trajectory pool memory usage for large buffers
```

## Bad Summary Line Examples

```
❌ updated files              # Past tense, vague
❌ fixes                      # Too brief, unclear
❌ Add new feature.           # Has period
❌ Changed some code in the trainer system to make it better
                             # Too long, vague
```

## Detailed Body (Optional)

When needed, add a blank line after summary and provide details:

```
Fix trajectory batching inconsistency in DataLoader

The batching logic was not properly handling variable-length
trajectories when using dynamic padding. This caused errors
when processing batches with diverse episode lengths.

Updates docs/developer-guide/03-data-loader/data-loader.md to
clarify the expected behavior for dynamic batching.
```

## Co-Author Policy

### ❌ Never Add AI Assistants as Co-Authors

**Do NOT use `Co-authored-by:` tags for any AI assistants:**

```
❌ Co-authored-by: Claude <claude@anthropic.com>
❌ Co-authored-by: ChatGPT <gpt@openai.com>
❌ Co-authored-by: Cursor AI <ai@cursor.com>
❌ Co-authored-by: GitHub Copilot <copilot@github.com>
```

### Why?

- AI assistants are **tools**, not collaborators
- Commits should reflect **human authorship** and decision-making
- AI cannot take responsibility for code quality or bugs
- AI doesn't understand the full context of design decisions
- Proper attribution is important for accountability

### ✅ Only Credit Human Contributors

Only add co-author tags for actual human collaborators:

```
✅ Co-authored-by: John Doe <john@example.com>
✅ Co-authored-by: Jane Smith <jane@example.com>
```

## Code review

Before committing, please review the code following instructions in `.ai-instructions/code-review/*`

## Summary

**Key Reminders:**
- Write clear, present-tense commit messages
- Never add AI assistants as co-authors
- Focus commit messages on "why" not just "what"

Good commit messages make the project history useful and maintainable for everyone!
