# Pre-Commit Review Checklist

## Overview

This checklist ensures code quality and consistency about committing changes to the PyPTO project.

### Code Quality

- [ ] Code follows project style and conventions
- [ ] No debug code or commented-out sections left in
- [ ] No TODOs or FIXMEs unless documented

### Documentation Alignment

- [ ] Documentation accurately reflects code changes
- [ ] Examples in docs still work
- [ ] C++ implementation matches Python bindings
- [ ] Type stubs (`.pyi`) match actual API

### Commit Content

- [ ] Only relevant changes included (no unrelated edits)
- [ ] No accidental commits of build artifacts
- [ ] No sensitive information (passwords, tokens, etc.)
- [ ] Commit message clearly explains the change

### Cross-Layer Consistency

## Summary

**Review thoroughly before committing!**

Taking time to verify these items prevents bugs, maintains code quality, and keeps the codebase consistent across all layers.
