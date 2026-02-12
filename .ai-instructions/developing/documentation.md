# Documentation Workflow

## Overview

NexRL follows a **documentation-first development** approach. This ensures that code and documentation remain aligned, making the codebase easier to understand and maintain.

## Core Principles

### 1. Read Documentation Before Coding

**Always read relevant documentation before making any code changes.**

Key documentation files:
- **[`docs/developer-guide/README.md`](../../docs/developer-guide/README.md)** - Developer guide overview

- **Core Architecture Documentation:**
  - [`docs/developer-guide/02-core-architecture/overview.md`](../../docs/developer-guide/02-core-architecture/overview.md) - System architecture overview
  - [`docs/developer-guide/02-core-architecture/controller.md`](../../docs/developer-guide/02-core-architecture/controller.md) - Main controller implementation
  - [`docs/developer-guide/02-core-architecture/data-types.md`](../../docs/developer-guide/02-core-architecture/data-types.md) - Core data structures
  - [`docs/developer-guide/02-core-architecture/activity-tracking.md`](../../docs/developer-guide/02-core-architecture/activity-tracking.md) - Activity tracking system

- **Component Documentation:**
  - [`docs/developer-guide/05-rollout-workers/`](../../docs/developer-guide/05-rollout-workers/) - Rollout worker implementations
  - [`docs/developer-guide/06-trainers/`](../../docs/developer-guide/06-trainers/) - Trainer implementations
  - [`docs/developer-guide/07-services/`](../../docs/developer-guide/07-services/) - Service layer documentation

- **Configuration and Setup:**
  - [`docs/developer-guide/01-getting-started/`](../../docs/developer-guide/01-getting-started/) - Getting started guides
  - [`docs/developer-guide/11-configuration-reference/`](../../docs/developer-guide/11-configuration-reference/) - Complete configuration reference

- **Other docs in [`docs/developer-guide/`](../../docs/developer-guide/)** - Check for additional relevant documentation

### 2. Review Documentation After Each Edit

**After making any code changes, review the documentation to ensure alignment.**

Ask yourself:
- Does the documentation still accurately describe how the code works?
- Are there any examples in the docs that might be affected?
- Did I change any APIs or behavior that's documented?
- Are field descriptors and reflection behavior still accurate?

### 3. Update Documentation When Needed

**If your code changes affect documented behavior, update the documentation.**

Update docs when you:
- Modify core architecture or add new components
- Change API signatures or usage patterns
- Add, remove, or modify rollout workers, trainers, or services
- Change configuration schemas or parameters
- Modify data structures or communication protocols
- Update deployment procedures or distributed execution behavior

## Documentation Structure

```
docs/
├── developer-guide/
│   ├── 01-getting-started/       # Quick start and setup guides
│   ├── 02-core-architecture/     # System architecture docs
│   ├── 03-data-loader/           # Data loading documentation
│   ├── 04-trajectory-pool/       # Trajectory pool implementation
│   ├── 05-rollout-workers/       # Rollout worker guides
│   ├── 06-trainers/              # Trainer implementation docs
│   ├── 07-services/              # Service layer documentation
│   ├── 08-features/              # Feature documentation
│   ├── 09-recipes/               # Recipe configuration guides
│   ├── 10-distributed-execution/ # Distributed execution with Ray
│   ├── 11-configuration-reference/ # Configuration schemas
│   └── 12-best-practices/        # Best practices and patterns
├── user-guide.md                 # User guide
└── README-CN.md                  # Chinese documentation
```

## Documentation Style Guide

When updating documentation:

1. **Use clear, descriptive headings** - Make it easy to scan
2. **Provide code examples** - Show Python usage and configuration examples
3. **Explain the "why"** - Don't just describe what code does, explain design decisions
4. **Use diagrams** - For complex architectures and data flows
5. **Keep examples working** - Test that examples actually run
6. **Link between docs** - Reference related documentation files
7. **Maintain consistency** - Follow the existing documentation style
8. **Include configuration examples** - Show relevant YAML/Python config snippets
9. **Keep the docs short and easy to read** - Each doc should usually be less than 300 lines


## Common Documentation Tasks

### Adding a New Component (Worker, Trainer, Service)

1. Read relevant component documentation in `docs/developer-guide/`
2. Implement the new component in Python
3. Update the component overview section
4. Add configuration examples
5. Provide usage examples and integration patterns
6. Update architecture diagrams if needed

### Modifying Existing Component

1. Read current documentation for that component
2. Make code changes
3. Update component description and examples
4. Verify all examples still work
5. Update configuration reference if parameters changed
6. Update any affected sections in other docs

### Changing Configuration Schema

1. Update configuration validation code
2. Update `docs/developer-guide/11-configuration-reference/`
3. Update affected recipe examples in `docs/developer-guide/09-recipes/`
4. Test that configuration examples are valid

## Documentation Quality Checklist

Before finalizing changes, verify:

- [ ] Code matches documented behavior
- [ ] All code examples in docs are valid and tested
- [ ] Python implementation matches documented API
- [ ] Configuration examples are valid and tested
- [ ] Links between documentation files are correct
- [ ] No broken references to moved/renamed files
- [ ] Examples use current API (no deprecated patterns)
- [ ] Documentation follows existing style and formatting
- [ ] Architecture diagrams are up-to-date if components changed

## Remember

**Good documentation is as important as good code.**

Documentation is the primary way developers understand the system. Keeping it accurate and up-to-date prevents bugs, reduces confusion, and makes the codebase more maintainable.

When in doubt, update the docs!
