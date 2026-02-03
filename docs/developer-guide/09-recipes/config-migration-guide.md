# Configuration Migration Guide

## Overview

This guide documents the configuration structure changes introduced in NexRL v2.0 and provides instructions for migrating from the legacy configuration format to the new format.

**Current Status**: Full backward compatibility is maintained. Legacy configurations will continue to work with deprecation warnings.

**Target Removal Date**: Legacy support planned for removal in v3.0 (TBD)

### 🎯 Key Feature: Centralized Migration

All backward compatibility is handled by **just 3 migration functions**:
- `nexrl/utils/config_utils.py::migrate_legacy_config()` - for the library
- `scripts/common/config_utils.py::migrate_legacy_config()` - for scripts
- `cli/common/config_utils.py::migrate_legacy_config()` - for CLI

**To remove in v3.0**: Delete these 3 functions + 3 call sites. That's it! ✨

See [Implementation Architecture](#implementation-architecture) for details.

---

## Summary of Changes

The configuration structure has been refactored to support:
- Multiple train service groups (e.g., student + teacher for on-policy distillation)
- Clearer service role identification
- Better separation of concerns between resource and service configs

### Quick Reference

| Category | Old Path | New Path | Status |
|----------|----------|----------|--------|
| Inference identifier | `service.inference_service.model_tag` | `service.inference_service.identifier` | ⚠️ Deprecated |
| Train identifier | `service.train_service.model_tag` | `service.train_service.<name>.identifier` | ⚠️ Deprecated |
| Train resources | `resource.train.<id>.world_size` | `service.train_service.<name>.resource.world_size` | ⚠️ Deprecated |
| Inference resources | `resource.inference.*` | `service.inference_service.resource.*` | ⚠️ Deprecated |
| Agent resources | `resource.agent.*` | `rollout_worker.resource.*` | ⚠️ Deprecated |
| Train service role | N/A | `service.train_service.<name>.role` | ⚠️ Optional (auto-detected) |

---

## Detailed Migration Instructions

### 1. Field Rename: `model_tag` → `identifier`

**Change**: The `model_tag` field has been renamed to `identifier` for clarity.

#### Old Configuration
```yaml
service:
  inference_service:
    model_tag: "default"
    # ...

  train_service:
    model_tag: "default"
    # ...
```

#### New Configuration
```yaml
service:
  inference_service:
    identifier: "default"
    # ...

  train_service:
    my_actor:  # Service name (new level)
      identifier: "default"
      role: "actor"
      # ...
```

#### Backward Compatibility
- ✅ Old `model_tag` field is still supported
- ⚠️ Deprecation warning will be shown
- 📅 Planned removal: v3.0

#### Migration Steps
1. Rename `model_tag` to `identifier` in `service.inference_service`
2. Rename `model_tag` to `identifier` in `service.train_service`
3. Test your configuration
4. Check logs for any remaining deprecation warnings

---

### 2. Train Service Structure: Flat → Nested

**Change**: Train service configuration now supports multiple service groups with explicit names and roles.

#### Old Configuration (Flat Structure)
```yaml
service:
  train_service:
    model_tag: "default"
    backend: http
    url: "http://localhost:8000"
    world_size: 8
    # ... other config
```

#### New Configuration (Nested Structure)
```yaml
service:
  train_service:
    my_actor:  # Service group name (you choose)
      identifier: "default"
      role: "actor"
      backend: http
      url: "http://localhost:8000"
      resource:
        world_size: 8
        gpus_per_pod: 8
      # ... other config
```

#### Backward Compatibility
- ✅ Old flat structure is still supported
- ⚠️ Deprecation warning will be shown
- ⚠️ Only works with single train service (no multi-service setups)
- 📅 Planned removal: v3.0

#### Migration Steps
1. Create a service group name (e.g., `my_actor`, `student`, etc.)
2. Move all train_service fields under this group name
3. Add `role: "actor"` field
4. Move resource-related fields under `resource:` section
5. Update any scripts that reference train_service config

---

### 3. Role Field Requirement

**Change**: Each train service group should explicitly specify its `role`.

#### Old Configuration
```yaml
service:
  train_service:
    # No role field
    backend: http
    # ...
```

#### New Configuration
```yaml
service:
  train_service:
    my_actor:
      role: "actor"  # Explicit role
      backend: http
      # ...
```

#### Backward Compatibility
- ✅ If only one train service exists, role="actor" is assumed
- ⚠️ Multiple services without roles will cause an error
- ⚠️ Deprecation warning will be shown for missing role
- 📅 Planned removal: v3.0

#### Valid Roles
- `actor`: Primary training service (required for all setups)
- `teacher`: Teacher model for distillation (optional, for OPD setups)
- Custom roles: Can be added for advanced use cases

#### Migration Steps
1. Add `role: "actor"` to your primary train service
2. If using multiple services (e.g., OPD), specify roles for each
3. Exactly one service must have `role: "actor"`

---

### 4. Resource Configuration Restructuring

**Change**: Resource configurations have been moved from top-level `resource` section to service-specific locations.

#### 4a. Train Resources

**Old:**
```yaml
resource:
  train:
    student:  # identifier
      world_size: 1
    teacher:
      world_size: 1
```

**New:**
```yaml
service:
  train_service:
    student_service:
      identifier: "student"
      role: "actor"
      resource:
        world_size: 1
        gpus_per_pod: 8
    teacher_service:
      identifier: "teacher"
      role: "teacher"
      resource:
        world_size: 1
        gpus_per_pod: 8
```

#### 4b. Inference Resources

**Old:**
```yaml
resource:
  inference:
    replicas: 4
    gpus_per_replica: 2
    served_model_name: "model-name"
    model_path: "/path/to/model"
    backend: "bp-sglang"
    extra_args: ""
    # ... any custom fields
```

**New:**
```yaml
service:
  inference_service:
    model: "model-name"
    model_path: "/path/to/model"
    resource:
      replicas: 4
      gpus_per_replica: 2
      backend: "bp-sglang"
      extra_args: ""
      # ... all other fields copied automatically
```

**Note**: ALL fields from `resource.inference` are automatically migrated to `inference_service.resource` (except `model_path` and `served_model_name` which go to top level).

#### 4c. Agent/Rollout Worker Resources

**Old:**
```yaml
resource:
  agent:
    num_workers: 1
    agents_per_worker: 32
```

**New:**
```yaml
rollout_worker:
  resource:
    num_workers: 1
    agents_per_worker: 32
```

#### Backward Compatibility
- ✅ Old `resource.*` structures are still supported
- ✅ **Smart identifier matching**: If there's only one train service and one resource entry, they're automatically matched even with different identifiers
- ⚠️ Deprecation warnings will be shown
- 📅 Planned removal: v3.0

---

## Complete Migration Example

### Before (Legacy Format)

```yaml
resource:
  train:
    default:
      world_size: 8
  inference:
    replicas: 4
    gpus_per_replica: 2
    served_model_name: "Qwen/Qwen3-8B"
    model_path: "/models/qwen3-8b"
    backend: "bp-sglang"
  agent:
    num_workers: 32
    agents_per_worker: 1

service:
  inference_service:
    model_tag: "default"
    api_key: "EMPTY"
    base_url: "http://localhost:8001"
    model: "Qwen/Qwen3-8B"
    max_tokens: 2048
    backend: "openai"
    weight_type: "sglang_nckpt"

  train_service:
    model_tag: "default"
    backend: http
    world_size: 8
    url: "http://localhost:8000"
```

### After (New Format)

```yaml
rollout_worker:
  resource:
    num_workers: 32
    agents_per_worker: 1

service:
  inference_service:
    identifier: "default"
    api_key: "EMPTY"
    base_url: "http://localhost:8001"
    model: "Qwen/Qwen3-8B"
    model_path: "/models/qwen3-8b"
    max_tokens: 2048
    backend: "openai"
    weight_type: "sglang_nckpt"
    resource:
      replicas: 4
      gpus_per_replica: 2
      backend: "bp-sglang"

  train_service:
    main_actor:
      identifier: "default"
      role: "actor"
      backend: http
      url: "http://localhost:8000"
      resource:
        world_size: 1
        gpus_per_pod: 8
```

---

## Advanced: Multi-Service Setup (On-Policy Distillation)

The new structure enables multiple train service groups, which is required for advanced training methods like on-policy distillation (OPD).

### Example: Student + Teacher Setup

```yaml
service:
  train_service:
    student_service:
      identifier: "student"
      role: "actor"
      backend: http
      url: "http://localhost:8000"
      resource:
        world_size: 1
        gpus_per_pod: 8
      actor:
        model_path: "/models/student"
        # ... student config

    teacher_service:
      identifier: "teacher"
      role: "teacher"
      backend: http
      url: "http://localhost:8000"
      resource:
        world_size: 1
        gpus_per_pod: 8
      actor:
        model_path: "/models/teacher"
        # ... teacher config
```

---

## Validation and Warnings

### Current Behavior

The system now includes comprehensive **automatic migration** that:
1. ✅ Automatically converts old format to new format at runtime
2. ✅ Accepts both old and new formats seamlessly
3. ⚠️ Shows deprecation warnings for old formats
4. ❌ Errors on ambiguous configurations

**No manual intervention needed** - old configs work automatically while you plan your migration!

### Warning Examples

```
DeprecationWarning: The 'model_tag' field is deprecated.
Please use 'identifier' instead.
See migration guide in docs/developer-guide/09-recipes/.
```

```
DeprecationWarning: The 'resource.train' config structure is deprecated.
Please migrate to the new 'service.train_service' structure.
See documentation for migration guide.
```

```
DeprecationWarning: Train service 'my_service' is missing 'role' field.
Assuming role='actor'. Please add explicit role='actor' field.
```

### Error Examples

```
ValueError: Cannot auto-migrate: found 'backend' at both top-level and nested levels in train_service.
This configuration is ambiguous. Please manually update to the new format.
```

```
ValueError: Cannot auto-migrate resource.train: identifier mismatch.
Service 'main_actor' has identifier 'default', but resource.train has keys: ['service1', 'service2'].
Please manually update identifiers to match.
```

```
ValueError: Multiple train services found without 'role' field: ['service1', 'service2'].
Cannot determine which is 'actor'. Please add explicit 'role' field to each service.
```

---

## Testing Your Migration

### Important: Old Configs Work Automatically!

Your old configs will work **without any changes** thanks to automatic migration. However, we recommend updating to the new format to prepare for v3.0.

### Step-by-Step Validation

1. **Test with existing config (no changes needed)**
   ```bash
   # Your old config works as-is!
   bash cli/self_hosted/train.sh recipe/my_old_recipe.yaml
   ```

   **Check console output** for deprecation warnings showing what needs updating.

2. **Backup your config**
   ```bash
   cp recipe/my_recipe.yaml recipe/my_recipe.yaml.backup
   ```

3. **Update configuration incrementally**
   - Follow migration instructions above
   - Update one section at a time
   - Old and new formats can coexist during migration

4. **Test after each change**
   ```bash
   python scripts/validate_recipe.py recipe/my_recipe.yaml
   bash cli/self_hosted/train.sh recipe/my_recipe.yaml
   ```

5. **Verify no warnings**
   - Run training one more time
   - Confirm **no** deprecation warnings in console
   - Training should work identically

### Zero-Downtime Migration

Thanks to automatic migration:
- ✅ Continue using old configs while planning migration
- ✅ Migrate services one at a time
- ✅ Roll back anytime by reverting config changes
- ✅ No service interruption needed

---

## Planned Deprecation Timeline

### Version 2.0 (Current)
- ✅ Full backward compatibility
- ⚠️ Deprecation warnings for legacy formats
- 📖 Migration guide available

### Version 2.x (Future Minor Releases)
- ⚠️ Increased warning visibility
- 📖 Updated documentation
- 🔧 Migration tools/scripts (if needed)

### Version 3.0 (Planned)
- ❌ Legacy formats removed
- ✅ Only new format supported
- 📖 Legacy configs will fail validation

**Recommendation**: Migrate to new format as soon as possible to avoid disruption when v3.0 is released.

---

## Implementation Architecture

### Centralized Migration Functions

All backward compatibility is handled by **three migration functions** - one for each deployment context. This centralized approach makes the codebase clean and easy to maintain:

#### 1. **NexRL Library** - `nexrl/utils/config_utils.py`
```python
def migrate_legacy_config(config: DictConfig):
    """Centralized migration of all legacy config structures.

    Handles ALL backward compatibility transformations:
    1. model_tag → identifier
    2. resource.train → service.train_service.*.resource
    3. resource.inference → service.inference_service.resource (all fields)
    4. resource.agent → rollout_worker.resource
    5. Flat train_service → nested with role
    6. Add missing role fields

    Smart features:
    - Auto-matches single service with single resource (even with different identifiers)
    - Copies all resource fields dynamically (future-proof)
    - Detects flat vs nested by backend location (robust)

    To remove backward compatibility: delete this function and its call.
    """
```

**Called from**: `main.py::main_task()` before creating controller

#### 2. **Deployment Scripts** - `scripts/common/config_utils.py`
```python
def migrate_legacy_config(cfg: dict) -> dict:
    """Centralized migration for deployment scripts.

    Same transformations as library version.
    To remove backward compatibility: delete this function and its calls.
    """
```

**Called from**: `load_config()` when loading YAML configs

#### 3. **CLI Tools** - `cli/common/config_utils.py`
```python
def migrate_legacy_config(cfg: dict) -> dict:
    """Centralized migration for CLI tools.

    Same transformations as library version.
    To remove backward compatibility: delete this function and its calls.
    """
```

**Called from**: Config loading in CLI scripts

### Benefits of This Architecture

| Benefit | Impact |
|---------|--------|
| **Single Source of Truth** | All migrations in 3 clearly marked functions |
| **Easy Removal** | Delete 3 functions + 3 call sites = done |
| **Clean Codebase** | Rest of code uses only new format |
| **Maintainable** | Add/remove migrations in one place |
| **Testable** | Easy to test migration logic independently |

---

## Removal Instructions for Maintainers

When it's time to remove backward compatibility support (v3.0), the process is extremely simple thanks to centralized migration functions:

### Step 1: Delete Migration Functions (3 files)

**Delete these functions:**

1. **`nexrl/utils/config_utils.py`** (~lines 182-380)
   ```python
   def migrate_legacy_config(config: DictConfig):
       # DELETE this entire function
   ```

2. **`scripts/common/config_utils.py`** (~lines 28-200)
   ```python
   def migrate_legacy_config(cfg: dict) -> dict:
       # DELETE this entire function
   ```

3. **`cli/common/config_utils.py`** (~lines 28-200)
   ```python
   def migrate_legacy_config(cfg: dict) -> dict:
       # DELETE this entire function
   ```

### Step 2: Remove Function Calls (3 locations)

**Remove these calls:**

1. **`nexrl/main.py::main_task()`**
   ```python
   # DELETE this line (around line 35)
   migrate_legacy_config(config)
   ```

2. **`scripts/common/config_utils.py::load_config()`**
   ```python
   # DELETE this line (around line 239)
   cfg_dict = migrate_legacy_config(cfg_dict)
   ```

3. **`cli/common/config_utils.py::load_config()`**
   ```python
   # DELETE the migration call in load_config()
   cfg_dict = migrate_legacy_config(cfg_dict)
   ```

### Step 3: Update Validation (Optional Strict Mode)

In `nexrl/utils/validate_config.py`, optionally make validation stricter:

```python
# Change from accepting both to requiring new format only
assert config.service.inference_service.get("identifier"), \
    "service.inference_service.identifier is required (model_tag no longer supported)"

# Add helpful error for old format
if "model_tag" in config.service.inference_service:
    raise ValueError(
        "The 'model_tag' field is no longer supported as of v3.0. "
        "Please rename to 'identifier'. "
        "See migration guide at: https://docs.nexrl.ai/migration-v3"
    )
```

### Step 4: Update Tests

- Remove tests for legacy format compatibility
- Add tests to ensure legacy format fails with clear error messages
- Verify all test configs use new format

### Step 5: Update Documentation

- Archive this migration guide (move to `docs/archive/`)
- Update all examples to use new format only
- Add v3.0 breaking changes notice

### That's It!

The entire codebase now uses only the new format. No scattered backward compatibility code to hunt down!

### Quick Verification

After removal, search to confirm no legacy code remains:

```bash
# Should find nothing (except in archived docs)
grep -r "migrate_legacy_config" nexrl/ scripts/ cli/
grep -r "model_tag.*backward" nexrl/
grep -r "resource.train.*deprecated" scripts/
```

### Estimated Effort

- **Time**: ~30 minutes
- **Files Changed**: 3-6 files
- **Lines Removed**: ~300-400 lines
- **Risk**: Low (all changes localized)

Compare this to the old approach which would require:
- ❌ ~15+ files to modify
- ❌ ~50+ scattered locations to find and remove
- ❌ High risk of missing hidden fallback code
- ❌ 4-8 hours of work

---

## Getting Help

### Documentation Resources
- [Recipe Configuration](./recipe-configuration.md) - Full config reference
- [Recipe Structure](./recipe-structure.md) - Understanding recipe organization
- [Environment Setup](./environment-setup.md) - Environment configuration

### Support Channels
- GitHub Issues: Report migration problems
- Documentation: Check latest updates
- Example Recipes: See `recipe/` directory for updated examples

### Common Issues

**Issue**: "Multiple train services without role"
- **Solution**: Add `role: "actor"` to your train service

**Issue**: "Cannot find identifier"
- **Solution**: Rename `model_tag` to `identifier`

**Issue**: "Resource config validation failed"
- **Solution**: Move resource fields under `resource:` subsection

**Issue**: Scripts can't find worker logs
- **Solution**: Update log paths to new format: `workers-{identifier}-rank{rank}.log`

---

## Conclusion

This migration system provides **automatic backward compatibility** while preparing for a cleaner, more flexible configuration structure. Your old configs continue to work seamlessly with just deprecation warnings.

### Key Takeaways

✅ **No Immediate Action Required**: Old configs work automatically
✅ **Centralized Migration**: All compatibility in 3 easy-to-remove functions
✅ **Zero Downtime**: Migrate at your own pace
✅ **Simple Removal**: Delete 3 functions when ready for v3.0

**Recommended Action**: While old configs work automatically, we recommend migrating to the new format when convenient to stay ahead of v3.0 changes.

---

## Questions or Issues?

- 📖 Review examples in `recipe/` directory
- 🐛 Report issues on GitHub
- 💬 Check console warnings for specific guidance

**The system will guide you** - just run your old config and check the deprecation warnings for what to update!
