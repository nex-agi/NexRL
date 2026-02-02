# Configuration Migration Guide

## Overview

This guide documents the configuration structure changes introduced in NexRL v2.0 and provides instructions for migrating from the legacy configuration format to the new format.

**Current Status**: Full backward compatibility is maintained. Legacy configurations will continue to work with deprecation warnings.

**Target Removal Date**: Legacy support planned for removal in v3.0 (TBD)

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
```

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

The system now includes comprehensive validation that:
1. ✅ Accepts both old and new formats
2. ⚠️ Shows deprecation warnings for old formats
3. ❌ Errors on ambiguous configurations

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
ValueError: Multiple train services found without 'role' field: ['service1', 'service2'].
Cannot determine which is 'actor'. Please add explicit 'role' field to each service.
```

```
ValueError: Exactly one train_service must have role='actor', found 2
```

---

## Testing Your Migration

### Step-by-Step Validation

1. **Backup your config**
   ```bash
   cp recipe/my_recipe.yaml recipe/my_recipe.yaml.backup
   ```

2. **Run validation**
   ```bash
   python scripts/validate_recipe.py recipe/my_recipe.yaml
   ```

3. **Check for warnings**
   - Look for `DeprecationWarning` in output
   - Note which fields need updating

4. **Update configuration**
   - Follow migration instructions above
   - Update one section at a time

5. **Re-validate**
   - Run validation again
   - Confirm no deprecation warnings

6. **Test training**
   ```bash
   bash cli/self_hosted/train.sh recipe/my_recipe.yaml
   ```

7. **Monitor logs**
   - Check for any runtime warnings
   - Confirm training starts successfully

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

## Removal Instructions for Maintainers

When it's time to remove backward compatibility support (v3.0):

### 1. Remove Fallback Code

**Files to update:**
- `scripts/common/config_utils.py` - Remove old `resource.agent` fallback
- `scripts/self_hosted/config_utils.py` - Remove old `resource.train` and `resource.inference` fallbacks
- `nexrl/utils/config_utils.py` - Remove flat structure and role detection fallbacks
- `nexrl/utils/validate_config.py` - Remove old field validation fallbacks
- `nexrl/inference_service_client/openai_inference_service_client.py` - Remove `model_tag` fallback
- `nexrl/inference_service_client/remote_api_inference_service_client.py` - Remove `model_tag` fallback
- `nexrl/mock/mock_inference_service_client.py` - Remove `model_tag` fallback
- `nexrl/weight_sync/weight_sync_controller.py` - Remove `model_tag` fallback
- `nexrl/trainer/base_trainer.py` - Remove `model_tag` fallback

### 2. Update Validation to Be Strict

In `nexrl/utils/validate_config.py`:

```python
# Remove backward compatibility checks
# Change warnings to errors
# Require new format fields

# Example:
assert config.service.inference_service.get("identifier"), \
    "service.inference_service.identifier is required"

# No more model_tag fallback
```

### 3. Search and Remove

Search for these patterns and remove:
```bash
# Search for deprecation warnings
grep -r "DeprecationWarning" nexrl/
grep -r "deprecated" scripts/

# Search for model_tag references
grep -r "model_tag" nexrl/
grep -r "model_tag" scripts/

# Search for old resource paths
grep -r "resource.train" scripts/
grep -r "resource.inference" scripts/
grep -r "resource.agent" scripts/
```

### 4. Update Tests

- Remove tests for legacy format
- Add tests to ensure legacy format fails with clear error messages
- Update all test configs to new format

### 5. Update Documentation

- Remove migration guide (or archive it)
- Update all examples to use new format only
- Add clear error messages when old format is detected

### 6. Create Migration Error Messages

When old format is detected in v3.0, show helpful errors:

```python
if "model_tag" in config.service.inference_service:
    raise ValueError(
        "The 'model_tag' field is no longer supported as of v3.0. "
        "Please use 'identifier' instead. "
        "See migration guide at: https://docs.nexrl.ai/migration-v3"
    )
```

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

This migration maintains full backward compatibility while preparing for a cleaner, more flexible configuration structure. Take your time to migrate, test thoroughly, and reach out if you encounter any issues.

**Recommended Action**: Start migrating your configurations now to stay ahead of the v3.0 changes.
