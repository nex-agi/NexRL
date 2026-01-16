# Error Handling

## Overview

`ErrorReporter` provides centralized error tracking and health monitoring for NexRL. It collects errors from all modules, maintains error history, and provides health status queries.

**Location**: `nexrl/error_reporter.py`

## Key Concepts

### Centralized Error Tracking

All modules report errors to a single `ErrorReporter` instance:

```
Module A → ErrorReporter ← Module B
            ↓
      Health Status Query
```

### Error Severity Levels

```python
class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
```

- **INFO**: Informational events (not failures)
- **WARNING**: Potential issues that don't stop execution
- **ERROR**: Actual errors requiring attention

## ErrorInfo

Data structure containing error details:

```python
@dataclass
class ErrorInfo:
    error_id: str                      # Unique identifier (UUID)
    timestamp: float                   # When error occurred
    module_name: str                   # Module reporting error
    work_context: str                  # What was being done
    severity: ErrorSeverity            # Severity level
    message: str                       # Error message
    details: dict[str, Any]            # Additional context
    exception_type: str | None         # Exception class name
    exception_traceback: str | None    # Full traceback
```

## Core Methods

### Constructor

```python
def __init__(self, max_errors: int = 1000)
```

Initialize error reporter with maximum error history size.

**Parameters**:
- `max_errors`: Maximum errors to keep in memory (oldest removed when exceeded)

### report_exception

```python
def report_exception(
    self,
    module_name: str,
    work_context: str,
    exception: Exception,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    details: dict[str, Any] | None = None
) -> str
```

Report an exception error.

**Parameters**:
- `module_name`: Name of module where error occurred
- `work_context`: Description of work being performed
- `exception`: The exception object
- `severity`: Error severity level
- `details`: Additional context information

**Returns**: Unique error ID (UUID string)

**Example**:

```python
try:
    result = process_data(item)
except ValueError as e:
    error_id = error_reporter.report_exception(
        module_name="rollout_worker",
        work_context="processing_data_item",
        exception=e,
        severity=ErrorSeverity.ERROR,
        details={"item_id": item.get("id")}
    )
```

### report_error

```python
def report_error(
    self,
    module_name: str,
    work_context: str,
    message: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    details: dict[str, Any] | None = None
) -> str
```

Report a non-exception error.

**Parameters**:
- `module_name`: Name of module where error occurred
- `work_context`: Description of work being performed
- `message`: Error message
- `severity`: Error severity level
- `details`: Additional context information

**Returns**: Unique error ID (UUID string)

**Example**:

```python
if response.status_code != 200:
    error_id = error_reporter.report_error(
        module_name="inference_service",
        work_context="http_request",
        message=f"HTTP request failed: {response.status_code}",
        severity=ErrorSeverity.WARNING,
        details={"url": url, "status": response.status_code}
    )
```

### get_health_status

```python
def get_health_status(self) -> dict[str, Any]
```

Get overall system health based on recent errors (last 5 minutes).

**Returns**: Dictionary with health status

**Return Value**:
```python
{
    "status": "healthy" | "warning" | "error",
    "message": "X errors, Y warnings in last 5 minutes",
    "recent_error_count": 5,      # Total recent errors
    "error_level_count": 2,        # ERROR severity
    "warning_level_count": 3       # WARNING severity
}
```

**Status Logic**:
- **healthy**: No recent errors or warnings
- **warning**: Recent warnings but no errors
- **error**: Recent ERROR severity errors

## Internal Methods

### _get_errors

```python
def _get_errors(
    self,
    module_name: str | None = None,
    severity: ErrorSeverity | None = None,
    since: float | None = None
) -> list[ErrorInfo]
```

Query errors with filtering.

**Parameters**:
- `module_name`: Filter by module
- `severity`: Filter by severity
- `since`: Only errors after this timestamp

**Returns**: List of ErrorInfo objects (newest first)

### _cleanup_old_errors

Automatically removes oldest errors when `max_errors` exceeded.

## Integration Example

### Module-Level Error Reporting

```python
from nexrl.error_reporter import ErrorReporter, ErrorSeverity

class MyRolloutWorker(BaseRolloutWorker):
    def __init__(self, config):
        super().__init__(config)
        self._error_reporter = ErrorReporter()

    def _run_single_work(self, data_item):
        try:
            result = self.process_item(data_item)
            return result
        except Exception as e:
            self._error_reporter.report_exception(
                module_name=self.get_module_name(),
                work_context="run_single_work",
                exception=e,
                details={"data_item_id": data_item.get("id")}
            )
            raise  # Re-raise after reporting
```

### Health Monitoring

```python
# In monitoring loop
def check_system_health():
    health = error_reporter.get_health_status()

    if health["status"] == "error":
        logger.error(f"System unhealthy: {health['message']}")
        # Trigger alerts
    elif health["status"] == "warning":
        logger.warning(f"System warnings: {health['message']}")
    else:
        logger.info("System healthy")
```

### Error Query

```python
# Get recent errors from specific module
from nexrl.error_reporter import ErrorSeverity

recent_time = time.time() - 600  # Last 10 minutes
errors = error_reporter._get_errors(
    module_name="trainer",
    severity=ErrorSeverity.ERROR,
    since=recent_time
)

for error in errors:
    print(f"[{error.timestamp}] {error.work_context}: {error.message}")
```

## Error Handling Patterns

### Pattern 1: Report and Continue

For non-critical errors where execution can continue:

```python
try:
    optional_operation()
except Exception as e:
    error_reporter.report_exception(
        module_name="worker",
        work_context="optional_operation",
        exception=e,
        severity=ErrorSeverity.WARNING
    )
    # Continue execution
```

### Pattern 2: Report and Re-raise

For critical errors that should propagate:

```python
try:
    critical_operation()
except Exception as e:
    error_reporter.report_exception(
        module_name="worker",
        work_context="critical_operation",
        exception=e,
        severity=ErrorSeverity.ERROR
    )
    raise  # Propagate the error
```

### Pattern 3: Report and Retry

For transient errors with retry logic:

```python
max_retries = 3
for attempt in range(max_retries):
    try:
        result = unreliable_operation()
        break
    except Exception as e:
        error_reporter.report_exception(
            module_name="worker",
            work_context=f"operation_attempt_{attempt}",
            exception=e,
            severity=ErrorSeverity.WARNING,
            details={"attempt": attempt, "max_retries": max_retries}
        )
        if attempt == max_retries - 1:
            raise  # Last attempt failed
```

## Best Practices

### 1. Provide Context

Always include meaningful `work_context` and `details`:

```python
# Good
error_reporter.report_error(
    module_name="data_loader",
    work_context="loading_batch_3",
    message="Dataset file not found",
    details={"file_path": path, "batch_id": 3}
)

# Bad
error_reporter.report_error(
    module_name="loader",
    work_context="load",
    message="error"
)
```

### 2. Choose Appropriate Severity

- **ERROR**: Failures that affect training
- **WARNING**: Issues that don't stop execution
- **INFO**: Notable events (rare, usually don't use ErrorReporter for this)

### 3. Include Relevant Details

Add details that help debug the issue:

```python
details = {
    "data_item_id": item["id"],
    "model_tag": model_tag,
    "attempt": retry_count,
    "response_code": response.status_code
}
```

### 4. Log and Report Separately

ErrorReporter automatically logs, but you can add contextual logging:

```python
try:
    operation()
except Exception as e:
    logger.debug(f"Operation failed, will retry: {e}")
    error_reporter.report_exception(...)
```

## Limitations

### Not a Replacement for Logging

ErrorReporter complements but doesn't replace standard logging:
- Use logging for general events and debugging
- Use ErrorReporter for actual errors and health monitoring

### Memory Bounded

Only keeps `max_errors` in memory. For long-term analysis, use logging infrastructure.

### Thread-Safe Only

Safe for multi-threading but not designed for distributed error collection across Ray actors (each actor has its own ErrorReporter instance).

## Related Documentation

- [Core Architecture](../02-core-architecture/overview.md) - System overview
- [Activity Tracking](../02-core-architecture/activity-tracking.md) - Monitoring and logging
- [Best Practices - Debugging](../12-best-practices/debugging.md) - Debugging strategies
