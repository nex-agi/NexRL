# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Error Reporter for NexRL framework - Centralized error tracking and reporting
"""

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ErrorInfo:
    """Information about a reported error"""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    module_name: str = ""
    work_context: str = ""
    severity: ErrorSeverity = ErrorSeverity.ERROR
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    exception_type: str | None = None
    exception_traceback: str | None = None


class ErrorReporter:
    """
    Centralized error reporting and aggregation system.

    This class can work in both local and distributed (Ray) modes.
    It maintains a central repository of errors from all modules and
    provides methods for querying and managing errors.
    """

    def __init__(self, max_errors: int = 1000):
        """
        Initialize the error reporter

        Args:
            max_errors: Maximum number of errors to keep in memory
        """
        self._lock = threading.Lock()
        self._errors: dict[str, ErrorInfo] = {}  # error_id -> ErrorInfo
        self._errors_by_module: dict[str, list[str]] = {}  # module -> [error_ids]
        self._max_errors = max_errors

    def report_exception(
        self,
        module_name: str,
        work_context: str,
        exception: Exception,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: dict[str, Any] | None = None,
    ) -> str:
        """
        Report an exception error

        Args:
            module_name: Name of the module where error occurred
            work_context: Context of the work being performed
            exception: The exception that occurred
            severity: Severity level of the error
            details: Additional details about the error

        Returns:
            str: Unique error ID
        """
        error_info = ErrorInfo(
            module_name=module_name,
            work_context=work_context,
            severity=severity,
            message=str(exception),
            details=details or {},
            exception_type=type(exception).__name__,
            exception_traceback=traceback.format_exc(),
        )

        return self._add_error(error_info)

    def report_error(
        self,
        module_name: str,
        work_context: str,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: dict[str, Any] | None = None,
    ) -> str:
        """
        Report a non-exception error

        Args:
            module_name: Name of the module where error occurred
            work_context: Context of the work being performed
            message: Error message
            severity: Severity level of the error
            details: Additional details about the error

        Returns:
            str: Unique error ID
        """
        error_info = ErrorInfo(
            module_name=module_name,
            work_context=work_context,
            severity=severity,
            message=message,
            details=details or {},
        )

        return self._add_error(error_info)

    def _add_error(self, error_info: ErrorInfo) -> str:
        """Add an error to the repository"""
        with self._lock:
            # Add to main errors dict
            self._errors[error_info.error_id] = error_info

            # Add to module index
            if error_info.module_name not in self._errors_by_module:
                self._errors_by_module[error_info.module_name] = []
            self._errors_by_module[error_info.module_name].append(error_info.error_id)

            # Cleanup old errors if needed
            self._cleanup_old_errors()

            # Log the error
            self._log_error(error_info)

        return error_info.error_id

    def _cleanup_old_errors(self):
        """Remove old errors if we exceed max_errors limit"""
        if len(self._errors) <= self._max_errors:
            return

        # Remove oldest errors
        sorted_errors = sorted(self._errors.items(), key=lambda x: x[1].timestamp)
        errors_to_remove = len(self._errors) - self._max_errors + 100  # Remove extra for buffer

        for error_id, error_info in sorted_errors[:errors_to_remove]:
            # Remove from main dict
            del self._errors[error_id]

            # Remove from module index
            if error_info.module_name in self._errors_by_module:
                if error_id in self._errors_by_module[error_info.module_name]:
                    self._errors_by_module[error_info.module_name].remove(error_id)

    def _log_error(self, error_info: ErrorInfo):
        """Log the error using standard logging"""
        log_message = f"[{error_info.module_name}:{error_info.work_context}] {error_info.message}"

        if error_info.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        # Log exception details if available
        if error_info.exception_traceback:
            logger.error(f"Exception traceback:\n{error_info.exception_traceback}")

    def _get_errors(
        self,
        module_name: str | None = None,
        severity: ErrorSeverity | None = None,
        since: float | None = None,
    ) -> list[ErrorInfo]:
        """
        Get errors based on filtering criteria

        Args:
            module_name: Filter by module name
            severity: Filter by severity level
            since: Only return errors since this timestamp

        Returns:
            list[ErrorInfo]: Filtered list of errors
        """
        with self._lock:
            errors = list(self._errors.values())

        # Apply filters
        if module_name:
            errors = [e for e in errors if e.module_name == module_name]
        if severity:
            errors = [e for e in errors if e.severity == severity]
        if since:
            errors = [e for e in errors if e.timestamp >= since]

        # Sort by timestamp (newest first)
        errors.sort(key=lambda e: e.timestamp, reverse=True)
        return errors

    def get_health_status(self) -> dict[str, Any]:
        """
        Get overall system health status based on recent errors

        Returns:
            dict with health status information
        """
        recent_threshold = time.time() - 300  # 5 minutes
        recent_errors = self._get_errors(since=recent_threshold)

        # Categorize health status
        error_level_errors = [e for e in recent_errors if e.severity == ErrorSeverity.ERROR]
        warning_level_errors = [e for e in recent_errors if e.severity == ErrorSeverity.WARNING]

        message = f"{len(error_level_errors)} errors, {len(warning_level_errors)} warnings in last 5 minutes"
        if len(error_level_errors) > 0:
            status = "error"
        elif len(warning_level_errors) > 0:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "message": message,
            "recent_error_count": len(recent_errors),
            "error_level_count": len(error_level_errors),
            "warning_level_count": len(warning_level_errors),
        }
