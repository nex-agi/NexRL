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
Base module class for all NexRL components
"""

from abc import ABC
from typing import Any, Callable

from .activity_tracker import ActivityTrackerProxy


class NexRLModule(ABC):
    """
    Minimal base class for all NexRL modules to enable colocation compatibility.

    This class serves as a common base for all NexRL modules to ensure they can
    be used in the colocation mechanism. It doesn't impose any specific
    functionality - each module implements its own behavior as needed.

    All NexRL modules should inherit from this class to ensure compatibility
    with the Ray resource management and colocation system.

    Attributes:
        status: A dictionary to store the status of the module.
    """

    def __init__(self):
        """
        Initialize the base module.
        """
        # Global control-plane status that can be accessed by other modules
        # Status updates are automatically broadcasted via AutoNotifyDict
        # Other modules can access the status through the activity tracker
        # Since every status update will be broadcasted to the activity tracker, it's better not to add frequently updated information to the status dictionary
        self._module_name: str = "invalid"
        self._activity_tracker: ActivityTrackerProxy = None  # type: ignore  # Set via set_activity_tracker()

    def set_activity_tracker(self, tracker: ActivityTrackerProxy):
        """
        Set the activity tracker for this module.
        """
        self._activity_tracker = tracker

    def set_module_name(self, module_name: str):
        """
        Set the name of this module.
        """
        self._module_name = module_name

    def get_module_name(self) -> str:
        """
        Get the name of this module.
        """
        return self._module_name

    def health_check(self) -> bool:
        """
        Health check method to verify the module is alive and responsive.
        This is used during initialization to ensure actors are properly created.

        Returns:
            bool: True if the module is healthy
        """
        return True

    def easy_dump(
        self,
        value: Any,
        keys: list[str] | None = None,
        value_formatter: Callable[[Any], str] | None = None,
    ) -> None:
        """
        Convenience method to dump values with automatic module context

        Args:
            value: Any value that needs to be written to files
            keys: List of keys that the user wants to show in the file name
            value_formatter: User-defined function to format the value for output

        Returns:
            str: Path to the created dump file
        """
        try:
            from easy_debug import easy_dump
        except ImportError:
            return

        # Get training step from activity tracker if available
        training_step = -1
        if self._activity_tracker is not None:
            try:
                training_step = self._activity_tracker.get_training_step()
            except Exception:
                # If we can't get the training step, use -1 as default
                training_step = -1

        # Prepend module name to keys for better organization
        if keys is None:
            keys = []
        enriched_keys = [self._module_name] + keys

        easy_dump(
            value=value,
            keys=enriched_keys,
            training_step=training_step,
            value_formatter=value_formatter,
        )
