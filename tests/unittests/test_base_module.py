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
Tests for base_module
"""

from nexrl.base_module import NexRLModule


class SimpleModule(NexRLModule):
    """Simple test module for testing base functionality"""

    def __init__(self):
        super().__init__()


def test_module_initialization():
    """Test basic module initialization"""
    module = SimpleModule()
    assert module._module_name == "invalid"
    assert module._activity_tracker is None


def test_module_name_operations():
    """Test setting and getting module name"""
    module = SimpleModule()
    module.set_module_name("test_module")
    assert module.get_module_name() == "test_module"


def test_health_check():
    """Test health check returns True"""
    module = SimpleModule()
    assert module.health_check() is True


def test_set_activity_tracker():
    """Test setting activity tracker"""
    module = SimpleModule()

    # Create a simple mock tracker
    class MockTracker:
        pass

    tracker = MockTracker()
    module.set_activity_tracker(tracker)
    assert module._activity_tracker is tracker
