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
Top-level pytest configuration file

This file makes fixtures from unittests/ available to all test subdirectories.
"""

# Import fixtures from unittests conftest to make them available globally
# This allows both unittests/ and lint/ to use the same fixtures if needed
pytest_plugins = ["tests.unittests.conftest"]
