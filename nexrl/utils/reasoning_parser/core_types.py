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

"""Core data types for reasoning parser."""

from dataclasses import dataclass


@dataclass
class ReasoningParseResult:
    """
    Result of parsing reasoning content from a response.

    Attributes:
        reasoning_content: The extracted reasoning content, or None if not found
        cleaned_content: The response content with reasoning removed
        is_valid: Whether the parsing operation succeeded
    """

    reasoning_content: str | None
    cleaned_content: str
    is_valid: bool
