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
Inference Service Client Module - Contains base and specialized inference clients
"""

from .base_inference_service_client import (
    ChatCompletionsWrapper,
    ChatWrapper,
    CompletionsWrapper,
    InferenceServiceClient,
    hf_tokenizer,
    set_pad_token_id,
)
from .openai_inference_service_client import OpenAIInferenceServiceClient
from .remote_api_inference_service_client import RemoteApiInferenceServiceClient

__all__ = [
    "InferenceServiceClient",
    "OpenAIInferenceServiceClient",
    "RemoteApiInferenceServiceClient",
    "CompletionsWrapper",
    "ChatCompletionsWrapper",
    "ChatWrapper",
    "hf_tokenizer",
    "set_pad_token_id",
]
