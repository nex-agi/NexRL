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
Tinker Inference Service Client for NexRL framework
"""

import logging

from omegaconf import DictConfig

from ..inference_service_client.remote_api_inference_service_client import (
    RemoteApiInferenceServiceClient,
)
from .tinker_service_holder import TinkerServiceHolder

logger = logging.getLogger(__name__)


class TinkerInferenceServiceClient(RemoteApiInferenceServiceClient):
    """
    Inference service client using Tinker's sampling API.

    Inherits all inference logic from RemoteApiInferenceServiceClient.
    Provides Tinker-specific naming for the service holder setter.
    """

    def __init__(self, config: DictConfig, tinker_service_holder: TinkerServiceHolder):
        """
        Initialize the Tinker inference service client.

        Args:
            config: Configuration containing LLM settings
            tinker_service_holder: Reference to shared Tinker service holder
        """
        super().__init__(config)
        self.set_tinker_service_holder(tinker_service_holder)

    def set_tinker_service_holder(self, tinker_service_holder: TinkerServiceHolder) -> None:
        """
        Set the Tinker service holder.

        Args:
            tinker_service_holder: TinkerServiceHolder instance
        """
        self._service_holder = tinker_service_holder  # type: ignore

    def set_service_holder(self, service_holder) -> None:
        """
        Set the service holder (generic interface).

        Args:
            service_holder: TinkerServiceHolder instance
        """
        self.set_tinker_service_holder(service_holder)
