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
Weaver Inference Service Client for NexRL framework.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from ..inference_service_client.remote_api_inference_service_client import (
    RemoteApiInferenceServiceClient,
)
from .weaver_service_holder import WeaverServiceHolder

logger = logging.getLogger(__name__)


class WeaverInferenceServiceClient(RemoteApiInferenceServiceClient):
    """
    Inference service client using Weaver's sampling API.

    Inherits all inference logic from RemoteApiInferenceServiceClient.
    Provides Weaver-specific naming for the service holder setter.
    """

    def __init__(self, config: DictConfig, weaver_service_holder: WeaverServiceHolder):
        """
        Initialize the Weaver inference service client.

        Args:
            config: Configuration containing LLM settings
            weaver_service_holder: Reference to shared Weaver service holder
        """
        super().__init__(config)
        self.set_weaver_service_holder(weaver_service_holder)

    def set_weaver_service_holder(self, weaver_service_holder: WeaverServiceHolder) -> None:
        """
        Set the Weaver service holder.

        Args:
            weaver_service_holder: WeaverServiceHolder instance
        """
        self._service_holder = weaver_service_holder  # type: ignore

    def set_service_holder(self, service_holder) -> None:
        """
        Set the service holder (generic interface).

        Args:
            service_holder: WeaverServiceHolder instance
        """
        self.set_weaver_service_holder(service_holder)
