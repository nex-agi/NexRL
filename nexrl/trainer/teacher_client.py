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
TeacherClient - Lightweight wrapper for teacher model logprob computation

This class provides a uniform interface for computing log probabilities from
teacher models using Weaver or Tinker sampling clients. It is created and owned
by the RemoteApiOpdTrainer, not by the service holder.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _create_service_client(backend: str, base_url: str, api_key: str | None = None):
    """Create a Weaver or Tinker ServiceClient for a teacher model.

    Args:
        backend: "weaver" or "tinker"
        base_url: Base URL for the service
        api_key: API key (for Weaver only, optional - reads from env if not provided)

    Returns:
        ServiceClient instance
    """
    from ..utils.url_utils import ensure_url_scheme

    normalized_url = ensure_url_scheme(base_url, default_scheme="https")

    if backend == "weaver":
        try:
            from weaver import ServiceClient
        except ImportError as exc:
            raise ImportError(
                "weaver package not installed. Install with: pip install weaver"
            ) from exc

        # Use provided api_key or fall back to environment variable
        weaver_api_key = api_key or os.getenv("WEAVER_API_KEY")
        if not weaver_api_key:
            raise ValueError(
                "WEAVER_API_KEY is required but not found in environment variables or config"
            )

        service_client = ServiceClient(base_url=normalized_url, api_key=weaver_api_key)
        service_client.connect()
        return service_client

    elif backend == "tinker":
        try:
            import tinker
        except ImportError as exc:
            raise ImportError(
                "tinker package not installed. Install with: pip install tinker"
            ) from exc

        service_client = tinker.ServiceClient(base_url=normalized_url)
        return service_client

    else:
        raise ValueError(f"Unsupported backend: {backend}. Must be 'weaver' or 'tinker'")


class TeacherClient:
    """Lightweight teacher model client for logprob computation.

    Wraps a Weaver or Tinker SamplingClient. Created by the trainer,
    NOT by the service holder. Supports same-service or separate-service teachers.

    The teacher model is typically a larger/better model than the student
    (e.g., student=Qwen3-8B, teacher=Qwen3-32B) but must share the same tokenizer.
    """

    @staticmethod
    def from_config(
        teacher_config: dict[str, Any],
        student_service_holder,  # WeaverServiceHolder or TinkerServiceHolder
    ) -> "TeacherClient":
        """Factory: create a TeacherClient from config + student service holder.

        If teacher_config has base_url, creates a new ServiceClient.
        Otherwise reuses the student's ServiceClient via get_service_client().

        Args:
            teacher_config: Dict with keys:
                - base_model: str (e.g., "Qwen/Qwen3-32B")
                - checkpoint: str | None (optional fine-tuned checkpoint path)
                - base_url: str | None (optional separate service URL)
                - api_key: str | None (optional API key for separate service)
                - backend: str | None (optional, inferred from holder if not provided)
            student_service_holder: WeaverServiceHolder or TinkerServiceHolder

        Returns:
            TeacherClient instance
        """
        base_model = teacher_config["base_model"]
        checkpoint = teacher_config.get("checkpoint")
        teacher_base_url = teacher_config.get("base_url")
        teacher_api_key = teacher_config.get("api_key")

        # Infer backend from service holder type if not explicitly provided
        backend = teacher_config.get("backend")
        if backend is None:
            from ..weaver.weaver_service_holder import WeaverServiceHolder

            if isinstance(student_service_holder, WeaverServiceHolder):
                backend = "weaver"
            else:
                backend = "tinker"

        logger.info(
            f"Creating TeacherClient: base_model={base_model}, "
            f"checkpoint={checkpoint}, backend={backend}, "
            f"separate_service={bool(teacher_base_url)}"
        )

        if teacher_base_url:
            # Teacher on separate service -- create new ServiceClient
            logger.info(f"Teacher on separate service: {teacher_base_url}")
            service_client = _create_service_client(backend, teacher_base_url, teacher_api_key)
        else:
            # Teacher on same service as student -- reuse ServiceClient
            service_client = student_service_holder.get_service_client()

        # Create sampling client for the teacher model
        if backend == "weaver":
            kwargs = {"base_model": base_model}
            if checkpoint:
                kwargs["model_path"] = checkpoint
            sampling_client = service_client.create_sampling_client(**kwargs)
        else:  # tinker
            sampling_client = service_client.create_sampling_client(
                base_model=base_model,
                model_path=checkpoint,
            )

        logger.info(f"TeacherClient created successfully for {base_model}")
        return TeacherClient(sampling_client, backend)

    def __init__(self, sampling_client, backend: str):
        """Initialize TeacherClient with a sampling client.

        Args:
            sampling_client: Weaver or Tinker SamplingClient
            backend: "weaver" or "tinker"
        """
        self._sampling_client = sampling_client
        self._backend = backend

    def compute_logprobs(self, tokens: list[int]) -> list[float | None]:
        """Compute log probabilities for a token sequence.

        Args:
            tokens: List of token IDs (must be compatible with teacher's tokenizer)

        Returns:
            List of log probabilities, one per token. First element is typically None
            (no logprob for the first token since there's no previous context).
        """
        if self._backend == "weaver":
            from weaver import types as weaver_types

            model_input = weaver_types.ModelInput.from_ints(tokens=tokens)
            logprobs = self._sampling_client.compute_logprobs(prompt=model_input)
            return list(logprobs) if logprobs else []

        else:  # tinker
            from tinker import types as tinker_types

            model_input = tinker_types.ModelInput.from_ints(tokens=tokens)
            logprobs_future = self._sampling_client.compute_logprobs(prompt=model_input)
            logprobs = logprobs_future.result()
            return list(logprobs) if logprobs else []
