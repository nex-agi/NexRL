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
Mock Algorithm Processor for testing purposes
"""

import logging
import time
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from ..algorithm_processor import BaseAlgorithmProcessor
from ..nexrl_types import Batch

logger = logging.getLogger(__name__)


class MockAlgorithmProcessor(BaseAlgorithmProcessor):
    """
    Mock Algorithm Processor for testing purposes
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._processed_count: int = 0
        self._mock_batch_size: int = getattr(config, "batch_size", 32)
        self._vocab_size: int = getattr(config, "vocab_size", 151936)
        self._max_prompt_length: int = getattr(config, "max_prompt_length", 256)
        self._max_response_length: int = getattr(config, "max_response_length", 128)

        # Tokenizer special tokens (typical for Qwen models)
        self._pad_token_id: int = 151643
        self._eos_token_id: int = 151645

    def _fit(self, batch: Batch, update_fn: str) -> None:
        """
        Mock implementation of the main loop that simulates batch processing.

        Args:
            batch: Batch to process
            update_fn: Update function name
        """
        # Simulate processing batch through models
        model = batch.metadata["model_tag"]
        logger.info(
            f"Mock algorithm processor: Forward pass for model={model}, update_fn={update_fn}"
        )

        # Simulate some processing on the batch

        # Generate mock training batch with all metadata using existing method
        logger.info("Generating mock training batch for processing simulation")
        mock_data = self.get_mock_batch_with_metadata()

        processed_batch = Batch(mock_data, {})

        processed_batch.metadata["model_tag"] = model
        processed_batch.metadata["batch_size"] = self._mock_batch_size

        # Simulate putting processed batch to training batch pool
        success = self._put_batch(processed_batch, "mock_update")
        if success:
            self._processed_count += 1
            logger.info(
                f"Mock algorithm processor: Successfully processed batch #{self._processed_count}"
            )
        else:
            logger.warning("Mock algorithm processor: Failed to put batch to training pool")

        # Simulate processing time
        time.sleep(0.5)

    def create_mock_training_batch(
        self,
        batch_size: int | None = None,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
        """
        Create a realistic mock training batch for policy updates with advantages and old_log_probs.

        Based on create_dummy_training_batch from NexTrainer example_demo.py.

        Args:
            batch_size: Number of sequences (uses config default if None)
            max_prompt_length: Maximum prompt length (uses config default if None)
            max_response_length: Maximum response length (uses config default if None)

        Returns:
            Tuple of (tensor_batch, non_tensor_batch)
        """
        batch_size = batch_size or self._mock_batch_size
        max_prompt_length = max_prompt_length or self._max_prompt_length
        max_response_length = max_response_length or self._max_response_length

        logger.info(f"Creating mock training batch (batch_size={batch_size})")

        dummy_sequences = []

        # Generate variable length sequences
        for i in range(batch_size):
            # Create variable length prompt and response
            prompt_len = np.random.randint(30, min(max_prompt_length, 120))
            response_len = np.random.randint(10, min(max_response_length, 50))

            # Generate prompt tokens (avoid special tokens)
            prompt_tokens = torch.randint(1000, self._vocab_size - 1000, (prompt_len,))

            # Generate response tokens (avoid special tokens)
            response_tokens = torch.randint(1000, self._vocab_size - 1000, (response_len,))
            response_tokens = torch.cat([response_tokens, torch.tensor([self._eos_token_id])])
            response_len += 1  # Include EOS

            dummy_sequences.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                }
            )

        # Calculate max lengths for padding
        actual_max_prompt_len = max(seq["prompt_len"] for seq in dummy_sequences)
        actual_max_resp_len = max(seq["response_len"] for seq in dummy_sequences)

        # Use the provided max lengths or actual max lengths, whichever is larger
        final_max_prompt_len = max(max_prompt_length, actual_max_prompt_len)
        final_max_resp_len = max(max_response_length, actual_max_resp_len)
        final_max_total_len = final_max_prompt_len + final_max_resp_len

        # Initialize tensors
        input_ids = torch.full(
            (batch_size, final_max_total_len), self._pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, final_max_total_len), dtype=torch.long)
        responses = torch.full(
            (batch_size, final_max_resp_len), self._pad_token_id, dtype=torch.long
        )
        position_ids = torch.zeros((batch_size, final_max_total_len), dtype=torch.long)

        # Create advantages tensor (key requirement for policy update)
        advantages = torch.randn(batch_size, final_max_resp_len, dtype=torch.float32) * 0.1

        # Create old_log_probs tensor (used for PPO clipping)
        old_log_probs = torch.randn(batch_size, final_max_resp_len, dtype=torch.float32) * 0.5 - 2.0

        # Fill tensors with actual data
        for i, seq in enumerate(dummy_sequences):
            prompt_tokens = seq["prompt_tokens"]
            response_tokens = seq["response_tokens"]
            prompt_len = seq["prompt_len"]
            response_len = seq["response_len"]

            # Right-pad prompt in the first part of input_ids
            input_ids[i, :prompt_len] = prompt_tokens

            # Right-pad response in the second part of input_ids
            input_ids[i, final_max_prompt_len : final_max_prompt_len + response_len] = (
                response_tokens
            )

            # Attention mask for all valid tokens
            attention_mask[i, :prompt_len] = 1  # Prompt tokens
            attention_mask[i, final_max_prompt_len : final_max_prompt_len + response_len] = (
                1  # Response tokens
            )

            # Responses tensor (right-padded)
            responses[i, :response_len] = response_tokens

            # Position IDs: sequential positions for valid tokens
            position_ids[i, :prompt_len] = torch.arange(prompt_len)
            position_ids[i, final_max_prompt_len : final_max_prompt_len + response_len] = (
                torch.arange(prompt_len, prompt_len + response_len)
            )

            # Set advantages to zero for padding tokens
            advantages[i, response_len:] = 0.0

            # Set old_log_probs to zero for padding tokens
            old_log_probs[i, response_len:] = 0.0

        # Tensor batch - main training data
        tensor_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "responses": responses,
            "position_ids": position_ids,
            "advantages": advantages,  # Required for policy update
            "old_log_probs": old_log_probs,  # Used for PPO clipping
        }

        # Non-tensor batch - metadata and auxiliary data
        non_tensor_batch = {
            "sequence_ids": np.array([f"seq_{i}" for i in range(batch_size)], dtype=object),
            "prompt_lengths": np.array(
                [seq["prompt_len"] for seq in dummy_sequences], dtype=object
            ),
            "response_lengths": np.array(
                [seq["response_len"] for seq in dummy_sequences], dtype=object
            ),
        }

        logger.info(f"Created mock batch: {input_ids.shape}, advantages: {advantages.shape}")

        return tensor_batch, non_tensor_batch

    def create_mock_inference_batch(
        self,
        batch_size: int | None = None,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Create a realistic mock batch for inference/log probability computation.

        Args:
            batch_size: Number of sequences (uses config default if None)
            max_prompt_length: Maximum prompt length (uses config default if None)
            max_response_length: Maximum response length (uses config default if None)

        Returns:
            Dict containing PyTorch tensors for inference
        """
        batch_size = batch_size or self._mock_batch_size
        max_prompt_length = max_prompt_length or self._max_prompt_length
        max_response_length = max_response_length or self._max_response_length

        logger.info(f"Creating mock inference batch (batch_size={batch_size})")

        dummy_sequences = []

        for i in range(batch_size):
            # Create variable length prompt and response
            prompt_len = np.random.randint(50, min(max_prompt_length, 200))
            response_len = np.random.randint(20, min(max_response_length, 100))

            # Generate prompt tokens (avoid special tokens)
            prompt_tokens = torch.randint(1000, self._vocab_size - 1000, (prompt_len,))

            # Generate response tokens (avoid special tokens)
            response_tokens = torch.randint(1000, self._vocab_size - 1000, (response_len,))
            response_tokens = torch.cat([response_tokens, torch.tensor([self._eos_token_id])])
            response_len += 1  # Include EOS

            dummy_sequences.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                }
            )

        # Calculate max lengths for padding
        max_total_len = max(seq["prompt_len"] + seq["response_len"] for seq in dummy_sequences)
        max_resp_len = max(seq["response_len"] for seq in dummy_sequences)

        # Initialize tensors
        input_ids = torch.full((batch_size, max_total_len), self._pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_total_len), dtype=torch.long)
        response_mask = torch.zeros((batch_size, max_total_len), dtype=torch.long)
        responses = torch.full((batch_size, max_resp_len), self._pad_token_id, dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_total_len), dtype=torch.long)

        for i, seq in enumerate(dummy_sequences):
            prompt_tokens = seq["prompt_tokens"]
            response_tokens = seq["response_tokens"]
            prompt_len = seq["prompt_len"]
            response_len = seq["response_len"]
            total_len = prompt_len + response_len

            # Left pad prompt, right pad response in input_ids
            input_ids[i, :prompt_len] = prompt_tokens
            input_ids[i, prompt_len:total_len] = response_tokens

            # Attention mask for all valid tokens
            attention_mask[i, :total_len] = 1

            # Response mask only for response tokens
            response_mask[i, prompt_len:total_len] = 1

            # Responses tensor (right-padded)
            responses[i, :response_len] = response_tokens

            # Position IDs: sequential positions for valid tokens
            position_ids[i, :total_len] = torch.arange(total_len)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "responses": responses,
            "position_ids": position_ids,
        }

        logger.info(f"Created inference batch: {input_ids.shape}")

        return batch

    def get_mock_batch_with_metadata(
        self,
        batch_size: int | None = None,
        temperature: float = 1.0,
        ppo_epochs: int = 1,
        ppo_mini_batch_size: int | None = None,
        ppo_micro_batch_size_per_gpu: int = 2,
        use_dynamic_bsz: bool = False,
    ) -> dict[str, Any]:
        """
        Create a complete mock batch with all metadata needed for policy updates.

        Returns:
            Dict containing 'batch', 'non_tensor_batch', and 'meta_info' keys
        """
        batch_size = batch_size or self._mock_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size or max(1, batch_size // 2)

        # Create the mock training batch
        tensor_batch, non_tensor_batch = self.create_mock_training_batch(batch_size=batch_size)

        # Calculate global token numbers
        global_token_num = [
            int(seq_mask.sum().item()) for seq_mask in tensor_batch["attention_mask"]
        ]

        # Create complete data structure
        data = {
            "batch": tensor_batch,
            "non_tensor_batch": non_tensor_batch,
            "meta_info": {
                "temperature": temperature,
                "ppo_epochs": ppo_epochs,
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": ppo_micro_batch_size_per_gpu,
                "use_dynamic_bsz": use_dynamic_bsz,
                "global_token_num": global_token_num,
            },
        }

        logger.info(
            f"Created complete mock batch with metadata, total tokens: {sum(global_token_num)}"
        )

        return data
