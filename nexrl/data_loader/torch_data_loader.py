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
TorchDataLoader - A real data loader that uses PyTorch's DataLoader
"""

import copy
import logging
from typing import Any, List, Union, override

import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader, Dataset

from .data_loader import SequentialDataLoader

logger = logging.getLogger(__name__)


class ParquetDataset(Dataset):
    """
    PyTorch Dataset for loading data from parquet files.

    Args:
        parquet_files: Single parquet file path or list of paths
        filter_prompts: Whether to filter prompts based on length
        max_prompt_length: Maximum allowed prompt length (in tokens)
        tokenizer: Tokenizer for filtering prompts
        prompt_key: Key in the data dict that contains the prompt
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        filter_prompts: bool = False,
        max_prompt_length: int = 1024,
        tokenizer=None,
        prompt_key: str = "prompt",
    ) -> None:
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume

        self.filter_prompts = filter_prompts
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.tokenizer = tokenizer

        self._read_files_and_filter()
        self.serialize_dataset = False

    def _read_files_and_filter(self) -> None:
        """Read parquet files and optionally filter based on prompt length"""
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        logger.info(
            f"ParquetDataset: Loaded {len(self.dataframe)} items from {len(self.parquet_files)} files"
        )

        # filter out too long prompts
        if self.filter_prompts:
            logger.info(
                f"ParquetDataset: Filtering prompts with max_prompt_length={self.max_prompt_length}"
            )
            assert self.tokenizer is not None, "tokenizer is required when filter_prompts is True"
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key

            def check_prompt_length(doc) -> bool:
                try:
                    # Apply chat template to get the formatted prompt
                    formatted_prompt = tokenizer.apply_chat_template(
                        doc[prompt_key],
                        add_generation_prompt=True,
                        tokenize=False,
                        add_special_tokens=True,
                    )
                    # Tokenize the formatted prompt to get actual token length
                    tokenized_prompt = tokenizer(
                        formatted_prompt, return_tensors="pt", add_special_tokens=False
                    )
                    input_ids = tokenized_prompt["input_ids"]
                    if isinstance(input_ids, list):
                        input_ids = torch.tensor([input_ids])
                    assert input_ids.ndim == 2
                    sequence_length = input_ids.shape[-1]
                    return sequence_length <= self.max_prompt_length
                except Exception as e:
                    logger.warning(f"Failed to process prompt: {e}")
                    return False

            original_len = len(self.dataframe)
            self.dataframe = self.dataframe[self.dataframe.apply(check_prompt_length, axis=1)]
            self.dataframe = self.dataframe.reset_index(drop=True)
            logger.info(
                f"ParquetDataset: Filtered dataset from {original_len} to {len(self.dataframe)} items"
            )

    def resume_dataset_state(self) -> None:
        """Resume dataset state by re-reading files"""
        self.serialize_dataset = False if hasattr(self, "original_parquet_files") else True
        self._read_files_and_filter()

    def add_item_back(self, item: dict[str, Any]) -> None:
        """
        Add a data item to the back of the dataset by appending to the dataframe.

        Args:
            item: Data item to add to the dataset
        """
        # Convert the item dict to a DataFrame row and append
        new_row = pd.DataFrame([item])
        self.dataframe = pd.concat([self.dataframe, new_row], ignore_index=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Get item from dataset as a dictionary"""
        row_dict = self.dataframe.iloc[item].to_dict()

        # add index for each prompt if available
        index = row_dict.get("extra_info", {}).get("index", item)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self) -> dict[str, Any]:
        """Custom serialization to avoid pickling large dataframes"""
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()


class TorchDataLoader(SequentialDataLoader):
    """
    A real data loader that uses PyTorch's DataLoader internally.

    This class wraps PyTorch's DataLoader and adapts it to the SequentialDataLoader interface.
    It reads data from parquet files and provides batched iteration.

    Config parameters:
        - data_files: List of parquet file paths to load
        - batch_size: Batch size for loading data
        - filter_prompts: Whether to filter prompts by length (default: False)
        - max_prompt_length: Maximum prompt length in tokens (default: 1024)
        - tokenizer_path: Path to tokenizer for filtering (optional)
        - prompt_key: Key for prompt in data dict (default: "prompt")
        - shuffle: Whether to shuffle data (default: False)
        - seed: Random seed for shuffling (default: 42). Validation dataloader uses seed+1000
        - drop_last: Whether to drop last incomplete batch (default: False)
    """

    def __init__(self, config: DictConfig, is_validate: bool = False) -> None:
        super().__init__(config, is_validate)
        self._config = config

        # Get data loading parameters
        self._data_files = config.get("data_files", [])
        if not self._data_files:
            raise ValueError("data_files must be specified in config")

        self._filter_prompts = config.get("filter_prompts", False)
        self._max_prompt_length = config.get("max_prompt_length", 1024)
        self._prompt_key = config.get("prompt_key", "prompt")
        self._shuffle = config.get("shuffle", False)
        self._drop_last = config.get("drop_last", False)

        # Initialize tokenizer if needed for filtering
        self._tokenizer = None
        if self._filter_prompts:
            tokenizer_path = config.get("tokenizer_path")
            if tokenizer_path:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                logger.info(f"TorchDataLoader: Loaded tokenizer from {tokenizer_path}")
            else:
                logger.warning("filter_prompts is True but no tokenizer_path provided")

        # Create dataset
        self._dataset = ParquetDataset(
            parquet_files=self._data_files,
            filter_prompts=self._filter_prompts,
            max_prompt_length=self._max_prompt_length,
            tokenizer=self._tokenizer,
            prompt_key=self._prompt_key,
        )

        # Create a dedicated random generator for this dataloader to ensure deterministic,
        # independent shuffling that doesn't interfere with other dataloaders

        self._generator = None
        if self._shuffle:
            seed = config.get("seed", 42)
            # Add an offset for validation dataloader to ensure different shuffle order
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)
            logger.info(
                f"TorchDataLoader: Created dedicated random generator with seed={seed} "
                f"(is_validate={is_validate})"
            )

        # Create DataLoader
        self._dataloader = DataLoader(
            dataset=self._dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
            collate_fn=self._collate_fn,
            generator=self._generator,  # Use dedicated generator for isolated random state
        )

        # Iterator for sequential access
        self._dataloader_iter = iter(self._dataloader)
        self._is_exhausted = False

        logger.info(
            f"TorchDataLoader initialized - "
            f"is_validate: {self._is_validate}, "
            f"files: {len(self._data_files)}, "
            f"dataset size: {len(self._dataset)}, "
            f"batch_size: {self._batch_size}, "
            f"shuffle: {self._shuffle}, "
            f"filter_prompts: {self._filter_prompts}"
        )

    @staticmethod
    def _collate_fn(batch: List[dict]) -> List[dict]:
        """
        Collate function that simply returns the batch as a list of dicts.
        This avoids PyTorch's default collation which tries to stack tensors.

        Args:
            batch: List of data items from dataset

        Returns:
            List of data items unchanged
        """
        return batch

    @override
    def _fetch_batch_data(self) -> list[dict[str, Any]]:
        """
        Fetch one batch of items from the PyTorch DataLoader.

        Returns:
            list[dict[str, Any]]: List of data items (up to batch_size)
        """
        if self._is_exhausted:
            return []

        try:
            batch = next(self._dataloader_iter)
            return batch
        except StopIteration:
            self._is_exhausted = True
            logger.info("TorchDataLoader: All data has been consumed")
            return []

    @override
    def add_item_back(self, item: dict[str, Any]) -> None:
        """
        Add a data item to the back of the data buffer.

        Note: This implementation does NOT support adding items back to the underlying
        dataset during iteration. This is a placeholder for future implementation.
        Currently, items are only added to the buffer, not persisted to the dataset.

        Args:
            item: Data item to add
        """
        logger.warning(
            "TorchDataLoader: Adding items back is not supported. This item will be added to the current batch."
        )
        with self._lock:
            self._data_buffer.append(item)

    @override
    def is_finished(self) -> bool:
        """
        Return True when all data has been fetched and buffer is empty.

        Checks that:
        1. Main DataLoader is exhausted
        2. Current buffer is fully consumed

        Returns:
            True if all data has been fetched and buffer is empty, False otherwise
        """
        with self._lock:
            return self._is_exhausted and self._buffer_index >= len(self._data_buffer)

    def get_dataset_size(self) -> int:
        """
        Get the total size of the underlying dataset.

        Returns:
            Total number of items in the dataset
        """
        return len(self._dataset)

    @override
    def _reset_iterator(self) -> None:
        """
        Reset the PyTorch DataLoader iterator to the beginning.

        Creates a new iterator from the DataLoader and resets the exhausted flag.
        """
        self._dataloader_iter = iter(self._dataloader)
        self._is_exhausted = False
