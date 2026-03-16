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
StreamingDatasetDataLoader - Data loader backed by the StreamingDataset package.

Wraps the StreamingDataset API to load pre-tokenized, packed SFT data and
expose it through the SequentialDataLoader interface.
"""

import logging
from typing import Any, Callable, override

from omegaconf import DictConfig, OmegaConf

from .data_loader import SequentialDataLoader
from .process_funcs import BUILTIN_PROCESS_FUNCS

logger = logging.getLogger(__name__)

# Module-level registry for process functions.
# Built-in functions are registered automatically; external callers can add
# more via :func:`register_process_func` before ``main_task()`` runs.
_PROCESS_FUNC_REGISTRY: dict[str, Callable] = {**BUILTIN_PROCESS_FUNCS}


def register_process_func(name: str, func: Callable) -> None:
    """Register a process function that can be referenced by name in subset_params."""
    _PROCESS_FUNC_REGISTRY[name] = func


class StreamingDatasetDataLoader(SequentialDataLoader):
    """
    Data loader that uses the StreamingDataset package for pre-tokenized, packed data.

    Each batch from the streaming dataloader is a tuple ``(inputs_dict, labels_tensor)``
    where ``inputs_dict["input_ids"]`` and ``labels_tensor`` are tensors of shape
    ``[micro_num, packed_length]``.  This class splits them into individual dicts
    ``{"input_ids": list[int], "labels": list[int]}`` — one per packed sequence.

    Config parameters:
        - train_folder: Path to training data folder
        - vocab_file: Path to tokenizer / model for tokenization
        - streaming_tokenizer_type: Tokenizer backend (default: "HF")
        - tokenizer_wrapper: Wrapper type (e.g., "sft_multi_round", "pretrain")
        - break_mode: How to split data (e.g., "pass_through", "cut")
        - packed_length: Length of each packed sequence
        - max_length_per_sample: Maximum length per individual sample
        - min_length: Minimum sample length (default: 0)
        - num_worker: Number of dataloader workers (default: 4)
        - micro_num: Number of micro-batches (default: 1)
        - seed: Random seed (default: 42)
        - subset_params: Subset parameters for the dataset (default: {})
        - dataset_weights: Optional dataset weighting
        - total_steps: Total training steps (used for token estimation)
    """

    def __init__(self, config: DictConfig, is_validate: bool = False) -> None:
        super().__init__(config, is_validate)
        self._config = config

        # Validate required fields
        train_folder = config.get("train_folder")
        if not train_folder:
            raise ValueError("train_folder must be specified for streaming_dataset data loader")

        vocab_file = config.get("vocab_file")
        if not vocab_file:
            raise ValueError("vocab_file must be specified for streaming_dataset data loader")

        # Resolve subset_params to a plain dict and inject registered process funcs
        raw_subset = config.get("subset_params", {})
        if isinstance(raw_subset, DictConfig):
            subset_params = OmegaConf.to_container(raw_subset, resolve=True)
        else:
            subset_params = dict(raw_subset)

        for _subset_name, params in subset_params.items():
            if not isinstance(params, dict):
                continue
            func_ref = params.get("process_func")
            if isinstance(func_ref, str) and func_ref in _PROCESS_FUNC_REGISTRY:
                params["process_func"] = _PROCESS_FUNC_REGISTRY[func_ref]

        # Build the config dict for StreamingDataset context
        sd_config = {
            "train_folder": train_folder,
            "vocab_file": vocab_file,
            "streaming_tokenizer_type": config.get("streaming_tokenizer_type", "HF"),
            "tokenizer_wrapper": config.get("tokenizer_wrapper", "pretrain"),
            "break_mode": config.get("break_mode", "cut"),
            "packed_length": config.get("packed_length", 2048),
            "max_length_per_sample": config.get("max_length_per_sample", 2048),
            "min_length": config.get("min_length", 0),
            "num_worker": config.get("num_worker", 4),
            "micro_num": config.get("micro_num", 1),
            "micro_bsz": 1,  # StreamingDataset packs internally; 1 micro-batch = 1 packed seq
            "seed": config.get("seed", 42),
            "subset_params": subset_params,
            "dataset_weights": config.get("dataset_weights", None),
            "total_steps": config.get("total_steps", 10000),
            "type": "streaming",
            "use_bos": config.get("use_bos", True),
            "use_eos": config.get("use_eos", True),
            "text_field": config.get("text_field", "content"),
        }

        # Initialize the StreamingDataset context and build the dataloader
        from streaming_dataset import build_train_loader_with_data_type
        from streaming_dataset.context import init_streaming_dataset_context

        init_streaming_dataset_context(
            config=sd_config,
            data_parallel_size=1,
            data_parallel_rank=0,
        )

        self._train_dl, _, _ = build_train_loader_with_data_type()
        self._train_dl_iter = iter(self._train_dl)
        self._is_exhausted = False

        logger.info(
            f"StreamingDatasetDataLoader initialized - "
            f"is_validate: {self._is_validate}, "
            f"train_folder: {train_folder}, "
            f"packed_length: {sd_config['packed_length']}, "
            f"tokenizer_wrapper: {sd_config['tokenizer_wrapper']}"
        )

    @override
    def _fetch_batch_data(self) -> list[dict[str, Any]]:
        """
        Fetch one batch from the streaming dataloader and split into individual dicts.

        The streaming dataloader returns ``(inputs_dict, labels_tensor)`` where
        ``inputs_dict["input_ids"]`` has shape ``[micro_num, packed_length]`` and
        ``labels_tensor`` has shape ``[micro_num, packed_length]``.

        Returns:
            list[dict[str, Any]]: List of dicts, each with "input_ids" and "labels" as
                lists of ints.
        """
        if self._is_exhausted:
            return []

        try:
            inputs_dict, labels_tensor = next(self._train_dl_iter)
        except StopIteration:
            self._is_exhausted = True
            logger.info("StreamingDatasetDataLoader: All data has been consumed")
            return []

        input_ids_tensor = inputs_dict["input_ids"]  # [micro_num, packed_length]

        batch_items: list[dict[str, Any]] = []
        batch_size = input_ids_tensor.shape[0]
        for i in range(batch_size):
            item = {
                "input_ids": input_ids_tensor[i].tolist(),
                "labels": labels_tensor[i].tolist(),
            }
            batch_items.append(item)

        return batch_items

    @override
    def add_item_front(self, item: dict[str, Any]) -> None:
        """
        Streaming datasets do not support item insertion.
        Logs a warning and inserts the item at the front of the current buffer.
        """
        logger.warning(
            "StreamingDatasetDataLoader: Streaming datasets do not support adding items to front. "
            "Item will be inserted at the front of the current buffer."
        )
        with self._lock:
            self._data_buffer.insert(self._buffer_index, item)

    @override
    def add_item_back(self, item: dict[str, Any]) -> None:
        """
        Streaming datasets do not support item insertion.
        Logs a warning and adds the item to the current buffer as a best-effort fallback.
        """
        logger.warning(
            "StreamingDatasetDataLoader: Streaming datasets do not support adding items back. "
            "Item will be added to the current buffer."
        )
        with self._lock:
            self._data_buffer.append(item)

    @override
    def is_finished(self) -> bool:
        """
        Return True when all data has been fetched and buffer is empty.
        """
        with self._lock:
            return self._is_exhausted and self._buffer_index >= len(self._data_buffer)

    @override
    def _reset_iterator(self) -> None:
        """
        Reset the streaming dataloader iterator.
        """
        self._train_dl_iter = iter(self._train_dl)
        self._is_exhausted = False
