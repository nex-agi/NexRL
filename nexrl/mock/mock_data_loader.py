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
Mock Data Loader for testing purposes
"""

import logging
import random
import time
from typing import Any, override

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..data_loader import SequentialDataLoader

logger = logging.getLogger(__name__)

# Pre-defined list of 32 real questions for mock data
MOCK_QUESTIONS = [
    "What are the main causes of climate change?",
    "Explain the concept of machine learning in simple terms.",
    "How does photosynthesis work in plants?",
    "What is the difference between DNA and RNA?",
    "Describe the process of how vaccines work.",
    "What are the benefits of regular exercise?",
    "Explain the theory of relativity.",
    "How do computers process binary code?",
    "What is the water cycle and why is it important?",
    "Describe the structure of an atom.",
    "What causes seasons to change on Earth?",
    "How does the human immune system fight infections?",
    "What is artificial intelligence and how is it used today?",
    "Explain the difference between weather and climate.",
    "How do solar panels convert sunlight into electricity?",
    "What are the main functions of the human brain?",
    "Describe the process of digestion in the human body.",
    "What is quantum computing and how does it differ from classical computing?",
    "How do plants and animals depend on each other in an ecosystem?",
    "What are the key principles of sustainable development?",
    "Explain how the Internet works.",
    "What is democracy and what are its core principles?",
    "How do mountains form over geological time?",
    "What is the difference between renewable and non-renewable energy?",
    "Describe the basic structure of a cell.",
    "How does gravity affect objects in space?",
    "What are the main types of clouds and how do they form?",
    "Explain the concept of supply and demand in economics.",
    "How do antibiotics work and why is antibiotic resistance a concern?",
    "What is the role of the ocean in regulating Earth's climate?",
    "Describe the process of evolution by natural selection.",
    "What are the key differences between classical and quantum physics?",
]


class MockDataLoader(SequentialDataLoader):
    """
    Mock Data Loader for testing purposes
    Mock data is generated in the _read_file method,
    which is a DataFrame with 2 columns: id, prompt, mock_generated.
    It will generate 2 * batch_size mock data items.
    Prompts are selected from a pre-defined list of 32 real questions,
    cycling through them if more items are needed.

    The prompt format can be controlled by the 'mock_api_type' config:
    - 'completion': generates simple string prompts for completion API
    - 'generate': generates message numpy array format for chat completion API with system and user roles
    """

    def __init__(self, config: DictConfig, is_validate: bool = False):
        super().__init__(config, is_validate)
        self._config: DictConfig = config
        self._data: pd.DataFrame = pd.DataFrame()  # Initialize as empty DataFrame
        self._mock_data_size: int = config.batch_size * 16
        self._fetched_data_index: int = 0
        # Get API type from config, default to 'completion'
        self._api_type: str = config.get("mock_api_type", "completion")

        if self._api_type not in ["completion", "generate"]:
            raise ValueError(
                f"Invalid api_type: {self._api_type}. Must be 'completion' or 'generate'"
            )

        self._read_file()
        logger.info(f"MockDataLoader initialized successfully with api_type={self._api_type}")

    def _read_file(self) -> None:
        """
        Generate mock data for testing purposes and store in DataFrame.
        Uses pre-defined list of real questions, cycling through them if needed.

        The prompt format depends on the api_type:
        - 'completion': generates simple string prompts
        - 'generate': generates message numpy array format with system and user roles:
          np.array([{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
        """
        logger.info(f"MockDataLoader: Generating {self._mock_data_size} mock data items")

        # Create mock data as list of dictionaries
        mock_data_list = []
        for i in range(self._mock_data_size):
            # Cycle through the pre-defined questions list
            question_index = i % len(MOCK_QUESTIONS)
            question_text = MOCK_QUESTIONS[question_index]

            # Format prompt based on api_type
            prompt: str | np.ndarray[Any, Any]
            if self._api_type == "completion":
                prompt = question_text
            elif self._api_type == "generate":
                prompt = np.array(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Please answer the following question clearly and accurately.",
                        },
                        {"role": "user", "content": question_text},
                    ]
                )
            else:
                raise ValueError(f"Invalid api_type: {self._api_type}")

            mock_item = {
                "id": i,
                "prompt": prompt,
                "mock_generated": True,
                "ground_truth": "whatever",
            }
            mock_data_list.append(mock_item)

        # Convert to DataFrame
        self._data = pd.DataFrame(mock_data_list)
        logger.info(
            f"MockDataLoader: Generated {len(self._data)} mock data items from {len(MOCK_QUESTIONS)} pre-defined questions (api_type={self._api_type})"
        )

    @override
    def add_item_back(self, item: dict[str, Any]) -> None:
        """
        Add a data item to the back of the data

        Args:
            item: Data item to add
        """
        # Convert to DataFrame row and append
        new_row = pd.DataFrame([item])
        self._data = pd.concat([self._data, new_row], ignore_index=True)
        logger.info("MockDataLoader: Added item to back")

    @override
    def _fetch_batch_data(self) -> list[dict[str, Any]]:
        """
        Fetch one batch of items from the DataFrame

        Returns:
            list[dict[str, Any]]: List of data items (up to batch_size)
        """
        batch_items = []
        end_index = min(self._fetched_data_index + self._batch_size, len(self._data))

        for i in range(self._fetched_data_index, end_index):
            if i < len(self._data):
                row = self._data.iloc[i]
                item = row.to_dict()
                # Simulate some data processing delay
                time.sleep(random.uniform(0.01, 0.05))
                batch_items.append(item)

        self._fetched_data_index = end_index

        return batch_items

    @override
    def is_finished(self) -> bool:
        """
        Return True when all data has been fetched.

        Returns:
            True if all data has been fetched, False otherwise
        """
        return self._fetched_data_index >= len(self._data)

    @override
    def _reset_iterator(self) -> None:
        """
        Reset the mock data loader iterator to the beginning.

        Resets the fetched data index to 0 so data can be iterated again.
        """
        self._fetched_data_index = 0
