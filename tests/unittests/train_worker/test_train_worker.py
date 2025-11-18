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
Tests for TrainWorker
"""

from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from nexrl.mock import MockActorWorkerClient
from nexrl.nexrl_types import Batch
from nexrl.train_batch_pool import TrainBatchPool
from nexrl.train_worker import TrainWorker
from nexrl.weight_sync.weight_sync_controller import WeightSyncController


@pytest.fixture
def train_worker_config():
    """Configuration for TrainWorker testing"""
    return OmegaConf.create(
        {
            "total_train_steps": 10,
            "save_freq": 5,
            "checkpoint_path": "/tmp/test_checkpoints",
            "sync_weight_path": "/tmp/test_sync_weights",
            "remove_previous_ckpt": False,
            "train_service": {
                "backend": "mock",
                "url": "http://localhost:8080",
                "identifier": "test_worker",
            },
        }
    )


@pytest.fixture
def train_worker(train_worker_config):
    """Create TrainWorker with mock backend"""
    with patch(
        "nexrl.train_worker.create_train_service_client",
        return_value=MockActorWorkerClient("test", "test"),
    ):
        worker = TrainWorker(train_worker_config)
    return worker


def test_train_worker_initialization(train_worker):
    """Test TrainWorker initialization"""
    assert train_worker._config is not None
    assert train_worker._train_step == 0
    assert train_worker._stop_event is not None
    assert train_worker._train_service_client is not None
    assert isinstance(train_worker.training_stats, dict)


def test_train_worker_set_module_references(train_worker, train_worker_config):
    """Test setting module references"""
    # Create mock train batch pool and weight sync controller
    train_batch_pool = TrainBatchPool(
        OmegaConf.create({"max_size": 100, "model_tags": ["default"]})
    )

    class MockWeightSyncController:
        def train_worker_notify_weight_update(self, worker_name, model_tag):
            pass

    weight_sync_controller = MockWeightSyncController()

    train_worker.set_module_references(train_batch_pool, weight_sync_controller)

    assert train_worker._train_batch_pool is train_batch_pool
    assert train_worker._weight_sync_controller is weight_sync_controller


def test_train_worker_get_train_step(train_worker):
    """Test getting train step"""
    assert train_worker.get_train_step() == 0

    train_worker._train_step = 5
    assert train_worker.get_train_step() == 5


def test_train_worker_set_train_step(train_worker):
    """Test setting train step"""
    train_worker.set_train_step(10)
    assert train_worker._train_step == 10


def test_train_worker_module_name(train_worker):
    """Test train worker module name functionality"""
    train_worker.set_module_name("test_train_worker")
    assert train_worker.get_module_name() == "test_train_worker"


def test_train_worker_health_check(train_worker):
    """Test train worker health check"""
    assert train_worker.health_check() is True


def test_train_worker_initialize_workers_mock(train_worker):
    """Test initialize_workers with mock backend"""
    # Should not raise an error
    train_worker.initialize_workers()


def test_train_worker_reduce_metrics():
    """Test reduce_metrics static method"""
    import numpy as np

    metrics = {
        "loss": [1.0, 2.0, 3.0],
        "accuracy": [0.5, 0.6, 0.7],
    }

    reduced = TrainWorker.reduce_metrics(metrics)

    assert reduced["loss"] == 2.0  # mean of [1, 2, 3]
    assert abs(reduced["accuracy"] - 0.6) < 1e-6  # mean of [0.5, 0.6, 0.7]


def test_train_worker_step(train_worker, train_worker_config):
    """Test train worker step method"""
    # Setup module references
    train_batch_pool = TrainBatchPool(
        OmegaConf.create({"max_size": 100, "model_tags": ["default"]})
    )

    class MockWeightSyncController:
        def train_worker_notify_weight_update(self, worker_name, model_tag):
            pass

    class MockActivityTracker:
        def track(self, module_name, activity_type):
            from contextlib import contextmanager

            @contextmanager
            def dummy_context():
                yield

            return dummy_context()

        def set_training_step(self, step):
            pass

        def experiment_logger_post(self, backend, data, step=None):
            pass

    train_worker.set_module_references(train_batch_pool, MockWeightSyncController())
    train_worker._activity_tracker = MockActivityTracker()

    # Create a simple batch
    batch_data = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "labels": torch.tensor([[2, 3, 4, 5]]),
    }
    batch_metadata = {"batch_size": 1, "model_tag": "default"}
    batch = Batch(values=batch_data, metadata=batch_metadata)

    # Execute step
    train_worker._step(batch)

    # Check training stats updated
    assert train_worker.training_stats["batches_processed"] == 1
    assert train_worker.training_stats["total_samples"] == 1
