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
Worker Process for ActorWorker

This module runs ActorWorker as a standalone worker process that
communicates with the API server via ZMQ. All ranks (including rank 0) run
as workers in this architecture.

Usage:
    torchrun --nproc_per_node=4 worker_process.py --config config.yaml --role actor

Requirements:
    - torch
    - tensordict
    - numpy
    - zmq
"""

import argparse
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torch.distributed
import zmq
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from ..utils.config_loader import load_nextrainer_config
from ..utils.protocol import DataProto

# Import the protocol first (no circular dependency)


# Lazy imports to avoid circular dependencies
FSDPModelWorker = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# Base Worker Classes (formerly from worker.py)
# ==============================================================================


class Worker:
    """
    Simplified Worker base class that provides basic functionality needed for
    distributed training without external dependencies.

    This is a minimal implementation that focuses on the essential worker
    functionality needed by ActorWorker.
    """

    def __init__(self, cuda_visible_devices=None):
        """Initialize the worker with basic distributed setup"""
        # Setup basic distributed info
        self._setup_distributed_info()

        # Setup CUDA if available
        if cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    def _setup_distributed_info(self):
        """Setup basic distributed information"""
        # Get distributed info from environment or use defaults
        self._rank = int(os.environ.get("RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

        # Set master address and port if not set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12345"

        self._master_addr = os.environ["MASTER_ADDR"]
        self._master_port = os.environ["MASTER_PORT"]

    @property
    def rank(self):
        """Get the rank of this worker"""
        return self._rank

    @property
    def world_size(self):
        """Get the total number of workers"""
        return self._world_size

    @property
    def local_rank(self):
        """Get the local rank of this worker"""
        return self._local_rank

    def get_master_addr_port(self):
        """Get the master address and port"""
        return self._master_addr, self._master_port

    def get_cuda_visible_devices(self):
        """Get the CUDA visible devices configuration"""
        return os.environ.get("CUDA_VISIBLE_DEVICES", "not set")


@dataclass
class DistRankInfo:
    """Distributed rank information"""

    tp_rank: int = 0
    dp_rank: int = 0
    pp_rank: int = 0
    cp_rank: int = 0
    ep_rank: int = 0


@dataclass
class DistGlobalInfo:
    """Distributed global information"""

    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1


# ==============================================================================
# ZMQ-based Worker Process Coordination
# ==============================================================================

# Global worker instance (typed as Any to avoid circular import issues)
worker = None


def get_worker_class(backend):
    """Select worker class based on backend configuration"""
    # Only FSDP backend is supported
    if backend not in ["fsdp"]:
        raise ValueError(f"Unsupported backend: {backend}. Only 'fsdp' is supported.")

    global FSDPModelWorker
    if FSDPModelWorker is None:
        from ..fsdp_worker.fsdp_workers import ModelWorker as FSDPModelWorker
    return FSDPModelWorker


class WorkerZMQCoordinator:
    """ZMQ-based coordinator for worker processes"""

    def __init__(
        self,
        rank: int,
        world_size: int,
        base_port: Optional[int] = None,
        identifier: Optional[str] = None,
        api_server_url: Optional[str] = None,
        dispatch_mode: Optional[str] = None,
        backend: Optional[str] = "fsdp",
    ):
        self.rank = rank
        self.world_size = world_size
        self.identifier = identifier
        self.context = zmq.Context()
        self.dispatch_mode = dispatch_mode

        # Extract hostname from API server URL
        parsed_url = urlparse(api_server_url or "http://localhost:8000")
        hostname = parsed_url.hostname
        # Ensure master_addr is a string
        self.master_addr: str = hostname if hostname else "localhost"

        # Heartbeat control
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_stop_event = threading.Event()

        # Update API server URL to use master address if it's using localhost
        if api_server_url is None:
            api_server_url = f"http://{self.master_addr}:8000"
        elif "localhost" in api_server_url or "127.0.0.1" in api_server_url:
            # Replace localhost/127.0.0.1 with master address for multi-node support
            api_server_url = api_server_url.replace("localhost", self.master_addr).replace(
                "127.0.0.1", self.master_addr
            )

        self.api_server_url: str = api_server_url
        logger.info(f"Rank {rank}: API server URL: {self.api_server_url}")

        # Get base port from API server if not provided
        if base_port is None:
            base_port, self.identifier, self.dispatch_mode = self._register_with_api_server()

        self.base_port = base_port

        # Worker: Subscriber to receive commands from API server (for simple commands)
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://{self.master_addr}:{base_port}")
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages

        # Worker: Pusher to send results to API server
        self.pusher = self.context.socket(zmq.PUSH)
        self.pusher.connect(f"tcp://{self.master_addr}:{base_port + 1}")

        # Worker: PULL socket to receive individual data chunks for DP-style operations
        self.data_puller = self.context.socket(zmq.PULL)
        self.data_puller.connect(f"tcp://{self.master_addr}:{base_port + 2 + rank}")

        self.backend = backend

        logger.info(
            f"Rank {rank}: ZMQ coordinator connected to {self.master_addr}:{base_port} (SUB), {base_port + 1} (PUSH), and {base_port + 2 + rank} (PULL for DP data)"
        )
        logger.info(
            f"Rank {rank}: Worker group identifier: {self.identifier}, dispatch mode: {self.dispatch_mode}"
        )

        # Give sockets time to connect
        time.sleep(1)

        # Start heartbeat thread for rank 0
        if rank == 0 and self.api_server_url and self.identifier:
            self._start_heartbeat()

    def _register_with_api_server(self) -> tuple:
        """Register with API server and get ZMQ port allocation

        Each rank independently queries the API. Rank 0 registers the group (idempotible),
        other ranks query for the registration. No collective communication needed.

        Returns:
            Tuple of (base_port, identifier, dispatch_mode)
        """
        session = requests.Session()
        request_timeout = 30

        # Rank 0: Register the worker group (or verify existing registration)
        if self.rank == 0:
            try:
                register_data = {
                    "world_size": self.world_size,
                    "dispatch_mode": self.dispatch_mode or "scatter",
                }

                if self.identifier:
                    register_data["identifier"] = self.identifier

                logger.info(
                    f"Rank 0: Registering worker group with API server at {self.api_server_url}"
                )
                response = session.post(
                    f"{self.api_server_url}/register_worker_group",
                    json=register_data,
                    timeout=request_timeout,
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Failed to register worker group: {response.text}")

                result = response.json()
                base_port = result["zmq_base_port"]
                identifier = result["identifier"]
                dispatch_mode = result["dispatch_mode"]

                logger.info(
                    f"Rank 0: Registered worker group '{identifier}' with base port {base_port}"
                )

                return base_port, identifier, dispatch_mode

            except Exception as e:
                logger.error(f"Rank 0: Failed to register with API server: {e}")
                raise

        # Non-rank-0: Query API for existing registration
        else:
            # Use identifier if provided, otherwise we need to wait and retry
            # GPU lock acquisition during registration may take time, so we wait up to 10 minutes
            max_retries = 120  # 120 retries * 5 seconds = 600 seconds = 10 minutes total
            retry_delay = 5.0  # 5 seconds between retries (lower frequency)

            for attempt in range(max_retries):
                try:
                    # If identifier is known, query directly
                    if self.identifier:
                        logger.info(
                            f"Rank {self.rank}: Querying API for worker group '{self.identifier}'"
                        )
                        response = session.get(
                            f"{self.api_server_url}/get_worker_group_info",
                            params={"identifier": self.identifier},
                            timeout=request_timeout,
                        )

                        if response.status_code == 200:
                            result = response.json()
                            base_port = result["zmq_base_port"]
                            identifier = result["identifier"]
                            dispatch_mode = result["dispatch_mode"]

                            logger.info(
                                f"Rank {self.rank}: Got base port {base_port} and identifier '{identifier}' from API"
                            )
                            return base_port, identifier, dispatch_mode
                        elif response.status_code == 404:
                            # Group not registered yet, wait and retry
                            logger.debug(
                                f"Rank {self.rank}: Worker group '{self.identifier}' not registered yet (attempt {attempt + 1}/{max_retries})"
                            )
                        else:
                            logger.warning(
                                f"Rank {self.rank}: Unexpected response {response.status_code}: {response.text}"
                            )
                    else:
                        # No identifier provided - this shouldn't happen in normal flow
                        logger.warning(
                            f"Rank {self.rank}: No identifier provided, cannot query API (attempt {attempt + 1}/{max_retries})"
                        )

                    # Wait before retry
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)

                except Exception as e:
                    logger.warning(
                        f"Rank {self.rank}: API query failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)

            # All retries exhausted
            total_wait_time = max_retries * retry_delay
            raise RuntimeError(
                f"Rank {self.rank}: Failed to get worker group registration from API after {max_retries} attempts "
                f"({total_wait_time:.0f} seconds / {total_wait_time/60:.1f} minutes). "
                f"Identifier: {self.identifier or 'None'}. "
                f"This may indicate that rank 0 registration is still waiting for GPU lock."
            )

    def _start_heartbeat(self):
        """Start heartbeat thread (rank 0 only)"""
        logger.info(
            f"Rank {self.rank}: Starting heartbeat thread for worker group '{self.identifier}'"
        )
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name=f"heartbeat-{self.identifier}"
        )
        self.heartbeat_thread.start()

    def _heartbeat_loop(self):
        """Heartbeat loop - sends heartbeat to API server every 5 seconds"""
        session = requests.Session()
        heartbeat_timeout = 5  # 5 second timeout for requests

        while not self.heartbeat_stop_event.is_set():
            try:
                # Send heartbeat to API server
                response = session.post(
                    f"{self.api_server_url}/heartbeat",
                    json={
                        "identifier": self.identifier,
                        "rank": self.rank,
                        "world_size": self.world_size,
                        "timestamp": time.time(),
                    },
                    timeout=heartbeat_timeout,
                )

                if response.status_code == 200:
                    logger.debug(
                        f"Rank {self.rank}: Heartbeat sent successfully for group '{self.identifier}'"
                    )
                else:
                    logger.warning(
                        f"Rank {self.rank}: Heartbeat failed with status {response.status_code}: {response.text}"
                    )

            except Exception as e:
                logger.error(f"Rank {self.rank}: Failed to send heartbeat: {e}")

            # Wait 5 seconds before next heartbeat
            self.heartbeat_stop_event.wait(5)

    def worker_loop(self):
        """Main worker loop to handle commands from API server"""
        logger.info(f"Rank {self.rank}: Starting ZMQ worker loop")

        # Set up polling for both subscriber and data_puller
        poller = zmq.Poller()
        poller.register(self.subscriber, zmq.POLLIN)
        poller.register(self.data_puller, zmq.POLLIN)

        while True:
            try:
                # Poll for messages with a short timeout
                socks = dict(poller.poll(timeout=100))  # 100ms timeout

                # Check for command messages (simple commands via subscriber)
                if self.subscriber in socks and socks[self.subscriber] == zmq.POLLIN:
                    message = self.subscriber.recv_pyobj(zmq.NOBLOCK)
                    operation = message.get("operation")

                    if operation == "command":
                        # Handle simple commands
                        command = message.get("command")
                        kwargs = message.get("kwargs", {})
                        logger.info(f"Rank {self.rank}: Received command: {command}")

                        try:
                            result = self._execute_command(command, **kwargs)
                            response = {"rank": self.rank, "success": True, "result": result}
                        except Exception as e:
                            tb = traceback.format_exc()
                            logger.error(f"Rank {self.rank}: Command failed: {e}")
                            logger.error(f"Rank {self.rank}: Traceback:\n{tb}")
                            response = {
                                "rank": self.rank,
                                "success": False,
                                "error": str(e),
                                "traceback": tb,
                            }

                        # Send result back to API server
                        self.pusher.send_pyobj(response)

                        # Exit worker loop if destroy command
                        if command == "destroy":
                            logger.info(
                                f"Rank {self.rank}: Exiting worker loop after destroy command"
                            )
                            return

                    elif operation in [
                        "compute_log_prob",
                        "update_actor",
                        "generate_sequences",
                        "compute_ref_log_prob",
                    ]:
                        # Handle broadcast DataProto operations (same data to all workers)
                        data_proto = message.get("data")
                        logger.info(f"Rank {self.rank}: Received broadcast operation: {operation}")

                        try:
                            result = self._execute_operation(operation, data_proto)
                            response = {"rank": self.rank, "success": True, "result": result}
                        except Exception as e:
                            tb = traceback.format_exc()
                            logger.error(f"Rank {self.rank}: Broadcast operation failed: {e}")
                            logger.error(f"Rank {self.rank}: Traceback:\n{tb}")
                            response = {
                                "rank": self.rank,
                                "success": False,
                                "error": str(e),
                                "traceback": tb,
                            }

                        # Send result back to API server
                        self.pusher.send_pyobj(response)

                # Check for scatter data messages (individual data chunks via data_puller)
                if self.data_puller in socks and socks[self.data_puller] == zmq.POLLIN:
                    message = self.data_puller.recv_pyobj(zmq.NOBLOCK)
                    operation = message.get("operation")

                    if operation in [
                        "compute_log_prob",
                        "update_actor",
                        "generate_sequences",
                        "compute_ref_log_prob",
                    ]:
                        # Handle scatter DataProto operations (data chunks sent to individual workers)
                        data_proto = message.get("data")
                        target_rank = message.get("rank", self.rank)
                        logger.info(
                            f"Rank {self.rank}: Received scatter operation: {operation} (target rank: {target_rank})"
                        )

                        # Verify this message is intended for this rank
                        if target_rank != self.rank:
                            logger.warning(
                                f"Rank {self.rank}: Received message intended for rank {target_rank}, ignoring"
                            )
                            continue

                        try:
                            result = self._execute_operation(operation, data_proto)
                            response = {"rank": self.rank, "success": True, "result": result}
                        except Exception as e:
                            tb = traceback.format_exc()
                            logger.error(f"Rank {self.rank}: Scatter operation failed: {e}")
                            logger.error(f"Rank {self.rank}: Traceback:\n{tb}")
                            response = {
                                "rank": self.rank,
                                "success": False,
                                "error": str(e),
                                "traceback": tb,
                            }

                        # Send result back to API server
                        self.pusher.send_pyobj(response)
                    else:
                        logger.warning(f"Rank {self.rank}: Unknown scatter operation: {operation}")
                        response = {
                            "rank": self.rank,
                            "success": False,
                            "error": f"Unknown scatter operation: {operation}",
                        }
                        self.pusher.send_pyobj(response)

            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Rank {self.rank}: Error in worker loop: {e}")
                logger.error(f"Rank {self.rank}: Worker loop traceback:\n{tb}")
                time.sleep(1)

    def _execute_command(self, command: str, **kwargs) -> Any:
        """Execute a simple command on the local worker"""
        global worker

        if command == "initialize":
            return self._initialize_worker(**kwargs)
        elif command == "init_model":
            return self._init_model()
        elif command == "load_checkpoint":
            return self._load_checkpoint(**kwargs)
        elif command == "save_checkpoint":
            return self._save_checkpoint(**kwargs)
        elif command == "convert_checkpoint_to_huggingface":
            return self._convert_checkpoint(**kwargs)
        elif command == "barrier":
            return self._barrier()
        elif command == "worker_info":
            return self._worker_info()
        elif command == "get_rank_info":
            return self._get_rank_info()
        elif command == "offload_actor":
            return self._offload_actor(**kwargs)
        elif command == "load_actor":
            return self._load_actor(**kwargs)
        elif command == "destroy":
            return self._destroy_worker(**kwargs)
        else:
            raise ValueError(f"Unknown command: {command}")

    def _initialize_worker(
        self,
        config_path: str | None = None,
        config_dict: dict[str, Any] | None = None,
        role: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Initialize the worker"""
        global worker

        if worker is not None:
            print("message: Worker already initialized")
            return {"message": "Worker already initialized"}

        # Update identifier if provided in kwargs
        if "worker_group_id" in kwargs and kwargs["worker_group_id"]:
            if not self.identifier:
                self.identifier = kwargs["worker_group_id"]
                logger.info(
                    f"Rank {self.rank}: Updated worker group identifier to: {self.identifier}"
                )

        # Load configuration
        if config_path:
            # Use enhanced config loader if available
            if load_nextrainer_config is not None:
                worker_config = load_nextrainer_config(config_path)
            else:
                worker_config = OmegaConf.load(config_path)
        elif config_dict:
            worker_config = OmegaConf.create(config_dict)
            worker_config = worker_config.config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        # Create worker instance based on backend
        WorkerClass = get_worker_class(self.backend)
        try:
            worker = WorkerClass(
                config=worker_config.actor, role=role, reward_fn=kwargs.get("reward_fn")
            )
        except Exception as e:
            logger.error(f"Rank {self.rank}: Error initializing worker: {e}")
            raise e

        logger.info(f"Rank {self.rank}: Worker initialized with role: {role}")

        return {"message": f"Worker initialized with role: {role}", "role": role}

    def _init_model(self) -> dict[str, Any]:
        """Initialize the model"""
        global worker

        if worker is None:
            raise RuntimeError("Worker not initialized")

        worker.init_model()
        logger.info(f"Rank {self.rank}: Model initialized")

        return {"message": "Model initialized"}

    def _load_checkpoint(
        self, path: str, del_local_after_load: bool = True, load_weight_only: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Load checkpoint"""
        global worker

        if worker is None:
            raise RuntimeError("Worker not initialized")

        worker.load_checkpoint(
            path=path, del_local_after_load=del_local_after_load, load_weight_only=load_weight_only
        )

        return {"message": f"Checkpoint loaded from {path}"}

    def _save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        saved_fully_shared_ckpt: bool = True,
        save_weight_only: bool = False,
        remove_previous_ckpt: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Save checkpoint"""
        global worker

        if worker is None:
            raise RuntimeError("Worker not initialized")

        if not hasattr(worker, "_is_actor") or not worker._is_actor:
            # Only actor workers can save checkpoints, return success for others
            return {"message": "Non-actor worker, checkpoint save skipped"}

        worker.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            saved_fully_shared_ckpt=saved_fully_shared_ckpt,
            save_weight_only=save_weight_only,
            remove_previous_ckpt=remove_previous_ckpt,
        )

        return {"message": f"Checkpoint saved to {local_path}"}

    def _convert_checkpoint(self, local_path: str, **kwargs) -> dict[str, Any]:
        """Convert checkpoint to HuggingFace format"""
        global worker

        if worker is None:
            raise RuntimeError("Worker not initialized")

        worker.convert_ckpt_to_huggingface(local_path=local_path)

        return {"message": f"Checkpoint converted to HuggingFace format at {local_path}"}

    def _offload_actor(self, **kwargs) -> dict[str, Any]:
        """Offload actor"""
        global worker

        if worker is None:
            raise RuntimeError("Worker not initialized")

        worker.offload_actor()

        return {"message": "Actor offloaded"}

    def _load_actor(self, **kwargs) -> dict[str, Any]:
        """Load actor"""
        global worker

        if worker is None:
            raise RuntimeError("Worker not initialized")

        worker.load_actor()

        return {"message": "Actor loaded"}

    def _barrier(self, **kwargs) -> dict[str, Any]:
        """Synchronization barrier"""
        global worker

        if worker is None:
            raise RuntimeError("Worker not initialized")

        worker.barrier()

        return {"message": "Barrier completed"}

    def _worker_info(self, **kwargs) -> dict[str, Any]:
        """Get worker information"""
        global worker

        if worker is None:
            return {
                "rank": self.rank,
                "worker_initialized": False,
                "message": "Worker not initialized",
            }

        return {
            "rank": self.rank,
            "worker_initialized": True,
            "role": getattr(worker, "role", None),
            "_is_actor": getattr(worker, "_is_actor", None),
            "_is_rollout": getattr(worker, "_is_rollout", None),
            "_is_ref": getattr(worker, "_is_ref", None),
            "_is_actor_forward": getattr(worker, "_is_actor_forward", None),
        }

    def _get_rank_info(self, **kwargs) -> dict[str, Any]:
        """Get rank information for this worker (required for megatron_pp0_only dispatching)"""
        global worker

        if worker is None:
            # Fallback: provide basic rank information even without worker
            return {
                "global_rank": self.rank,
                "world_size": self.world_size,
                "worker_initialized": False,
                "backend": "unknown",
            }

        # Check if worker has get_rank_info method (Megatron workers)
        if hasattr(worker, "get_rank_info") and callable(getattr(worker, "get_rank_info")):
            try:
                return worker.get_rank_info()
            except Exception as e:
                logger.warning(f"Failed to get rank info from worker: {e}")
                # Fallback to basic info if worker method fails

        # Fallback for non-Megatron workers
        rank_info = {
            "global_rank": self.rank,
            "world_size": self.world_size,
            "worker_initialized": True,
            "role": getattr(worker, "role", None),
            "backend": getattr(worker, "backend", "fsdp"),  # Assume FSDP for non-Megatron
        }

        return rank_info

    def _destroy_worker(self, **kwargs) -> dict[str, Any]:
        """Destroy worker and clean up all distributed resources"""
        global worker

        logger.info(f"Rank {self.rank}: Destroying worker...")

        if worker is not None:
            try:
                # Call worker's destroy method to clean up distributed resources
                if hasattr(worker, "destroy"):
                    logger.info(f"Rank {self.rank}: Calling worker.destroy()...")
                    worker.destroy()
                else:
                    logger.warning(f"Rank {self.rank}: Worker does not have destroy method")

                # Clear worker reference
                worker = None
                logger.info(f"Rank {self.rank}: Worker destroyed successfully")
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error during worker destruction: {e}")
                # Still clear the reference even if destroy failed
                worker = None
        else:
            logger.info(f"Rank {self.rank}: No worker initialized, nothing to destroy")

        return {"message": "Worker destroyed successfully", "rank": self.rank}

    def _execute_operation(self, operation: str, data_proto: DataProto) -> Any:
        """Execute a DataProto operation on the local worker"""
        global worker

        if worker is None:
            raise RuntimeError("Worker not initialized")

        if operation == "compute_log_prob":
            if not (hasattr(worker, "_is_actor") and worker._is_actor) and not (
                hasattr(worker, "_is_actor_forward") and worker._is_actor_forward
            ):
                raise RuntimeError("Worker is not configured for log prob computation")
            return worker.compute_log_prob(data_proto)
        elif operation == "update_actor":
            if not hasattr(worker, "_is_actor") or not worker._is_actor:
                raise RuntimeError("Worker is not configured as actor")
            return worker.update_actor(data_proto)
        elif operation == "compute_ref_log_prob":
            if not hasattr(worker, "_is_ref") or not worker._is_ref:
                raise RuntimeError("Worker is not configured as reference")
            return worker.compute_ref_log_prob(data_proto)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def close(self):
        """Close ZMQ sockets and stop heartbeat thread"""
        # Stop heartbeat thread
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            logger.info(f"Rank {self.rank}: Stopping heartbeat thread...")
            self.heartbeat_stop_event.set()
            self.heartbeat_thread.join(timeout=2)

        # Close ZMQ sockets
        if hasattr(self, "subscriber"):
            self.subscriber.close()
        if hasattr(self, "pusher"):
            self.pusher.close()
        if hasattr(self, "data_puller"):
            self.data_puller.close()
        self.context.term()


def main():
    """Main entry point for worker process"""
    parser = argparse.ArgumentParser(description="ActorWorker Process")
    parser.add_argument(
        "--backend",
        type=str,
        default="fsdp",
        choices=["fsdp"],
        help="Backend for worker (default: fsdp)",
    )
    parser.add_argument(
        "--zmq-base-port",
        type=int,
        default=None,
        help="Base port for ZMQ communication (optional, will query API if not provided)",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        help="Worker group identifier (optional, will be auto-generated by API if not provided)",
    )
    parser.add_argument(
        "--api-server-url",
        type=str,
        default="http://localhost:8000",
        help="API server URL for registration and heartbeat (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--dispatch-mode",
        type=str,
        choices=["broadcast", "scatter"],
        help="Dispatch mode (optional, will use API default if not provided)",
    )

    args = parser.parse_args()

    # Get distributed training info
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Keep for CUDA device setting

    logger.info(
        f"Starting worker process: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}"
    )

    # Get identifier from args
    identifier = args.identifier

    # Note: If identifier is still None, it will be auto-generated by API server during registration
    if identifier:
        logger.info(f"Rank {rank}: Using worker group identifier: {identifier}")
    else:
        logger.info(f"Rank {rank}: Worker group identifier will be auto-generated by API server")

    # Set up ZMQ coordination
    # Note: If zmq_base_port is None, coordinator will register with API to get port allocation
    zmq_coordinator = WorkerZMQCoordinator(
        rank,
        world_size,
        base_port=args.zmq_base_port,
        identifier=identifier,
        api_server_url=args.api_server_url,
        dispatch_mode=args.dispatch_mode,
        backend=args.backend,
    )

    logger.info(
        f"Rank {rank}: Worker initialized with identifier '{zmq_coordinator.identifier}', entering ZMQ worker loop..."
    )

    # Set CUDA device (use LOCAL_RANK for device selection)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        logger.info(f"Rank {rank}: Set CUDA device to {local_rank} (LOCAL_RANK)")

    def cleanup_on_exit():
        """Cleanup function to be called on any exit"""
        global worker

        logger.info(f"Rank {rank}: Performing cleanup on exit...")

        # Track if worker.destroy() was called successfully
        worker_destroyed = False

        # Call worker's destroy method
        if worker is not None:
            try:
                if hasattr(worker, "destroy"):
                    logger.info(f"Rank {rank}: Calling worker.destroy() on exit...")
                    worker.destroy()
                    logger.info(f"Rank {rank}: Worker destroyed successfully on exit")
                    worker_destroyed = True
                else:
                    logger.warning(f"Rank {rank}: Worker does not have destroy method")
            except Exception as e:
                logger.error(f"Rank {rank}: Error destroying worker on exit: {e}")
            finally:
                worker = None

        # No fallback destroy - even calling destroy_process_group() causes segfaults
        # Just let the process exit and OS will clean up everything

        # Deregister from API server (rank 0 only)
        if rank == 0 and zmq_coordinator and zmq_coordinator.identifier:
            try:
                logger.info(
                    f"Rank 0: Deregistering worker group '{zmq_coordinator.identifier}' from API server..."
                )
                response = requests.post(
                    f"{args.api_server_url}/deregister",
                    params={"identifier": zmq_coordinator.identifier},
                    timeout=5,
                )
                if response.status_code == 200:
                    logger.info(
                        f"Rank 0: Successfully deregistered worker group '{zmq_coordinator.identifier}'"
                    )
                else:
                    logger.warning(
                        f"Rank 0: Failed to deregister: {response.status_code} - {response.text}"
                    )
            except Exception as e:
                logger.warning(f"Rank 0: Error deregistering from API server: {e}")

        # Close ZMQ coordinator
        if zmq_coordinator:
            zmq_coordinator.close()

    try:
        # Run ZMQ worker loop to handle operations from API server
        zmq_coordinator.worker_loop()
    except KeyboardInterrupt:
        logger.info(f"Rank {rank}: Received KeyboardInterrupt, shutting down...")
    except Exception as e:
        logger.error(f"Rank {rank}: Worker loop failed with exception: {e}", exc_info=True)
    finally:
        cleanup_on_exit()


if __name__ == "__main__":
    main()
