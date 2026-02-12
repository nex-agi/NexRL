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
Standalone API Server for ActorWorker

This module provides a RESTful HTTP interface that coordinates with distributed
ActorWorker processes via ZMQ. The API server runs as a separate process
from the workers, allowing all worker ranks (including rank 0) to function as
normal workers.

Usage:
    python api_server.py --port 8000 --world-size 4 --config config.yaml --role actor

Requirements:
    - fastapi
    - uvicorn
    - pydantic
    - torch
    - tensordict
    - numpy
    - zmq
"""

import argparse
import asyncio
import gc
import json
import logging
import logging.config
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import uvicorn
import zmq
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from tensordict import TensorDict

# Import DataProto from current project
from ..utils.protocol import DataProto

# Configure logging (include timestamps) for both application logs and uvicorn access/error logs.
# Note: uvicorn has its own loggers (uvicorn, uvicorn.error, uvicorn.access). We configure them here
# and also pass the same config to uvicorn.run().
LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s:%(name)s:%(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "format": "%(asctime)s %(levelname)s:%(name)s:%(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "access",
            "stream": "ext://sys.stderr",
        },
    },
    "root": {"level": "INFO", "handlers": ["default"]},
    "loggers": {
        "uvicorn": {"level": "INFO", "handlers": ["default"], "propagate": False},
        "uvicorn.error": {"level": "INFO", "handlers": ["default"], "propagate": False},
        "uvicorn.access": {"level": "INFO", "handlers": ["access"], "propagate": False},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Global ZMQ coordinator instances - keyed by worker group identifier
zmq_coordinators: Dict[str, "APIZMQCoordinator"] = {}

# Global heartbeat tracking - keyed by worker group identifier
heartbeat_timestamps: Dict[str, float] = {}
heartbeat_lock = threading.Lock()

# Heartbeat monitoring thread
heartbeat_monitor_thread = None
heartbeat_monitor_stop_event = threading.Event()

# Registration lock to prevent race conditions during port allocation
registration_lock = threading.Lock()

# ============================================================================
# OPERATION QUEUE SYSTEM - Parallel Per-Group Processing
# ============================================================================
#
# Purpose: Process operations for multiple worker groups in parallel.
#          Each worker group (Student, Teacher, etc.) has its own:
#          1. Operation Queue
#          2. Processor Thread
#          3. ZMQ Coordinator
#
# Key Components:
# ---------------
# 1. group_operation_queues: Dict[str, Queue] - Ops waiting for execution
# 2. group_processor_threads: Dict[str, Thread] - Dedicated execution threads
# 3. group_stop_events: Dict[str, Event] - Signals to stop threads
#
# Workflow:
# ---------
# 1. When a group registers, a new queue and processor thread are created.
# 2. submit_queued_operation puts the op into the specific group's queue.
# 3. The group's thread picks it up and executes it via its ZMQ coordinator.
# 4. Operations from different groups run completely in parallel.
#
# ============================================================================


@dataclass
class QueuedOperation:
    """Represents a queued operation"""

    operation_id: str
    operation_type: str  # 'command' or 'data_operation'
    identifier: str
    data: Dict[str, Any]
    result_future: Any  # asyncio.Event
    result: Optional[Any] = None
    enqueue_time: float = 0.0  # Timestamp when operation was enqueued
    error: Optional[str] = None
    requires_gpu: bool = True  # Whether this operation requires GPU access
    event_loop: Optional[asyncio.AbstractEventLoop] = None  # Event loop for setting the future


# Global dicts for managing parallel groups
group_operation_queues: Dict[str, Queue] = {}
group_processor_threads: Dict[str, threading.Thread] = {}
group_stop_events: Dict[str, threading.Event] = {}


def _start_group_processor(identifier: str) -> None:
    """Create and start a processor thread for the given worker group."""
    with registration_lock:
        queue = group_operation_queues.setdefault(identifier, Queue())
        stop_event = threading.Event()
        group_stop_events[identifier] = stop_event

        thread = threading.Thread(
            target=process_parallel_group_operations,
            args=(identifier,),
            daemon=True,
            name=f"processor-{identifier}",
        )
        group_processor_threads[identifier] = thread
        thread.start()

    logger.info(
        f"Spawned processor thread '{thread.name}' for group '{identifier}' (queue size={queue.qsize()})"
    )


def _ensure_group_processor(identifier: str) -> None:
    """Make sure the worker group has a live processor thread."""
    with registration_lock:
        queue_exists = identifier in group_operation_queues
        if not queue_exists:
            group_operation_queues[identifier] = Queue()

        thread = group_processor_threads.get(identifier)
        thread_alive = thread is not None and thread.is_alive()
        if thread_alive:
            return

        # Stop any lingering thread/event
        stop_event = group_stop_events.pop(identifier, None)
        if stop_event:
            stop_event.set()

        stop_event = threading.Event()
        group_stop_events[identifier] = stop_event

        thread = threading.Thread(
            target=process_parallel_group_operations,
            args=(identifier,),
            daemon=True,
            name=f"processor-{identifier}",
        )
        group_processor_threads[identifier] = thread
        thread.start()
        logger.info(f"Restarted processor thread '{thread.name}' for group '{identifier}'")


def _stop_group_processor(identifier: str, join_timeout: float = 5.0) -> None:
    """Stop and clean up the processor thread for a worker group."""
    thread = None
    with registration_lock:
        stop_event = group_stop_events.pop(identifier, None)
        if stop_event:
            stop_event.set()
        thread = group_processor_threads.pop(identifier, None)

    if thread:
        thread.join(timeout=join_timeout)


class APIZMQCoordinator:
    """ZMQ-based coordinator for the standalone API server

    In the new direct communication architecture, the API server is coordination-only:
    - PUB socket for broadcasting commands and execute signals to all workers
    - PULL socket for collecting results from all workers
    - NO data pushing - data goes directly from client to workers

    Workers bind their own PULL sockets for data reception and report their
    endpoints via /report_worker_endpoint. Direct clients connect to these
    worker-reported endpoints.
    """

    def __init__(self, world_size: int, base_port: int = 5555, dispatch_mode: str = "scatter"):
        self.world_size = world_size
        self.base_port = base_port
        self.dispatch_mode = dispatch_mode  # Always "scatter" in new architecture
        self.context = zmq.Context()

        # Reference counting for safe shutdown
        self._ref_count = 0
        self._ref_lock = threading.Lock()
        self._is_closing = False

        # API Server: Publisher to broadcast commands AND execute signals to all workers
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://*:{base_port}")

        # API Server: Collector to gather results from all workers
        self.collector = self.context.socket(zmq.PULL)
        self.collector.bind(f"tcp://*:{base_port + 1}")

        # API Server: Result forwarder to push results to clients
        # This creates a forwarding pipeline: Workers PUSH → API Server PULL → API Server PUSH → Client PULL
        self.result_forwarder = self.context.socket(zmq.PUSH)
        # Set send timeout to prevent blocking forever if no client is connected
        # 5000ms timeout allows client time to connect but prevents indefinite hangs
        self.result_forwarder.setsockopt(zmq.SNDTIMEO, 5000)
        # Set high water mark to queue messages if client is slow
        self.result_forwarder.setsockopt(zmq.SNDHWM, 10000)
        self.result_forwarder.bind(f"tcp://*:{base_port + 2}")

        # Worker endpoints: populated by workers via /report_worker_endpoint
        # Format: {rank: {"ip": "10.0.1.5", "port": 6000}, ...}
        self.worker_endpoints: Dict[int, Dict[str, Any]] = {}
        self._worker_endpoints_lock = threading.Lock()

        # NOTE: data_pushers removed - API server no longer pushes data to workers.
        # Data flows directly from DirectZMQTrainServiceClient to workers.
        # This eliminates the API server as a data bottleneck (OOM, GIL, double serialization).

        # Command result queue: forwarder stores command results here for command processor
        self._command_result_queue: List[Dict[str, Any]] = []
        self._command_result_lock = threading.Lock()

        # Start background result forwarder thread for direct-zmq mode
        self._forwarder_running = True
        self._forwarder_thread = threading.Thread(
            target=self._result_forwarder_loop, name=f"result-forwarder-{base_port}", daemon=True
        )
        self._forwarder_thread.start()

        logger.info(
            f"API Server: Coordination-only ZMQ coordinator started on ports "
            f"{base_port} (PUB for commands/execute), {base_port + 1} (PULL for results from workers), "
            f"{base_port + 2} (PUSH for results to clients)"
        )
        logger.info(
            f"API Server: Dispatch mode set to '{self.dispatch_mode}' (direct client required)"
        )
        logger.info(f"API Server: Result forwarder thread started")

        # Give sockets time to bind
        time.sleep(0.5)

    def register_worker_endpoint(self, rank: int, worker_ip: str, data_port: int) -> None:
        """Register a worker's data endpoint (called via /report_worker_endpoint)

        Workers call this after binding their PULL sockets to report their
        IP and port so that direct clients can connect to them.
        """
        with self._worker_endpoints_lock:
            self.worker_endpoints[rank] = {
                "ip": worker_ip,
                "port": data_port,
            }
        logger.info(f"Registered worker endpoint: rank={rank}, {worker_ip}:{data_port}")

    def get_worker_endpoints(self) -> Dict[int, Dict[str, Any]]:
        """Get all registered worker endpoints"""
        with self._worker_endpoints_lock:
            return dict(self.worker_endpoints)

    def broadcast_execute(self, op_id: str) -> None:
        """Broadcast execute signal for two-phase protocol

        This is Phase 2 of the two-phase scatter protocol. After the direct client
        has sent data chunks to all workers (Phase 1), it calls POST /execute
        which triggers this method to broadcast the execute signal.

        All workers receive this signal simultaneously via SUB socket and begin
        processing their pending data for this op_id. This ensures NCCL timing
        spread is minimal (<1 second) regardless of how long data streaming took.
        """
        message = {
            "phase": "execute",
            "op_id": op_id,
            "timestamp": time.time(),
        }
        serialized = pickle.dumps(message)
        self.publisher.send(serialized)
        logger.info(f"Broadcast execute signal for op_id={op_id}")

    def _result_forwarder_loop(self) -> None:
        """Background thread that forwards direct-zmq operation results to clients

        IMPORTANT: This forwarder only handles direct-zmq scatter operations (with op_id).
        Regular command results are handled by _collect_results_from_workers() directly.

        The forwarder creates a pipeline for direct-zmq operations:
        Workers (PUSH) → API Server collector (PULL) → [forward if has op_id] → API Server forwarder (PUSH) → Client (PULL)
                                                      → [pass through if command result] → _collect_results_from_workers()

        Results with op_id (direct-zmq): Forwarded to client immediately
        Results without op_id (commands): Buffered in queue for command processor
        """
        logger.info("Result forwarder loop started - forwarding direct-zmq results only")

        # Set 1 second timeout for clean shutdown
        self.collector.setsockopt(zmq.RCVTIMEO, 1000)

        last_heartbeat = time.time()
        heartbeat_interval = 10  # Log heartbeat every 10 seconds

        receive_attempts = 0
        while self._forwarder_running:
            try:
                # Periodic heartbeat to show forwarder is alive
                if time.time() - last_heartbeat > heartbeat_interval:
                    logger.info(
                        f"Forwarder heartbeat: alive and polling (queue size: {len(self._command_result_queue)}, receive attempts: {receive_attempts})"
                    )
                    last_heartbeat = time.time()
                    receive_attempts = 0  # Reset counter after heartbeat

                # Receive result from worker (using timeout, not NOBLOCK)
                receive_attempts += 1
                result = self.collector.recv_pyobj()

                # Log that we received something
                logger.info(f"Forwarder received a result (type: {type(result)})")

                # Check if this is a direct-zmq operation result (has op_id) or command result
                op_id = result.get("op_id")
                rank = result.get("rank", "unknown")

                logger.info(f"Result analysis: op_id={op_id}, rank={rank}, has_op_id={bool(op_id)}")

                if op_id:
                    # Direct-zmq operation result: forward to client
                    logger.info(
                        f"Attempting to forward direct-zmq result from rank {rank} for op_id={op_id}"
                    )
                    try:
                        self.result_forwarder.send_pyobj(result)
                        logger.info(
                            f"✓ Forwarded direct-zmq result from rank {rank} for op_id={op_id}"
                        )
                    except zmq.Again:
                        # Send timed out - no client connected or client too slow
                        logger.warning(
                            f"⚠ Send timeout forwarding result from rank {rank} for op_id={op_id}. "
                            f"Client may not be connected to result forwarder port."
                        )
                else:
                    # Command result: store in queue for command processor to collect
                    with self._command_result_lock:
                        self._command_result_queue.append(result)
                    logger.info(
                        f"Buffered command result from rank {rank} (queue size: {len(self._command_result_queue)})"
                    )

            except zmq.Again:
                # Timeout - normal, just continue to check _forwarder_running flag
                continue
            except AttributeError as e:
                # This might happen if result.get() fails
                logger.error(
                    f"AttributeError in forwarder (bad result format?): {e}", exc_info=True
                )
                logger.error(
                    f"Result that caused error: {result if 'result' in locals() else 'not yet assigned'}"
                )
                continue
            except Exception as e:
                if self._forwarder_running:
                    logger.error(f"Unexpected error in result forwarder loop: {e}", exc_info=True)
                    # Don't break, just continue to be resilient
                    continue
                else:
                    break

        logger.info("Result forwarder loop stopped")

    def broadcast_command(self, command: str, **kwargs) -> List[Dict[str, Any]]:
        """Broadcast a simple command to all workers and collect acknowledgments"""
        # Acquire reference to prevent coordinator from being closed during operation
        if not self._acquire_ref():
            raise RuntimeError("Coordinator is closing, cannot perform operation")

        try:
            message = {
                "operation": "command",
                "command": command,
                "kwargs": kwargs,
                "timestamp": time.time(),
            }
            # Serialize manually and release GIL
            serialized = pickle.dumps(message)
            time.sleep(0)  # Release GIL to allow heartbeat processing
            self.publisher.send(serialized)

            logger.info(f"Broadcasted command: {command} to {self.world_size} workers")

            # Collect acknowledgments from all workers using improved collection logic
            # Note: broadcast_command doesn't raise on failures, just returns the results
            results = self._collect_results_from_workers(
                operation_name=f"command_{command}",
                total_timeout_ms=3600000,  # 60 minutes total
                per_recv_timeout_ms=600000,  # 10 minutes per recv attempt
            )

            return results
        finally:
            self._release_ref()

    def _collect_results_from_workers(
        self,
        operation_name: str,
        total_timeout_ms: int = 3600000,
        per_recv_timeout_ms: int = 600000,
    ) -> List[Dict[str, Any]]:
        """
        Collect results from all workers with improved timeout and retry logic.

        Args:
            operation_name: Name of the operation for logging
            total_timeout_ms: Total timeout in milliseconds (default: 10 minutes)
            per_recv_timeout_ms: Timeout for each recv attempt in milliseconds (default: 30 seconds)
            raise_on_failure: Whether to raise RuntimeError on failures (default: True)

        Returns:
            List of result dictionaries sorted by rank

        Raises:
            RuntimeError: If raise_on_failure=True and total timeout is reached or some ranks failed with errors
        """
        start_time = time.time() * 1000  # Convert to milliseconds
        results_by_rank: Dict[int, Any] = {}  # Use dict to deduplicate by rank

        logger.info(
            f"Collecting results for {operation_name} from {self.world_size} workers, total timeout: {total_timeout_ms}ms"
        )

        while len(results_by_rank) < self.world_size:
            # Calculate remaining time
            elapsed = time.time() * 1000 - start_time
            remaining_total = total_timeout_ms - elapsed

            if remaining_total <= 0:
                # Total timeout reached
                missing_ranks = set(range(self.world_size)) - set(results_by_rank.keys())
                logger.error(
                    f"Total timeout ({total_timeout_ms}ms) reached for {operation_name}. "
                    f"Received {len(results_by_rank)}/{self.world_size} results. "
                    f"Missing ranks: {sorted(missing_ranks)}"
                )
                raise RuntimeError(
                    f"Total timeout ({total_timeout_ms}ms) reached for {operation_name}. Received {len(results_by_rank)}/{self.world_size} results. Missing ranks: {sorted(missing_ranks)}"
                )

            # First, check if forwarder has buffered any command results for us
            result = None
            if hasattr(self, "_command_result_queue"):
                with self._command_result_lock:
                    if self._command_result_queue:
                        result = self._command_result_queue.pop(0)
                        logger.debug(
                            f"Retrieved command result from forwarder queue (queue size: {len(self._command_result_queue)})"
                        )

            # If no buffered result, wait a bit and try again (don't block on socket since forwarder handles it)
            if result is None:
                time.sleep(0.1)  # 100ms poll interval
                continue

            try:
                # Extract rank from result
                rank = result.get("rank", None)
                if rank is None:
                    logger.error(
                        f"Received result without rank field for {operation_name}: {result}"
                    )
                    continue

                # Check if this is a duplicate
                if rank in results_by_rank:
                    logger.error(
                        f"Received duplicate result from rank {rank} for {operation_name}, "
                        f"ignoring (keeping first result)"
                    )
                    continue

                # Store the result
                results_by_rank[rank] = result
                logger.info(
                    f"Received result from rank {rank} for {operation_name} "
                    f"({len(results_by_rank)}/{self.world_size})"
                )

                # Release GIL after each result to allow heartbeat processing
                time.sleep(0)
            except Exception as e:
                # Real error during processing - log but continue trying to collect other results
                logger.error(f"Error receiving result for {operation_name}: {e}", exc_info=True)
                raise e

        # Convert to sorted list
        results = [results_by_rank[rank] for rank in sorted(results_by_rank.keys())]
        return results

    def _acquire_ref(self) -> bool:
        """Acquire a reference to this coordinator. Returns False if closing."""
        with self._ref_lock:
            if self._is_closing:
                return False
            self._ref_count += 1
            return True

    def _release_ref(self):
        """Release a reference to this coordinator."""
        with self._ref_lock:
            self._ref_count -= 1
            if self._ref_count < 0:
                logger.error("Reference count went negative!")
                self._ref_count = 0

    def can_close(self) -> bool:
        """Check if coordinator can be safely closed (no active operations)."""
        with self._ref_lock:
            return self._ref_count == 0

    def close(self):
        """Close ZMQ sockets and context. Waits for active operations to complete."""
        # Mark as closing to prevent new operations
        with self._ref_lock:
            self._is_closing = True

        # Stop forwarder thread
        if hasattr(self, "_forwarder_running"):
            self._forwarder_running = False
            if hasattr(self, "_forwarder_thread"):
                self._forwarder_thread.join(timeout=5.0)
                logger.info("Result forwarder thread stopped")

        # Wait for active operations to complete (with timeout)
        max_wait_time = 60  # 60 seconds
        wait_start = time.time()
        while not self.can_close():
            if time.time() - wait_start > max_wait_time:
                logger.warning(
                    f"Timeout waiting for operations to complete. Forcing close with {self._ref_count} active operations."
                )
                break
            time.sleep(0.1)

        # Close sockets
        if hasattr(self, "publisher"):
            self.publisher.close()
        if hasattr(self, "collector"):
            self.collector.close()
        if hasattr(self, "result_forwarder"):
            self.result_forwarder.close()
        # NOTE: data_pushers removed - API server is coordination-only

        # Terminate context
        self.context.term()


# Import data models and serialization utilities from current project
from ..utils.core_utils import (
    CheckpointRequest,
    ConvertCheckpointRequest,
    SaveCheckpointRequest,
    StatusResponse,
)


def monitor_heartbeats():
    """Background thread to monitor worker heartbeats and remove stale workers"""
    global zmq_coordinators, heartbeat_timestamps, heartbeat_lock
    global group_stop_events, group_processor_threads, group_operation_queues

    logger.info("Starting heartbeat monitoring thread")
    # Increased to 5 minutes to handle GIL contention during large data operations
    # (pickle deserialization and tensor concatenation can hold the GIL for extended periods)
    HEARTBEAT_TIMEOUT = 300  # 300 seconds (5 minutes)
    CHECK_INTERVAL = 10  # Check every 10 seconds

    while not heartbeat_monitor_stop_event.is_set():
        try:
            current_time = time.time()
            stale_groups = []

            # Copy heartbeat timestamps under lock to minimize lock hold time
            with heartbeat_lock:
                heartbeat_copy = heartbeat_timestamps.copy()

            # Check for stale heartbeats without holding the lock
            for identifier, last_heartbeat in heartbeat_copy.items():
                time_since_heartbeat = current_time - last_heartbeat

                if time_since_heartbeat > HEARTBEAT_TIMEOUT:
                    logger.warning(
                        f"Worker group '{identifier}' heartbeat is stale "
                        f"({time_since_heartbeat:.1f}s since last heartbeat). Marking for removal."
                    )
                    stale_groups.append(identifier)

            # Remove stale worker groups
            for identifier in stale_groups:
                try:
                    logger.info(f"Removing stale worker group '{identifier}'")

                    # Stop processor thread
                    if identifier in group_stop_events:
                        logger.info(f"Stopping processor thread for stale group '{identifier}'")
                        _stop_group_processor(identifier)

                    # Remove queue
                    group_operation_queues.pop(identifier, None)

                    # Close coordinator
                    if identifier in zmq_coordinators:
                        coordinator = zmq_coordinators[identifier]

                        # Check if coordinator can be safely closed
                        if not coordinator.can_close():
                            logger.warning(
                                f"Worker group '{identifier}' has active operations "
                                f"({coordinator._ref_count} refs). Will retry later."
                            )
                            continue

                        # Safe to close
                        coordinator.close()
                        del zmq_coordinators[identifier]

                    # Remove heartbeat tracking
                    with heartbeat_lock:
                        if identifier in heartbeat_timestamps:
                            del heartbeat_timestamps[identifier]

                    logger.info(f"Successfully removed stale worker group '{identifier}'")
                except Exception as e:
                    logger.error(f"Failed to remove stale worker group '{identifier}': {e}")

        except Exception as e:
            logger.error(f"Error in heartbeat monitoring thread: {e}", exc_info=True)

        # Wait before next check
        heartbeat_monitor_stop_event.wait(CHECK_INTERVAL)

    logger.info("Heartbeat monitoring thread stopped")


def process_parallel_group_operations(identifier: str):
    """Background thread to process operations for a specific worker group

    This thread handles the queue for a single worker group, allowing completely
    parallel execution across different worker groups (e.g., student vs teacher).
    """
    global group_operation_queues, group_stop_events, zmq_coordinators

    logger.info(f"Starting processor thread for worker group '{identifier}'")

    while True:
        queue = group_operation_queues.get(identifier)
        stop_event = group_stop_events.get(identifier)

        if queue is None or stop_event is None:
            logger.info(f"Group '{identifier}' structures missing; terminating processor thread")
            break

        if stop_event.is_set():
            break

        queued_op = None
        try:
            queued_op = queue.get(timeout=1.0)
        except Empty:
            continue
        except Exception as queue_exc:
            logger.error(f"Queue error for group '{identifier}': {queue_exc}", exc_info=True)
            time.sleep(0.5)
            continue

        if not queued_op:
            continue

        logger.info(
            f"Processing queued operation: {queued_op.operation_id} ({queued_op.operation_type}) for group '{queued_op.identifier}'"
        )

        try:
            coordinator = None
            is_register = queued_op.operation_type == "register"

            if not is_register:
                if queued_op.identifier not in zmq_coordinators:
                    raise RuntimeError(f"Worker group '{queued_op.identifier}' not found")
                coordinator = zmq_coordinators[queued_op.identifier]

            if queued_op.operation_type == "register":
                result = {"status": "registered", "identifier": queued_op.identifier}
            elif queued_op.operation_type == "command":
                assert coordinator is not None
                result = coordinator.broadcast_command(**queued_op.data)  # type: ignore
            elif queued_op.operation_type == "data_operation":
                # Data operations are no longer supported via API server
                # Use DirectZMQTrainServiceClient instead
                raise RuntimeError(
                    "Data operations are no longer supported via API server. "
                    "Use DirectZMQTrainServiceClient (backend='direct-zmq') instead."
                )
            else:
                raise ValueError(f"Unknown operation type: {queued_op.operation_type}")

            queued_op.result = result
            queued_op.error = None
            logger.info(f"Completed queued operation: {queued_op.operation_id}")

        except Exception as op_exc:
            logger.error(
                f"Failed to process queued operation {queued_op.operation_id}: {op_exc}",
                exc_info=True,
            )
            queued_op.error = str(op_exc)
            queued_op.result = None
            raise RuntimeError(
                f"Failed to process queued operation {queued_op.operation_id}: {op_exc}"
            )

        finally:
            if queued_op.result_future and queued_op.event_loop:
                queued_op.event_loop.call_soon_threadsafe(queued_op.result_future.set)

    logger.info(f"Processor thread for worker group '{identifier}' stopped")


def get_coordinator(identifier: Optional[str] = None) -> "APIZMQCoordinator":
    """Get the ZMQ coordinator for a specific worker group

    Args:
        identifier: Worker group identifier. If None, returns the first/only coordinator

    Returns:
        APIZMQCoordinator instance

    Raises:
        HTTPException: If coordinator not found
    """
    global zmq_coordinators

    if not zmq_coordinators:
        raise HTTPException(status_code=500, detail="No ZMQ coordinators initialized")

    # If identifier is None and there's only one coordinator, return it
    if identifier is None:
        if len(zmq_coordinators) == 1:
            return next(iter(zmq_coordinators.values()))
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple worker groups exist. Please specify an identifier. Available: {list(zmq_coordinators.keys())}",
            )

    # Return specific coordinator
    if identifier not in zmq_coordinators:
        raise HTTPException(
            status_code=404,
            detail=f"Worker group '{identifier}' not found. Available: {list(zmq_coordinators.keys())}",
        )

    return zmq_coordinators[identifier]


async def submit_queued_operation(
    identifier: str, operation_type: str, data: Dict[str, Any], requires_gpu: bool = True
) -> Any:
    """Submit an operation to the queue and wait for completion (async)

    This is an async function that submits an operation to the background queue
    processor and waits asynchronously for completion. This allows FastAPI to
    handle other concurrent requests while waiting.

    Args:
        identifier: Worker group identifier
        operation_type: Type of operation ('command' or 'data_operation')
        data: Operation data
        requires_gpu: Whether this operation requires GPU access (default: True)

    Returns:
        Operation result

    Raises:
        RuntimeError: If operation fails
    """
    global group_operation_queues, zmq_coordinators

    # Check if coordinator exists BEFORE starting processor thread
    # This prevents race conditions where heartbeat timeout removes the coordinator
    # but operations are still submitted
    if identifier not in zmq_coordinators:
        raise RuntimeError(
            f"Worker group '{identifier}' not found. The group may have been removed due to "
            f"heartbeat timeout. Workers need to re-register via /register_worker_group."
        )

    _ensure_group_processor(identifier)

    # Ensure this group has a queue
    if identifier not in group_operation_queues:
        # Queue creation usually happens at registration, but this is a safety fallback
        # Note: Ideally we shouldn't create threads here implicitly
        logger.warning(
            f"Queue not found for '{identifier}', this should have happened at registration"
        )
        raise RuntimeError(f"Worker group '{identifier}' not fully registered (no queue found)")

    # Create operation with asyncio.Event
    operation_id = str(uuid.uuid4())
    result_event = asyncio.Event()
    event_loop = asyncio.get_event_loop()

    queued_op = QueuedOperation(
        operation_id=operation_id,
        operation_type=operation_type,
        identifier=identifier,
        data=data,
        result_future=result_event,
        requires_gpu=requires_gpu,
        enqueue_time=time.time(),  # Record when operation was enqueued
        event_loop=event_loop,  # Store event loop for cross-thread signaling
    )

    # Submit to group's specific queue
    group_operation_queues[identifier].put(queued_op)
    logger.info(f"Queued operation {operation_id} ({operation_type}) for group '{identifier}'")

    # Wait for completion asynchronously (allows other requests to be handled)
    await result_event.wait()

    # Check for errors
    if queued_op.error:
        raise RuntimeError(f"Queued operation failed: {queued_op.error}")

    return queued_op.result


# Create FastAPI app
app = FastAPI(
    title="ActorWorker API Server",
    description="Standalone RESTful HTTP interface for distributed ActorWorker",
    version="1.0.0",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint"""
    global zmq_coordinators
    total_workers = sum(coord.world_size for coord in zmq_coordinators.values())
    return StatusResponse(
        status="healthy",
        message="API server is running",
        workers_connected=total_workers,
        total_workers=total_workers,
    )


@app.get("/get_models")
async def get_models():
    """Get all available worker group identifiers

    Returns:
        Dict containing list of all worker group identifiers and their information
    """
    global zmq_coordinators, heartbeat_timestamps, heartbeat_lock

    models_info = {}
    current_time = time.time()

    # Copy heartbeat timestamps under lock to minimize lock hold time
    with heartbeat_lock:
        heartbeat_copy = heartbeat_timestamps.copy()

    # Process coordinator info without holding the lock
    for identifier, coordinator in zmq_coordinators.items():
        last_heartbeat = heartbeat_copy.get(identifier, None)
        time_since_heartbeat = current_time - last_heartbeat if last_heartbeat else None

        models_info[identifier] = {
            "world_size": coordinator.world_size,
            "dispatch_mode": coordinator.dispatch_mode,
            "base_port": coordinator.base_port,
            "last_heartbeat": last_heartbeat,
            "time_since_heartbeat": time_since_heartbeat,
            "healthy": time_since_heartbeat is None
            or time_since_heartbeat < 300,  # Match HEARTBEAT_TIMEOUT
            "active_operations": coordinator._ref_count,
        }

    return {
        "status": "success",
        "identifiers": list(zmq_coordinators.keys()),
        "count": len(zmq_coordinators),
        "models": models_info,
    }


class HeartbeatRequest(BaseModel):
    """Heartbeat request model"""

    identifier: str
    rank: int = 0
    world_size: Optional[int] = None
    timestamp: Optional[float] = None


@app.post("/heartbeat")
async def receive_heartbeat(request: HeartbeatRequest):
    """Receive heartbeat from worker group

    Args:
        request: Heartbeat request with identifier and optional metadata

    Returns:
        Acknowledgment of heartbeat
    """
    global heartbeat_timestamps, heartbeat_lock

    if request.rank != 0:
        logger.warning(
            f"Received heartbeat from non-rank-0 worker (rank {request.rank}) in group '{request.identifier}'"
        )

    with heartbeat_lock:
        heartbeat_timestamps[request.identifier] = time.time()

    logger.debug(
        f"Received heartbeat from worker group '{request.identifier}' (rank {request.rank}, world_size {request.world_size})"
    )

    return {
        "status": "success",
        "message": f"Heartbeat received for worker group '{request.identifier}'",
        "identifier": request.identifier,
    }


class RegisterWorkerGroupRequest(BaseModel):
    """Register worker group request model"""

    world_size: int
    identifier: Optional[str] = None
    dispatch_mode: Optional[str] = None
    api_server_url: Optional[str] = None


class InitializeWorkersRequest(BaseModel):
    """Initialize workers request model"""

    config_path: Optional[str] = None
    role: str = "actor"
    config_dict: Optional[Dict[str, Any]] = None
    identifier: Optional[str] = None
    world_size: Optional[int] = None
    zmq_base_port: Optional[int] = None
    dispatch_mode: Optional[str] = None


@app.post("/register_worker_group")
async def register_worker_group(request: RegisterWorkerGroupRequest):
    """Register a new worker group and allocate ZMQ ports

    This endpoint is called by workers to register themselves when launched externally.
    The API server creates a coordinator and allocates ports, then returns the connection info.

    Args:
        request: Registration request with world_size and optional configuration

    Returns:
        Connection information including identifier and ZMQ ports
    """
    global zmq_coordinators, heartbeat_lock, heartbeat_timestamps, registration_lock
    global group_operation_queues, group_processor_threads, group_stop_events

    try:
        identifier = request.identifier

        # Generate identifier if not provided
        if identifier is None:
            identifier = str(uuid.uuid4())
            logger.info(f"Generated new worker group identifier: {identifier}")

        is_new_group = False
        # Use lock for safe port allocation and state mutation
        with registration_lock:
            if identifier in zmq_coordinators:
                zmq_coordinator = zmq_coordinators[identifier]
                logger.info(
                    f"Worker group '{identifier}' already registered, returning existing info"
                )
            else:
                used_ports = set(coord.base_port for coord in zmq_coordinators.values())

                zmq_base_port = 5555
                while zmq_base_port in used_ports:
                    zmq_base_port += 100

                dispatch_mode = request.dispatch_mode or "scatter"
                zmq_coordinator = APIZMQCoordinator(
                    world_size=request.world_size,
                    base_port=zmq_base_port,
                    dispatch_mode=dispatch_mode,
                )

                with heartbeat_lock:
                    heartbeat_timestamps[identifier] = time.time()

                group_operation_queues[identifier] = Queue()
                zmq_coordinators[identifier] = zmq_coordinator
                is_new_group = True

                logger.info(
                    f"Registered new worker group '{identifier}' with {request.world_size} workers on ports {zmq_base_port}+"
                )

        if is_new_group:
            _start_group_processor(identifier)
        else:
            _ensure_group_processor(identifier)

        return {
            "status": "success",
            "message": f"Worker group '{identifier}' registered",
            "identifier": identifier,
            "world_size": zmq_coordinator.world_size,
            "zmq_base_port": zmq_coordinator.base_port,
            "dispatch_mode": zmq_coordinator.dispatch_mode,
            "api_server_url": request.api_server_url or f"http://localhost:8000",
        }

    except Exception as e:
        logger.error(f"Failed to register worker group: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to register worker group: {str(e)}")


@app.get("/get_worker_group_info")
async def get_worker_group_info_endpoint(identifier: Optional[str] = None):
    """Get connection information for a registered worker group

    This endpoint is called by workers or direct clients to get ZMQ connection details.
    Direct clients use this to establish direct ZMQ connections to workers.

    Args:
        identifier: Worker group identifier. If None, returns info for the first/only group.

    Returns:
        Connection information including ZMQ ports and API server host for direct connections
    """
    global zmq_coordinators

    try:
        # Use get_coordinator to handle None identifier (returns first/only group)
        zmq_coordinator = get_coordinator(identifier)

        # Get the actual identifier from coordinator if it was None
        if identifier is None:
            identifier = next(iter(zmq_coordinators.keys()))

        # Get worker endpoints for direct client connections
        worker_endpoints = zmq_coordinator.get_worker_endpoints()

        return {
            "status": "success",
            "identifier": identifier,
            "world_size": zmq_coordinator.world_size,
            "zmq_base_port": zmq_coordinator.base_port,
            "dispatch_mode": zmq_coordinator.dispatch_mode,
            "worker_endpoints": worker_endpoints,  # For direct client connections
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get worker group info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get worker group info: {str(e)}")


class ReportWorkerEndpointRequest(BaseModel):
    """Request model for reporting worker endpoint"""

    identifier: str
    rank: int
    worker_ip: str
    data_port: int


@app.post("/report_worker_endpoint")
async def report_worker_endpoint(request: ReportWorkerEndpointRequest):
    """Workers report their data endpoint after binding PULL socket

    In the new direct communication architecture, workers bind their own PULL
    sockets for data reception. After binding, each worker calls this endpoint
    to report its IP and port to the API server.

    Direct clients query /get_worker_group_info to get these endpoints and
    establish direct ZMQ connections to workers.

    Args:
        request: Contains identifier, rank, worker_ip, and data_port

    Returns:
        Success status
    """
    global zmq_coordinators

    try:
        if request.identifier not in zmq_coordinators:
            raise HTTPException(
                status_code=404,
                detail=f"Worker group '{request.identifier}' not found. Register the group first.",
            )

        zmq_coordinator = zmq_coordinators[request.identifier]
        zmq_coordinator.register_worker_endpoint(
            rank=request.rank, worker_ip=request.worker_ip, data_port=request.data_port
        )

        return {
            "status": "success",
            "message": f"Worker {request.rank} endpoint registered: {request.worker_ip}:{request.data_port}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to report worker endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to report worker endpoint: {str(e)}")


class ExecuteRequest(BaseModel):
    """Request model for execute signal (two-phase protocol)"""

    op_id: str
    identifier: str


@app.post("/execute")
async def execute_operation(request: ExecuteRequest):
    """Broadcast execute signal to trigger pending operation on all workers

    This is Phase 2 of the two-phase scatter protocol:

    Phase 1: Direct client sends data chunks to workers (streaming, memory-efficient)
             Workers store data in pending_ops dict, do NOT execute yet

    Phase 2: Direct client calls this endpoint to broadcast execute signal
             All workers receive signal simultaneously via SUB socket
             Workers lookup op_id in pending_ops and begin execution together
             NCCL timing spread is minimized (<1 second)

    Args:
        request: Contains op_id and identifier

    Returns:
        Success status
    """
    global zmq_coordinators

    try:
        if request.identifier not in zmq_coordinators:
            raise HTTPException(
                status_code=404, detail=f"Worker group '{request.identifier}' not found."
            )

        zmq_coordinator = zmq_coordinators[request.identifier]
        zmq_coordinator.broadcast_execute(request.op_id)

        return {
            "status": "success",
            "op_id": request.op_id,
            "message": f"Execute signal broadcast to worker group '{request.identifier}'",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to broadcast execute signal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to broadcast execute signal: {str(e)}")


class LaunchWorkersRequest(BaseModel):
    """Request model for launching workers"""

    world_size: int
    backend: str = "fsdp"
    identifier: Optional[str] = None
    dispatch_mode: Optional[str] = None
    nproc_per_node: Optional[int] = None
    api_server_url: Optional[str] = None


@app.post("/launch_workers")
async def launch_workers(request: LaunchWorkersRequest):
    """Launch worker processes via torchrun (active launch from API server)

    This endpoint registers a worker group and launches workers as a subprocess.
    The workers will automatically connect to the allocated ZMQ ports.
    Workers must be initialized later via the /initialize endpoint.

    Args:
        request: Launch request with backend and worker configuration

    Returns:
        Launch status and worker group information
    """
    global zmq_coordinators

    try:
        # Detect API server URL if not provided
        api_server_url = request.api_server_url or "http://localhost:8000"

        # Register worker group and allocate ports
        registration = await register_worker_group(
            RegisterWorkerGroupRequest(
                world_size=request.world_size,
                identifier=request.identifier,
                dispatch_mode=request.dispatch_mode,
                api_server_url=api_server_url,
            )
        )

        identifier = registration["identifier"]
        zmq_base_port = registration["zmq_base_port"]

        # Find worker_process.py script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        worker_script = os.path.join(current_dir, "..", "distributed", "worker_process.py")

        if not os.path.exists(worker_script):
            raise HTTPException(status_code=500, detail=f"Worker script not found: {worker_script}")

        # Prepare torchrun command
        nproc_per_node = request.nproc_per_node or request.world_size

        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            "--standalone",
            worker_script,
            f"--zmq-base-port={zmq_base_port}",
            f"--identifier={identifier}",
            f"--api-server-url={api_server_url}",
            f"--backend={request.backend}",
        ]

        # Launch workers in background
        logger.info(f"Launching worker group '{identifier}' with command: {' '.join(cmd)}")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        logger.info(f"Worker group '{identifier}' launched with PID {process.pid}")

        return {
            "status": "success",
            "message": f"Worker group '{identifier}' launched with {request.world_size} workers",
            "identifier": identifier,
            "world_size": request.world_size,
            "zmq_base_port": zmq_base_port,
            "dispatch_mode": registration["dispatch_mode"],
            "process_pid": process.pid,
            "command": " ".join(cmd),
            "note": "Workers are launching in background. Check /worker_info for status.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to launch workers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to launch workers: {str(e)}")


@app.post("/initialize")
async def initialize_workers(request: InitializeWorkersRequest):
    """Initialize worker processes with configuration

    This endpoint assumes workers are already connected (coordinator exists).
    For new worker groups, use /register_worker_group first.

    Args:
        request: Initialization request with config and identifier
    """
    global zmq_coordinators

    try:
        # Extract parameters from request
        config_path = request.config_path
        role = request.role
        config_dict = request.config_dict
        identifier = request.identifier

        # Get coordinator
        zmq_coordinator = get_coordinator(identifier)

        # Use identifier from coordinator if not explicitly provided
        if identifier is None:
            identifier = next(iter(zmq_coordinators.keys()))
        logger.info(f"Initializing workers with identifier: {identifier}")
        # Broadcast initialization command to all workers
        init_data = {
            "config_path": config_path,
            "config_dict": config_dict,
            "role": role,
            "worker_group_id": identifier,
        }

        results = zmq_coordinator.broadcast_command("initialize", **init_data)

        # Check if all workers initialized successfully
        failed_workers = [r for r in results if not r["success"]]
        if failed_workers:
            raise RuntimeError(f"Failed to initialize {len(failed_workers)} workers")

        logger.info(
            f"All {len(results)} workers in group '{identifier}' initialized with role: {role}"
        )

        return {
            "status": "success",
            "message": f"All {len(results)} workers initialized with role: {role}",
            "identifier": identifier,
            "role": role,
            "workers_initialized": len(results),
        }

    except Exception as e:
        logger.error(f"Failed to initialize workers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize workers: {str(e)}")


@app.post("/init_model")
async def init_model(identifier: Optional[str] = None):
    """Initialize models on all workers

    This operation loads models to GPU and requires GPU access.
    It will be queued if GPU is occupied by another worker group.

    Args:
        identifier: Worker group identifier. If None, uses the only/first group.
    """
    try:
        # Get coordinator to validate identifier
        get_coordinator(identifier)

        # Use identifier from coordinator if not explicitly provided
        if identifier is None:
            identifier = next(iter(zmq_coordinators.keys()))

        # Queue the operation (requires GPU - loads model)
        results = await submit_queued_operation(
            identifier=identifier,
            operation_type="command",
            data={"command": "init_model"},
            requires_gpu=True,
        )

        failed_workers = [r for r in results if not r["success"]]
        if failed_workers:
            raise RuntimeError(f"Failed to initialize models on {len(failed_workers)} workers")

        return {
            "status": "success",
            "message": f"Models initialized on all {len(results)} workers",
            "identifier": identifier,
        }

    except Exception as e:
        logger.error(f"Failed to initialize models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize models: {str(e)}")


@app.post("/load_checkpoint")
async def load_checkpoint(request: CheckpointRequest, identifier: Optional[str] = None):
    """Load checkpoint on all workers

    This operation loads checkpoint weights to GPU and requires GPU access.
    It will be queued if GPU is occupied by another worker group.

    Args:
        request: Checkpoint request
        identifier: Worker group identifier. If None, uses the only/first group.
    """
    try:
        # Get coordinator to validate identifier
        get_coordinator(identifier)

        # Use identifier from coordinator if not explicitly provided
        if identifier is None:
            identifier = next(iter(zmq_coordinators.keys()))

        # Queue the operation (requires GPU - loads weights)
        results = await submit_queued_operation(
            identifier=identifier,
            operation_type="command",
            data={
                "command": "load_checkpoint",
                "path": request.path,
                "del_local_after_load": request.del_local_after_load,
                "load_weight_only": request.load_weight_only,
            },
            requires_gpu=True,
        )

        failed_workers = [r for r in results if not r["success"]]
        if failed_workers:
            raise RuntimeError(f"Failed to load checkpoint on {len(failed_workers)} workers")

        return {
            "status": "success",
            "message": f"Checkpoint loaded from {request.path} on all {len(results)} workers",
            "identifier": identifier,
        }

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {str(e)}")


@app.post("/save_checkpoint")
async def save_checkpoint(request: SaveCheckpointRequest, identifier: Optional[str] = None):
    """Save checkpoint (typically on actor workers only)

    This operation saves model from GPU and requires GPU access.
    It will be queued if GPU is occupied by another worker group.

    Args:
        request: Checkpoint save request
        identifier: Worker group identifier. If None, uses the only/first group.
    """
    try:
        # Get coordinator to validate identifier
        get_coordinator(identifier)

        # Use identifier from coordinator if not explicitly provided
        if identifier is None:
            identifier = next(iter(zmq_coordinators.keys()))

        # Queue the operation (requires GPU - saves from model)
        results = await submit_queued_operation(
            identifier=identifier,
            operation_type="command",
            data={
                "command": "save_checkpoint",
                "local_path": request.local_path,
                "hdfs_path": request.hdfs_path,
                "global_step": request.global_step,
                "saved_fully_shared_ckpt": request.saved_fully_shared_ckpt,
                "save_weight_only": request.save_weight_only,
                "remove_previous_ckpt": request.remove_previous_ckpt,
            },
            requires_gpu=True,
        )

        # Note: Some workers might not support save_checkpoint (e.g., non-actor workers)
        # Count successful saves instead of requiring all workers to succeed
        successful_saves = [r for r in results if r["success"]]

        return {
            "status": "success",
            "message": f"Checkpoint saved to {request.local_path} on {len(successful_saves)} workers",
            "workers_saved": len(successful_saves),
            "total_workers": len(results),
            "identifier": identifier,
        }

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save checkpoint: {str(e)}")


@app.post("/convert_checkpoint_to_huggingface")
async def convert_checkpoint_to_huggingface(
    request: ConvertCheckpointRequest, identifier: Optional[str] = None
):
    """Convert checkpoint to HuggingFace format

    This operation may require model to be loaded and requires GPU access.
    It will be queued if GPU is occupied by another worker group.

    Args:
        request: Checkpoint convert request
        identifier: Worker group identifier. If None, uses the only/first group.
    """
    try:
        # Get coordinator to validate identifier
        get_coordinator(identifier)

        # Use identifier from coordinator if not explicitly provided
        if identifier is None:
            identifier = next(iter(zmq_coordinators.keys()))

        # Queue the operation (requires GPU - may need model loaded)
        results = await submit_queued_operation(
            identifier=identifier,
            operation_type="command",
            data={"command": "convert_checkpoint_to_huggingface", "local_path": request.local_path},
            requires_gpu=True,
        )

        successful_converts = [r for r in results if r["success"]]

        return {
            "status": "success",
            "message": f"Checkpoint converted to HuggingFace format at {request.local_path} on {len(successful_converts)} workers",
            "identifier": identifier,
        }

    except Exception as e:
        logger.error(f"Failed to convert checkpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to convert checkpoint: {str(e)}")


@app.post("/barrier")
async def barrier(identifier: Optional[str] = None):
    """Synchronization barrier for all workers

    This operation requires GPU access to ensure consistent state across workers.
    It will be queued if GPU is occupied by another worker group.

    Args:
        identifier: Worker group identifier. If None, uses the only/first group.
    """
    try:
        # Get coordinator to validate identifier
        get_coordinator(identifier)

        # Use identifier from coordinator if not explicitly provided
        if identifier is None:
            identifier = next(iter(zmq_coordinators.keys()))

        # Queue the operation (requires GPU - needs consistent state)
        results = await submit_queued_operation(
            identifier=identifier,
            operation_type="command",
            data={"command": "barrier"},
            requires_gpu=True,
        )

        failed_workers = [r for r in results if not r["success"]]
        if failed_workers:
            raise RuntimeError(f"Barrier failed on {len(failed_workers)} workers")

        return {
            "status": "success",
            "message": f"Barrier completed on all {len(results)} workers",
            "identifier": identifier,
        }

    except Exception as e:
        logger.error(f"Failed to execute barrier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to execute barrier: {str(e)}")


@app.get("/worker_info")
async def worker_info(identifier: Optional[str] = None):
    """Get information about all workers

    Args:
        identifier: Worker group identifier. If None, returns info for all groups.
    """
    global zmq_coordinators

    if not zmq_coordinators:
        return {"zmq_coordinators_initialized": False, "message": "No ZMQ coordinators initialized"}

    try:
        # If identifier is specified, get info for that group
        if identifier is not None:
            zmq_coordinator = get_coordinator(identifier)
            results = zmq_coordinator.broadcast_command("worker_info")

            return {
                "zmq_coordinators_initialized": True,
                "identifier": identifier,
                "total_workers": len(results),
                "workers": results,
            }

        # Otherwise, get info for all groups
        all_workers_info = {}
        for group_id, coordinator in zmq_coordinators.items():
            try:
                results = coordinator.broadcast_command("worker_info")
                all_workers_info[group_id] = {"total_workers": len(results), "workers": results}
            except Exception as e:
                all_workers_info[group_id] = {
                    "total_workers": coordinator.world_size,
                    "error": str(e),
                }

        return {"zmq_coordinators_initialized": True, "worker_groups": all_workers_info}

    except Exception as e:
        logger.error(f"Failed to get worker info: {e}", exc_info=True)
        return {"zmq_coordinators_initialized": True, "error": str(e)}


@app.get("/dispatch_mode")
async def get_dispatch_mode(identifier: Optional[str] = None):
    """Get current dispatch mode

    Args:
        identifier: Worker group identifier. If None, returns info for all groups.
    """
    global zmq_coordinators

    if identifier is not None:
        zmq_coordinator = get_coordinator(identifier)
        return {
            "identifier": identifier,
            "dispatch_mode": zmq_coordinator.dispatch_mode,
            "available_modes": ["broadcast", "scatter"],
            "description": {
                "broadcast": "Send same data to all workers (original behavior)",
                "scatter": "Split data among workers (DP-style)",
            },
        }

    # Return info for all groups
    modes_info = {}
    for group_id, coordinator in zmq_coordinators.items():
        modes_info[group_id] = coordinator.dispatch_mode

    return {
        "worker_groups": modes_info,
        "available_modes": ["broadcast", "scatter"],
        "description": {
            "broadcast": "Send same data to all workers (original behavior)",
            "scatter": "Split data among workers (DP-style)",
        },
    }


@app.post("/set_dispatch_mode")
async def set_dispatch_mode(mode: str, identifier: Optional[str] = None):
    """Set dispatch mode (requires restart for socket changes)

    Args:
        mode: Dispatch mode to set
        identifier: Worker group identifier. If None, uses the only/first group.
    """
    if mode not in ["broadcast", "scatter"]:
        raise HTTPException(
            status_code=400, detail="Invalid dispatch mode. Must be 'broadcast' or 'scatter'"
        )

    zmq_coordinator = get_coordinator(identifier)

    if mode != zmq_coordinator.dispatch_mode:
        return {
            "status": "warning",
            "message": f"Dispatch mode change from '{zmq_coordinator.dispatch_mode}' to '{mode}' requires API server restart",
            "identifier": identifier,
            "current_mode": zmq_coordinator.dispatch_mode,
            "requested_mode": mode,
            "restart_required": True,
        }
    else:
        return {
            "status": "success",
            "message": f"Dispatch mode is already set to '{mode}'",
            "identifier": identifier,
            "current_mode": zmq_coordinator.dispatch_mode,
        }


@app.get("/gpu_status")
async def get_gpu_status():
    """Get queue status of all worker groups

    Returns information about operation queues for each worker group.

    Returns:
        Queue status information including:
        - total_queue_length: Total number of operations waiting across all queues
        - worker_groups: Status of all registered worker groups
    """
    global group_operation_queues, zmq_coordinators

    # Get status of all worker groups
    worker_group_status = {}
    total_queue_length = 0

    for identifier, coordinator in zmq_coordinators.items():
        # Get queue length for this group
        group_queue_length = 0
        if identifier in group_operation_queues:
            group_queue_length = group_operation_queues[identifier].qsize()
            total_queue_length += group_queue_length

        worker_group_status[identifier] = {
            "world_size": coordinator.world_size,
            "dispatch_mode": coordinator.dispatch_mode,
            "queue_length": group_queue_length,
        }

    return {
        "total_queue_length": total_queue_length,
        "worker_groups": worker_group_status,
        "total_worker_groups": len(zmq_coordinators),
    }


@app.post("/deregister")
async def deregister_worker_group(identifier: str):
    """Deregister a worker group that is shutting down on its own

    This endpoint is called by workers when they exit (e.g., on KeyboardInterrupt).
    Unlike /destroy, this does NOT send commands to workers - it just cleans up
    the API server's tracking of the worker group.

    Args:
        identifier: Worker group identifier to deregister

    Returns:
        Status message indicating successful deregistration

    Raises:
        HTTPException: If worker group not found or cleanup fails
    """
    global zmq_coordinators, heartbeat_timestamps, heartbeat_lock
    global group_operation_queues, group_processor_threads, group_stop_events

    try:
        # Check if worker group exists
        if identifier not in zmq_coordinators:
            logger.warning(f"Deregister request for unknown worker group '{identifier}'")
            return {
                "status": "success",
                "message": f"Worker group '{identifier}' not found (may already be deregistered)",
                "identifier": identifier,
            }

        logger.info(f"Deregistering worker group '{identifier}'...")

        zmq_coordinator = zmq_coordinators[identifier]

        # Stop processor thread
        _stop_group_processor(identifier)

        # Remove from operation queues
        if identifier in group_operation_queues:
            del group_operation_queues[identifier]
            logger.info(f"Removed operation queue for worker group '{identifier}'")

        # Remove from heartbeat tracking
        with heartbeat_lock:
            if identifier in heartbeat_timestamps:
                del heartbeat_timestamps[identifier]
                logger.info(f"Removed heartbeat tracking for worker group '{identifier}'")

        # Close and remove coordinator
        zmq_coordinator.close()
        del zmq_coordinators[identifier]
        logger.info(f"Closed and removed coordinator for worker group '{identifier}'")

        return {
            "status": "success",
            "message": f"Worker group '{identifier}' deregistered successfully",
            "identifier": identifier,
        }

    except Exception as e:
        logger.error(f"Failed to deregister worker group '{identifier}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to deregister worker group: {str(e)}")


@app.post("/destroy")
async def destroy_worker_group(identifier: Optional[str] = None):
    """Destroy a worker group and clean up all resources

    This endpoint sends a destroy command to all workers, waits for them to shut down,
    then cleans up the coordinator and removes it from the registry.

    Args:
        identifier: Worker group identifier. If None, uses the only/first group.

    Returns:
        Status message indicating successful destruction

    Raises:
        HTTPException: If worker group not found or cleanup fails
    """
    global zmq_coordinators, heartbeat_timestamps, heartbeat_lock
    global group_operation_queues, group_processor_threads, group_stop_events

    try:
        # Get coordinator to validate identifier
        zmq_coordinator = get_coordinator(identifier)

        # Use identifier from coordinator if not explicitly provided
        if identifier is None:
            identifier = next(iter(zmq_coordinators.keys()))

        logger.info(f"Destroying worker group '{identifier}'...")

        # Send destroy command to all workers
        try:
            results = zmq_coordinator.broadcast_command("destroy")
            logger.info(f"Destroy command sent to {len(results)} workers in group '{identifier}'")
        except Exception as e:
            logger.warning(f"Failed to send destroy command to workers: {e}")
            # Continue with cleanup even if workers don't respond

        # Wait a moment for workers to shut down gracefully
        time.sleep(1)

        # Stop processor thread
        _stop_group_processor(identifier)

        # Remove from operation queues
        if identifier in group_operation_queues:
            del group_operation_queues[identifier]
            logger.info(f"Removed operation queue for worker group '{identifier}'")

        # Remove from heartbeat tracking
        with heartbeat_lock:
            if identifier in heartbeat_timestamps:
                del heartbeat_timestamps[identifier]
                logger.info(f"Removed heartbeat tracking for worker group '{identifier}'")

        # Close and remove coordinator
        zmq_coordinator.close()
        del zmq_coordinators[identifier]
        logger.info(f"Closed and removed coordinator for worker group '{identifier}'")

        return {
            "status": "success",
            "message": f"Worker group '{identifier}' destroyed successfully",
            "identifier": identifier,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to destroy worker group: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to destroy worker group: {str(e)}")


@app.get("/getattr/{attr_name}")
async def get_coordinator_attr(attr_name: str, identifier: Optional[str] = None):
    """Get attribute from ZMQ coordinator

    This endpoint allows clients to access coordinator attributes dynamically.
    Used by HTTPTrainServiceClient.__getattr__ to proxy attribute access.

    Args:
        attr_name: Name of the attribute to retrieve from coordinator
        identifier: Worker group identifier. If None, uses the only/first group.

    Returns:
        Dict with 'value' key containing the attribute value

    Raises:
        HTTPException: If coordinator not initialized, attribute doesn't exist,
                      or attribute is not serializable
    """
    try:
        zmq_coordinator = get_coordinator(identifier)

        if not hasattr(zmq_coordinator, attr_name):
            raise HTTPException(
                status_code=404, detail=f"ZMQ coordinator has no attribute '{attr_name}'"
            )

        value = getattr(zmq_coordinator, attr_name)

        # Handle non-serializable types
        if callable(value):
            return {
                "error": f"Attribute '{attr_name}' is callable and cannot be serialized",
                "type": "callable",
                "callable_info": str(value),
            }

        # Handle complex objects that might not be JSON serializable
        try:
            # Test if the value can be JSON serialized without any conversion
            json.dumps(value)
            return {"value": value}
        except (TypeError, ValueError) as e:
            # For non-serializable types, return type info and string representation
            logger.warning(f"Attribute '{attr_name}' not JSON serializable: {e}")
            return {
                "error": f"Attribute '{attr_name}' is not JSON serializable",
                "type": type(value).__name__,
                "string_representation": str(value),
                "serialization_error": str(e),
            }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to get coordinator attribute '{attr_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get attribute '{attr_name}': {str(e)}"
        )


def main():
    """Main entry point for the standalone API server

    The API server starts without any worker groups. Worker groups are created dynamically via:
    - /register_worker_group - Workers register themselves (passive launch)
    - /launch_workers - API server launches workers (active launch)
    """
    parser = argparse.ArgumentParser(description="Standalone API Server for ActorWorker")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    global zmq_coordinators, heartbeat_monitor_thread, heartbeat_timestamps, heartbeat_lock
    global group_stop_events, group_processor_threads

    logger.info("Starting API server - worker groups will be created dynamically via API endpoints")

    # Start heartbeat monitoring thread
    heartbeat_monitor_thread = threading.Thread(
        target=monitor_heartbeats, daemon=True, name="heartbeat-monitor"
    )
    heartbeat_monitor_thread.start()
    logger.info("Started heartbeat monitoring thread")

    logger.info(f"Starting standalone API server on {args.host}:{args.port}")

    try:
        uvicorn.run(
            app, host=args.host, port=args.port, log_level="info", log_config=LOGGING_CONFIG
        )
    except KeyboardInterrupt:
        logger.info("Shutting down API server...")
    finally:
        # Stop heartbeat monitor
        logger.info("Stopping heartbeat monitoring thread...")
        heartbeat_monitor_stop_event.set()
        if heartbeat_monitor_thread:
            heartbeat_monitor_thread.join(timeout=5)

        # Stop all processor threads
        logger.info("Stopping processor threads...")
        for identifier in list(group_processor_threads.keys()):
            _stop_group_processor(identifier)

        # Close all coordinators
        for coordinator in zmq_coordinators.values():
            coordinator.close()


if __name__ == "__main__":
    main()
