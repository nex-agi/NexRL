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
Direct ZMQ Train Service Client

This module provides a client that sends data directly to workers via ZMQ,
bypassing the API server for data transfer. The API server is only used for
coordination (commands, health checks, worker registration).

Benefits:
- Eliminates double data transfer (driver→API→workers becomes driver→workers)
- Reduces API server memory pressure (no large data buffering)
- Faster data transfer (single hop instead of two)

The client implements the same TrainServiceClient interface, so it's a
drop-in replacement for HTTPTrainServiceClient.

Example:
    >>> # Use just like HTTPTrainServiceClient
    >>> client = DirectZMQTrainServiceClient("http://localhost:8000", identifier="nexrl-group-1")
    >>> client.initialize_worker(config_path="config.yaml", role="actor", world_size=32)
    >>> client.update_actor(batch)  # Data goes directly to workers via ZMQ
"""

import logging
import pickle
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import requests
import zmq

from ...train_service_client import TrainServiceClient
from ...utils.url_utils import ensure_url_scheme
from ..utils.core_utils import (
    DataProtoResponse,
    data_to_numpy,
    data_to_tensor,
    numpy_to_data,
    prepare_data_proto_request,
    process_data_proto_response,
    restore_payload,
    split_for_requests,
    tensor_to_data,
)
from ..utils.protocol import DataProto

logger = logging.getLogger(__name__)


class DirectZMQTrainServiceClient(TrainServiceClient):
    """
    Hybrid client that uses:
    - HTTP to API server for coordination (commands, health, registration)
    - Direct ZMQ to workers for data transfer (update_actor, compute_log_prob, etc.)

    This eliminates the double data transfer and API server memory bottleneck.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        identifier: str | None = None,
        zmq_recv_timeout_ms: int = 600000,  # 10 minutes per recv
        zmq_total_timeout_ms: int = 3600000,  # 60 minutes total
    ):
        """Initialize the client

        Args:
            base_url: Base URL of the API server (for coordination only)
            identifier: Worker group identifier
            zmq_recv_timeout_ms: Timeout for each ZMQ recv operation
            zmq_total_timeout_ms: Total timeout for collecting all results
        """
        super().__init__(base_url, identifier)
        normalized_url = ensure_url_scheme(base_url)
        if not normalized_url:
            raise ValueError("base_url is required for HTTPTrainServiceClient")
        super().__init__(normalized_url, identifier)
        self.base_url = normalized_url.rstrip("/")
        self.session = requests.Session()
        self.request_timeout = 200

        # ZMQ configuration
        self.zmq_recv_timeout_ms = zmq_recv_timeout_ms
        self.zmq_total_timeout_ms = zmq_total_timeout_ms

        # ZMQ connections (initialized lazily after worker group is registered)
        self._zmq_context: zmq.Context | None = None
        self._worker_pushers: list[zmq.Socket] | None = None
        self._result_collector: zmq.Socket | None = None
        self._world_size: int | None = None
        self._api_server_host: str | None = None

    @property
    def identifier(self) -> str | None:
        """Get the immutable identifier for this client"""
        return self._identifier

    def _ensure_zmq_connected(self) -> None:
        """Ensure ZMQ connections to workers are established

        In the new architecture, workers bind their own PULL sockets and report
        their endpoints to the API server. This method queries the API server
        for worker endpoints and establishes direct PUSH connections to each worker.
        """
        if self._worker_pushers is not None:
            return

        # Get worker group info from API server (includes worker endpoints)
        info = self._get_worker_group_info()

        self._world_size = info["world_size"]
        base_port = info["zmq_base_port"]
        dispatch_mode = info.get("dispatch_mode", "scatter")
        worker_endpoints = info.get("worker_endpoints", {})

        import os

        self._api_server_host = os.environ.get("API_SERVER_URL")

        if dispatch_mode != "scatter":
            raise ValueError(
                f"DirectZMQTrainServiceClient only supports 'scatter' dispatch mode, "
                f"but worker group uses '{dispatch_mode}'"
            )

        # Check if all workers have reported their endpoints
        if len(worker_endpoints) < self._world_size:
            raise RuntimeError(
                f"Only {len(worker_endpoints)}/{self._world_size} workers have reported endpoints. "
                f"Workers may still be starting up. Please wait and retry."
            )

        # Create ZMQ context
        self._zmq_context = zmq.Context()

        # Create PUSH sockets to each worker's data port
        # Workers bind their own ports and report them to API server
        self._worker_pushers = []
        for rank in range(self._world_size):
            # Worker endpoints are keyed by rank (as string in JSON)
            endpoint = worker_endpoints.get(str(rank)) or worker_endpoints.get(rank)
            if not endpoint:
                raise RuntimeError(f"No endpoint found for worker rank {rank}")

            worker_ip = endpoint["ip"]
            worker_port = endpoint["port"]

            socket = self._zmq_context.socket(zmq.PUSH)
            # Set socket options to avoid dropping messages
            socket.setsockopt(zmq.LINGER, 0)  # Don't block on close
            socket.setsockopt(zmq.SNDHWM, 0)  # Unlimited send buffer (no dropping)
            socket.connect(f"tcp://{worker_ip}:{worker_port}")
            self._worker_pushers.append(socket)

        # Create PULL socket to collect results from API server's result forwarder
        # Results flow: Workers PUSH → API Server PULL (5556) → API Server PUSH (5557) → Client PULL
        # IMPORTANT: Connect to base_port + 2 (the forwarder PUSH), NOT base_port + 1 (the collector PULL)
        self._result_collector = self._zmq_context.socket(zmq.PULL)
        forwarder_port = base_port + 2  # API server's result forwarder PUSH socket
        self._result_collector.connect(f"tcp://{self._api_server_host}:{forwarder_port}")

        time.sleep(1.0)  # 1 second initial wait for TCP connection setup
        logger.info(f"DirectZMQ connections successfully established to {self._world_size} workers")

        # Actually test connectivity by sending ping messages to all workers
        # Workers will respond with pong, proving the PUSH-PULL path works
        self._verify_worker_connectivity()

    def _verify_worker_connectivity(self, max_retries: int = 10, retry_delay: float = 1.0) -> None:
        """Verify that all workers can receive data via PUSH-PULL connection

        Sends a ping message to each worker and waits for pong responses.
        This ensures the ZMQ connections are actually established before
        sending real data.

        NOTE: This is critical for cross-node ZMQ connections. The 500ms delay
        is often insufficient for TCP handshakes across network nodes. Without
        this verification, messages can be silently dropped due to ZMQ's
        "slow joiner" problem.

        Args:
            max_retries: Maximum number of ping attempts per worker
            retry_delay: Delay between retries in seconds
        """
        assert self._worker_pushers is not None
        assert self._world_size is not None
        assert self._result_collector is not None

        ping_op_id = f"ping_{time.time_ns()}"
        # Send ping to all workers via PUSH sockets
        # Workers respond immediately via their PUSH -> API server PULL -> client PULL
        for rank, pusher in enumerate(self._worker_pushers):
            ping_msg = {
                "op_id": ping_op_id,
                "phase": "ping",
                "rank": rank,
                "timestamp": time.time(),
            }
            pusher.send_pyobj(ping_msg)
            logger.debug(f"[PING] Sent ping to worker {rank}")

        # Collect pong responses from all workers
        received_pongs: set[int] = set()
        start_time = time.time()
        timeout_seconds = max_retries * retry_delay + 30  # Total timeout

        self._result_collector.setsockopt(zmq.RCVTIMEO, int(retry_delay * 1000))

        while len(received_pongs) < self._world_size:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                missing = set(range(self._world_size)) - received_pongs
                raise RuntimeError(
                    f"[PING] Timeout waiting for worker connectivity verification. "
                    f"Received pong from {len(received_pongs)}/{self._world_size} workers. "
                    f"Missing workers: {sorted(missing)}. "
                    f"This usually means ZMQ PUSH-PULL connection failed to establish. "
                    f"Check network connectivity between client and workers, "
                    f"and ensure ports 6000-600{self._world_size-1} are reachable on worker nodes."
                )

            try:
                result = self._result_collector.recv_pyobj()
                result_op_id = result.get("op_id")
                rank = result.get("rank")

                if result_op_id == ping_op_id and result.get("pong"):
                    received_pongs.add(rank)
                else:
                    # Not a pong for our ping, skip (could be stale result from previous operation)
                    logger.warning(
                        f"[PING] Ignoring non-pong result: op_id={result_op_id}, pong={result.get('pong')}"
                    )
            except zmq.Again:
                # Timeout, log progress and continue waiting
                logger.info(
                    f"[PING] Still waiting for pong... ({len(received_pongs)}/{self._world_size} received)"
                )
                continue

        elapsed_total = time.time() - start_time
        logger.info(f"✓ All {self._world_size} workers verified reachable in {elapsed_total:.2f}s")

        # Reset recv timeout for normal operations
        self._result_collector.setsockopt(zmq.RCVTIMEO, self.zmq_recv_timeout_ms)

    def _get_worker_group_info(self) -> dict[str, Any]:
        """Get worker group info from API server"""
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.get(f"{self.base_url}/get_worker_group_info", params=params)
        response.raise_for_status()
        return response.json()

    def _data_proto_to_dict(self, data_proto: DataProto) -> dict[str, Any]:
        """Convert DataProto to dictionary format

        Args:
            data_proto: DataProto object to convert

        Returns:
            Dictionary with batch, non_tensor_batch, and meta_info
        """
        # Use DataProtoResponse for proper serialization
        response = DataProtoResponse.from_data_proto(data_proto)

        # Convert to dict format
        result: dict[str, Any] = {}

        if response.batch:
            result["batch"] = {
                k: v.to_tensor() if hasattr(v, "to_tensor") else v
                for k, v in response.batch.items()
            }

        if response.non_tensor_batch:
            result["non_tensor_batch"] = {
                k: v.to_numpy() if hasattr(v, "to_numpy") else v
                for k, v in response.non_tensor_batch.items()
            }

        if response.meta_info:
            result["meta_info"] = response.meta_info

        return result

    def _scatter_to_workers(self, operation: str, data: DataProto) -> DataProto:
        """NCCL-based scatter with ACK-ordered two-phase protocol

        Flow:
        1. Send full batch to rank 0 via direct PUSH socket
        2. Wait for ACK from rank 0 (confirms data received)
        3. Broadcast execute signal to all workers via PUB/SUB
        4. Rank 0 scatters data via NCCL (respects DP/SP topology)
        5. All workers execute operation on their data chunk
        6. All workers gather results to rank 0 via NCCL
        7. Rank 0 returns gathered result via PULL socket

        ACK prevents race condition (execute arriving before data).
        NCCL provides fast inter-worker communication and topology awareness.
        """
        self._ensure_zmq_connected()

        # Type narrowing assertions after ensuring connections are established
        assert self._world_size is not None
        assert self._worker_pushers is not None

        # Generate unique operation ID for two-phase protocol
        op_id = f"{operation}_{time.time_ns()}"

        # PHASE 1: Send full batch to rank 0 only
        # Rank 0 will handle NCCL scatter based on device_mesh topology
        send_start = time.time()
        message = {
            "op_id": op_id,
            "phase": "data",
            "operation": operation,
            "data": data,  # Full batch, not chunked
            "rank": 0,
            "timestamp": time.time(),
            "use_nccl_scatter": True,  # Signal to use NCCL scatter/gather
        }

        # Estimate message size using DataProto's built-in method (no memory overhead)
        message_size_mb = -1.0
        if hasattr(data, "estimate_size_mb"):
            message_size_mb = data.estimate_size_mb()

        if message_size_mb > 0:
            logger.info(
                f"Sending full batch to rank 0 for op_id={op_id} (estimated size: ~{message_size_mb:.1f} MB)"
            )
        else:
            logger.info(f"Sending full batch to rank 0 for op_id={op_id}")
        self._worker_pushers[0].send_pyobj(message)
        send_duration = time.time() - send_start
        if message_size_mb > 0:
            logger.info(
                f"Phase 1: sent full batch to rank 0 in {send_duration:.2f}s (~{message_size_mb:.1f} MB)"
            )
        else:
            logger.info(f"Phase 1: sent full batch to rank 0 in {send_duration:.2f}s")

        # Wait for ACK from rank 0 confirming data receipt
        # Timeout scales with message size: base 60s + 30s per 100MB
        ack_start = time.time()
        if message_size_mb > 0:
            ack_timeout = 600.0 + (message_size_mb / 100.0) * 60.0  # Scale with size
            logger.info(
                f"Waiting for data ACK from rank 0 (timeout: {ack_timeout:.0f}s, message size: ~{message_size_mb:.1f} MB)"
            )
        else:
            ack_timeout = 600.0  # Default 10 minutes if size unknown
            logger.info(f"Waiting for data ACK from rank 0 (timeout: {ack_timeout:.0f}s)")

        assert self._result_collector is not None
        original_timeout = self._result_collector.getsockopt(zmq.RCVTIMEO)
        self._result_collector.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second polling

        last_log_time = ack_start
        try:
            while (time.time() - ack_start) < ack_timeout:
                # Log progress every 10 seconds to show we're still waiting
                elapsed = time.time() - ack_start
                if elapsed - (last_log_time - ack_start) >= 10.0:
                    if message_size_mb > 0:
                        logger.info(
                            f"Still waiting for ACK from rank 0... ({elapsed:.1f}s elapsed, ~{message_size_mb:.1f} MB message)"
                        )
                    else:
                        logger.info(
                            f"Still waiting for ACK from rank 0... ({elapsed:.1f}s elapsed)"
                        )
                    last_log_time = time.time()

                try:
                    msg = self._result_collector.recv_pyobj()
                    # Check if this is the ACK we're waiting for
                    if (
                        msg.get("op_id") == op_id
                        and msg.get("phase") == "data_ack"
                        and msg.get("rank") == 0
                    ):
                        if message_size_mb > 0:
                            logger.info(
                                f"✓ Received data ACK from rank 0 in {time.time() - ack_start:.3f}s (~{message_size_mb:.1f} MB)"
                            )
                        else:
                            logger.info(
                                f"✓ Received data ACK from rank 0 in {time.time() - ack_start:.3f}s"
                            )
                        break
                    # Discard non-ACK messages (might be stale results)
                    logger.debug(
                        f"Discarding non-ACK message: op_id={msg.get('op_id')}, phase={msg.get('phase')}"
                    )
                except zmq.Again:
                    continue  # Timeout, keep polling
            else:
                # Loop completed without break = timeout
                if message_size_mb > 0:
                    error_msg = (
                        f"Timeout waiting for data ACK from rank 0 for op_id={op_id} after {ack_timeout:.0f}s. "
                        f"Message size: ~{message_size_mb:.1f} MB. This may indicate: "
                        f"(1) Large message taking too long to transfer, "
                        f"(2) Rank 0 crashed or hung, "
                        f"(3) Network connectivity issues. "
                        f"Check rank 0 worker logs for errors."
                    )
                else:
                    error_msg = (
                        f"Timeout waiting for data ACK from rank 0 for op_id={op_id} after {ack_timeout:.0f}s. "
                        f"This may indicate: "
                        f"(1) Rank 0 crashed or hung, "
                        f"(2) Network connectivity issues. "
                        f"Check rank 0 worker logs for errors."
                    )
                raise RuntimeError(error_msg)
        finally:
            self._result_collector.setsockopt(zmq.RCVTIMEO, original_timeout)

        # PHASE 2: Broadcast execute signal via API server
        # All workers receive this simultaneously and participate in NCCL scatter/gather
        execute_start = time.time()
        logger.debug(f"Sending execute signal for op_id={op_id} to {self.base_url}/execute")
        try:
            response = self.session.post(
                f"{self.base_url}/execute",
                json={"op_id": op_id, "identifier": self.identifier},
                timeout=30,
            )
            response.raise_for_status()
            logger.debug(f"Execute signal sent successfully, status={response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send execute signal for op_id={op_id}: {e}")
            raise RuntimeError(f"Execute signal failed: {e}")

        execute_duration = time.time() - execute_start
        logger.debug(f"Execute signal sent in {execute_duration:.3f}s")

        # Collect results from rank 0 only (after NCCL gather)
        logger.debug(f"Collecting gathered results from rank 0 for op_id={op_id}")
        results = self._collect_results(op_id, collect_from_rank_0_only=True)
        logger.debug(f"Collected gathered results successfully")

        # Return the gathered result from rank 0
        return results[0]["result"]

    def _collect_results(
        self, op_id: str, collect_from_rank_0_only: bool = False
    ) -> list[dict[str, Any]]:
        """Collect results from workers, filtering by op_id

        Results are filtered by op_id to support concurrent operations.
        Results that don't match the op_id are logged and skipped.

        Args:
            op_id: Operation ID to filter results by
            collect_from_rank_0_only: If True, only collect from rank 0 (after NCCL gather).
                                      If False, collect from all workers (legacy mode).

        Returns:
            List of result dicts sorted by rank
        """
        # Type narrowing assertions
        assert self._world_size is not None
        assert self._result_collector is not None

        start_time = time.time() * 1000
        results_by_rank: dict[int, Any] = {}
        last_log_time = start_time

        # Determine how many results to expect
        expected_results = 1 if collect_from_rank_0_only else self._world_size

        self._result_collector.setsockopt(zmq.RCVTIMEO, self.zmq_recv_timeout_ms)

        while len(results_by_rank) < expected_results:
            elapsed = time.time() * 1000 - start_time

            # Log progress every 5 seconds
            if elapsed - last_log_time + start_time > 5000:
                last_log_time = time.time() * 1000

            if elapsed > self.zmq_total_timeout_ms:
                if collect_from_rank_0_only:
                    raise RuntimeError(
                        f"Timeout collecting result from rank 0 for op_id={op_id}. "
                        f"No result received."
                    )
                else:
                    missing = set(range(self._world_size)) - set(results_by_rank.keys())
                    raise RuntimeError(
                        f"Timeout collecting results for op_id={op_id}. "
                        f"Got {len(results_by_rank)}/{self._world_size}. Missing: {sorted(missing)}"
                    )

            try:
                # Receive using recv_pyobj to match worker's send_pyobj
                result = self._result_collector.recv_pyobj()

                # Filter by op_id (for two-phase protocol)
                result_op_id = result.get("op_id")
                if result_op_id and result_op_id != op_id:
                    continue

                rank = result.get("rank")

                # If collecting from rank 0 only, ignore results from other ranks
                if collect_from_rank_0_only and rank != 0:
                    logger.debug(f"Ignoring result from rank {rank} (expecting rank 0 only)")
                    continue

                if rank is not None and rank not in results_by_rank:
                    # Check for errors
                    if not result.get("success", True):
                        error_msg = result.get("error", "Unknown error")
                        traceback_str = result.get("traceback", "")
                        logger.error(f"Worker {rank} failed: {error_msg}")
                        if traceback_str:
                            logger.error(f"Traceback:\n{traceback_str}")
                        raise RuntimeError(f"Worker {rank} failed: {error_msg}")

                    results_by_rank[rank] = result
                    logger.debug(
                        f"Received result from rank {rank} "
                        f"({len(results_by_rank)}/{expected_results})"
                    )
            except zmq.Again:
                logger.debug(f"Timeout waiting for result, retrying...")
                continue

        # Sort by rank
        if collect_from_rank_0_only:
            return [results_by_rank[0]]
        else:
            return [results_by_rank[i] for i in range(self._world_size)]

    def close(self) -> None:
        """Close ZMQ connections"""
        if self._worker_pushers:
            for socket in self._worker_pushers:
                socket.close()
            self._worker_pushers = None

        if self._result_collector:
            self._result_collector.close()
            self._result_collector = None

        if self._zmq_context:
            self._zmq_context.term()
            self._zmq_context = None

    def __del__(self):
        self.close()

    # =========================================================================
    # Data operations - use direct ZMQ
    # =========================================================================

    def update_actor(self, data: dict[str, Any]) -> dict[str, Any]:
        """Update actor policy - sends data directly to workers via ZMQ"""
        # For ZMQ transport, create DataProto directly from input data
        # (no need for HTTP serialization like _prepare_data_proto_request does)
        serializable_metadata, rejected_metadata = split_for_requests(data.get("meta_info", {}))

        data_proto = DataProto.from_dict(
            tensors=data.get("batch", {}),
            non_tensors=data.get("non_tensor_batch"),
            meta_info=serializable_metadata,
        )

        # Send directly to workers
        result_proto = self._scatter_to_workers("update_actor", data_proto)

        # Convert back to dict
        ret = self._data_proto_to_dict(result_proto)
        ret["meta_info"] = restore_payload(ret.get("meta_info", {}), rejected_metadata)
        return ret

    def update_actor_with_distillation(self, data: dict[str, Any]) -> dict[str, Any]:
        """Update actor using on-policy distillation - sends data directly to workers via ZMQ

        Args:
            data: Data to send to workers (should include teacher_logits)

        Returns:
            dict containing training metrics
        """
        # For ZMQ transport, create DataProto directly from input data
        serializable_metadata, rejected_metadata = split_for_requests(data.get("meta_info", {}))

        data_proto = DataProto.from_dict(
            tensors=data.get("batch", {}),
            non_tensors=data.get("non_tensor_batch"),
            meta_info=serializable_metadata,
        )

        # Send directly to workers
        result_proto = self._scatter_to_workers("update_actor_with_distillation", data_proto)

        # Convert back to dict
        ret = self._data_proto_to_dict(result_proto)
        ret["meta_info"] = restore_payload(ret.get("meta_info", {}), rejected_metadata)
        return ret

    def compute_log_prob(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute log probabilities - sends data directly to workers via ZMQ"""
        # For ZMQ transport, create DataProto directly from input data
        data_proto = DataProto.from_dict(
            tensors=data.get("batch", {}),
            non_tensors=data.get("non_tensor_batch"),
            meta_info=data.get("meta_info", {}),
        )

        result_proto = self._scatter_to_workers("compute_log_prob", data_proto)
        return self._data_proto_to_dict(result_proto)

    def compute_ref_log_prob(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute reference log probabilities - sends data directly to workers via ZMQ"""
        # For ZMQ transport, create DataProto directly from input data
        data_proto = DataProto.from_dict(
            tensors=data.get("batch", {}),
            non_tensors=data.get("non_tensor_batch"),
            meta_info=data.get("meta_info", {}),
        )

        result_proto = self._scatter_to_workers("compute_ref_log_prob", data_proto)
        return self._data_proto_to_dict(result_proto)

    # =========================================================================
    # Coordination operations - use HTTP to API server
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Check service health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def initialize_worker(
        self,
        config_path: str | None = None,
        config_dict: dict[str, Any] | None = None,
        role: str = "actor",
        world_size: int | None = None,
        zmq_base_port: int | None = None,
        dispatch_mode: str | None = None,
    ) -> dict[str, Any]:
        """Initialize the worker - uses HTTP to API server"""
        request_data: dict[str, Any] = {"role": role}

        if config_path:
            request_data["config_path"] = config_path
        elif config_dict:
            request_data["config_dict"] = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        if self.identifier:
            request_data["identifier"] = self.identifier
        if world_size:
            request_data["world_size"] = world_size
        if zmq_base_port:
            request_data["zmq_base_port"] = zmq_base_port

        # Force scatter mode for direct ZMQ
        request_data["dispatch_mode"] = "scatter"

        response = self.session.post(f"{self.base_url}/initialize", json=request_data)
        response.raise_for_status()
        return response.json()

    def init_model(self) -> dict[str, Any]:
        """Initialize the model - uses HTTP to API server"""
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.post(f"{self.base_url}/init_model", params=params)
        response.raise_for_status()
        return response.json()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        saved_fully_shared_ckpt: bool = True,
        save_weight_only: bool = False,
        remove_previous_ckpt: bool = True,
    ) -> dict[str, Any]:
        """Save checkpoint - uses HTTP to API server"""
        request_data = {
            "local_path": local_path,
            "hdfs_path": hdfs_path,
            "global_step": global_step,
            "saved_fully_shared_ckpt": saved_fully_shared_ckpt,
            "save_weight_only": save_weight_only,
            "remove_previous_ckpt": remove_previous_ckpt,
        }
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.post(
            f"{self.base_url}/save_checkpoint", json=request_data, params=params
        )
        response.raise_for_status()
        return response.json()

    def load_checkpoint(
        self, path: str, del_local_after_load: bool = True, load_weight_only: bool = False
    ) -> dict[str, Any]:
        """Load checkpoint - uses HTTP to API server"""
        request_data = {
            "path": path,
            "del_local_after_load": del_local_after_load,
            "load_weight_only": load_weight_only,
        }
        params = {"identifier": self.identifier} if self.identifier else {}
        response = self.session.post(
            f"{self.base_url}/load_checkpoint", json=request_data, params=params
        )
        response.raise_for_status()
        return response.json()

    @contextmanager
    def actor_context(self):
        """Context manager for actor model GPU loading"""
        yield

    # =========================================================================
    # Utility methods
    # =========================================================================

    def _tensor_to_data(self, tensor) -> dict[str, Any]:
        return tensor_to_data(tensor)

    def _data_to_tensor(self, data: dict[str, Any]):
        return data_to_tensor(data)

    def _numpy_to_data(self, array: np.ndarray) -> dict[str, Any]:
        return numpy_to_data(array)

    def _data_to_numpy(self, data: dict[str, Any]) -> np.ndarray:
        return data_to_numpy(data)

    def _prepare_data_proto_request(self, data: dict[str, Any]) -> dict[str, Any]:
        return prepare_data_proto_request(data)

    def _process_data_proto_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        return process_data_proto_response(response_data)
