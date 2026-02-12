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

from ...utils.url_utils import ensure_url_scheme
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
    """ZMQ-based coordinator for worker processes

    Direct communication architecture with ACK-based two-phase protocol:
    - Workers BIND PULL sockets for data reception (clients connect to them)
    - Workers report their IP:port to API server for client discovery
    - Two-phase protocol with ACK prevents race conditions:
      1. Client → Rank 0: send data (PUSH socket)
      2. Rank 0 → Client: send ACK (confirms receipt)
      3. Client → All workers: broadcast execute (PUB/SUB via API)
      4. Workers: participate in NCCL scatter/gather

    This eliminates API server data bottleneck while ensuring correct ordering.
    """

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

        # Ensure api_server_url has proper http:// scheme
        normalized_url = ensure_url_scheme(api_server_url or "http://localhost:8000")
        if not normalized_url:
            normalized_url = "http://localhost:8000"

        # Extract hostname from API server URL
        parsed_url = urlparse(normalized_url)
        hostname = parsed_url.hostname
        # Ensure master_addr is a string
        self.master_addr: str = hostname if hostname else "localhost"

        # Heartbeat control
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_stop_event = threading.Event()

        # Two-phase protocol: pending operations waiting for execute signal
        # Format: {op_id: {"operation": str, "data": DataProto, "use_nccl_scatter": bool}, ...}
        self.pending_ops: dict[str, dict[str, Any]] = {}

        # Update API server URL to use master address if it's using localhost
        if "localhost" in normalized_url or "127.0.0.1" in normalized_url:
            # Replace localhost/127.0.0.1 with master address for multi-node support
            normalized_url = normalized_url.replace("localhost", self.master_addr).replace(
                "127.0.0.1", self.master_addr
            )

        self.api_server_url: str = normalized_url
        logger.info(f"Rank {rank}: API server URL: {self.api_server_url}")

        # Get base port from API server if not provided
        if base_port is None:
            base_port, self.identifier, self.dispatch_mode = self._register_with_api_server()

        self.base_port = base_port

        # Worker: Subscriber to receive commands AND execute signals from API server
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://{self.master_addr}:{base_port}")
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages

        # Worker: Pusher to send results to API server
        self.pusher = self.context.socket(zmq.PUSH)
        self.pusher.connect(f"tcp://{self.master_addr}:{base_port + 1}")

        # Worker: BIND PULL socket for data reception (direct from client)
        # In the new architecture, workers bind and clients connect (reversed from old)
        self.data_port = self._get_data_port()
        self.worker_ip = self._get_own_ip()
        self.data_puller = self.context.socket(zmq.PULL)
        self.data_puller.bind(f"tcp://*:{self.data_port}")

        self.backend = backend

        logger.info(
            f"Rank {rank}: ZMQ coordinator - SUB connected to {self.master_addr}:{base_port}, "
            f"PUSH connected to {self.master_addr}:{base_port + 1}, "
            f"PULL bound on *:{self.data_port}"
        )
        logger.info(
            f"Rank {rank}: Worker group identifier: {self.identifier}, dispatch mode: {self.dispatch_mode}"
        )

        # Give sockets time to connect/bind
        time.sleep(1)

        # Report our data endpoint to API server so direct clients can find us
        self._report_endpoint_to_api_server()

        # Start heartbeat thread for rank 0
        if rank == 0 and self.api_server_url and self.identifier:
            self._start_heartbeat()

    def _get_data_port(self) -> int:
        """Get the data port for this worker (fixed range based on rank)

        Uses NEXRL_WORKER_DATA_BASE_PORT env var (default: 6000) + rank.
        """
        base = int(os.environ.get("NEXRL_WORKER_DATA_BASE_PORT", "6000"))
        return base + self.rank

    def _get_own_ip(self) -> str:
        """Get this worker's externally reachable IP address

        Priority:
        1. NEXRL_WORKER_IP environment variable (for K8s pod IP injection)
        2. IP address from socket connection to API server (most reliable for cross-node)
        3. Resolved IP from hostname
        4. Fallback to localhost

        NOTE: Using hostname instead of IP can cause cross-node connection issues
        when nodes are on different subnets and DNS resolution differs between nodes.
        """
        import socket as sock_module
        from urllib.parse import urlparse

        # Check for explicit override (K8s downward API: status.podIP)
        explicit_ip = os.environ.get("NEXRL_WORKER_IP")
        if explicit_ip:
            logger.info(f"Rank {self.rank}: Using explicit worker IP from env: {explicit_ip}")
            return explicit_ip

        # Best method: Get our IP by connecting to the API server
        # This gives us the IP that's routable from the API server's network
        if self.api_server_url:
            try:
                parsed = urlparse(self.api_server_url)
                api_host = parsed.hostname
                api_port = parsed.port or 80

                # Create a temporary socket to API server to discover our IP
                with sock_module.socket(sock_module.AF_INET, sock_module.SOCK_DGRAM) as s:
                    # UDP socket doesn't actually connect, just sets destination for getsockname
                    s.connect((api_host, api_port))
                    local_ip = s.getsockname()[0]
                    logger.info(
                        f"Rank {self.rank}: Using routable IP from API server connection: {local_ip}"
                    )
                    return local_ip
            except Exception as e:
                logger.warning(
                    f"Rank {self.rank}: Could not determine IP from API server connection: {e}"
                )

        # Fallback: Try to get IP from hostname
        try:
            hostname = sock_module.gethostname()
            ip_addr = sock_module.gethostbyname(hostname)
            logger.info(
                f"Rank {self.rank}: Using IP from hostname resolution: {ip_addr} (hostname={hostname})"
            )
            return ip_addr
        except sock_module.error:
            logger.warning(f"Rank {self.rank}: Could not resolve hostname, using localhost")
            return "localhost"

    def _report_endpoint_to_api_server(self) -> None:
        """Report our data endpoint to API server

        After binding the PULL socket, report our IP and port to the API server
        so that direct clients can discover and connect to us.
        """
        if not self.identifier:
            logger.warning(f"Rank {self.rank}: No identifier, skipping endpoint reporting")
            return

        try:
            response = requests.post(
                f"{self.api_server_url}/report_worker_endpoint",
                json={
                    "identifier": self.identifier,
                    "rank": self.rank,
                    "worker_ip": self.worker_ip,
                    "data_port": self.data_port,
                },
                timeout=10,
            )
            response.raise_for_status()
            logger.info(
                f"Rank {self.rank}: Reported endpoint {self.worker_ip}:{self.data_port} to API server"
            )
        except Exception as e:
            logger.error(f"Rank {self.rank}: Failed to report endpoint to API server: {e}")
            raise

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
                            logger.info(
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
        heartbeat_timeout = (
            10  # 10 second timeout for requests (increased to handle API server load)
        )

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
                    logger.info(
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

    def _nccl_scatter_data(self, data: Any, src_rank: int = 0) -> Any:
        """DP-aware NCCL scatter: rank 0 sends data, split by DP groups, replicated within SP groups

        This ensures that:
        - Data is split by Data Parallel (DP) groups
        - Workers in the same DP group but different SP ranks get the SAME data
        - Ulysses all-gather becomes unnecessary (workers already have what they need)

        Args:
            data: Full data on rank 0, None on other ranks
            src_rank: Source rank (default 0)

        Returns:
            Scattered chunk for this rank
        """
        global worker

        # Get SP size from worker config (if available)
        sp_size = 1
        if worker is not None and hasattr(worker, "ulysses_sequence_parallel_size"):
            sp_size = worker.ulysses_sequence_parallel_size
            logger.info(f"[WORKER-{self.rank}] Using SP size {sp_size} from worker config")

        # Calculate DP size
        dp_size = self.world_size // sp_size

        logger.info(
            f"[WORKER-{self.rank}] NCCL scatter topology: world_size={self.world_size}, "
            f"sp_size={sp_size}, dp_size={dp_size}"
        )

        # Use scatter_object_list for DataProto objects
        scatter_list: Optional[list[Any]] = None
        if self.rank == src_rank:
            from ..utils.protocol import DataProto

            if isinstance(data, DataProto):
                # Split by DP groups
                dp_chunks = data.chunk(dp_size)

                # Replicate each DP chunk to all SP ranks in that DP group
                # Worker layout: [dp0_sp0, dp0_sp1, ..., dp0_sp(N-1), dp1_sp0, dp1_sp1, ...]
                scatter_list = []
                for dp_rank in range(dp_size):
                    for sp_rank in range(sp_size):
                        scatter_list.append(dp_chunks[dp_rank])  # Same chunk for all SP ranks

                logger.info(
                    f"[WORKER-{self.rank}] Split data into {dp_size} DP chunks, "
                    f"replicated {sp_size} times each = {len(scatter_list)} total chunks"
                )
            else:
                # Fallback: replicate data to all ranks
                logger.warning(
                    f"[WORKER-{self.rank}] NCCL scatter received non-DataProto data, "
                    f"replicating to all ranks"
                )
                scatter_list = [data] * self.world_size
        else:
            scatter_list = [None] * self.world_size

        # Scatter: each rank receives scatter_list[rank]
        output_list: list[Any] = [None]
        torch.distributed.scatter_object_list(output_list, scatter_list, src=src_rank)

        logger.info(
            f"[WORKER-{self.rank}] ✓ DP-aware NCCL scatter complete (dp_size={dp_size}, sp_size={sp_size})"
        )
        return output_list[0]

    def _nccl_gather_result(self, result: Any, dst_rank: int = 0) -> Any:
        """NCCL gather: all ranks send results to rank 0

        Args:
            result: Result from this rank
            dst_rank: Destination rank (default 0)

        Returns:
            Gathered results list on rank 0, None on other ranks
        """
        # Use gather_object for results
        gather_list: Optional[list[Any]]
        if self.rank == dst_rank:
            gather_list = [None] * self.world_size
        else:
            gather_list = None

        torch.distributed.gather_object(result, gather_list, dst=dst_rank)

        if self.rank == dst_rank:
            assert gather_list is not None  # Type narrowing for mypy
            logger.info(
                f"[WORKER-{self.rank}] ✓ NCCL gather complete, received {len(gather_list)} results"
            )
            # Concatenate gathered DataProto results
            from ..utils.protocol import DataProto

            if gather_list and isinstance(gather_list[0], DataProto):
                return DataProto.concat(gather_list)
            else:
                # Return list as-is if not DataProto
                return gather_list
        else:
            logger.info(f"[WORKER-{self.rank}] ✓ NCCL gather sent result to rank {dst_rank}")
            return None

    def worker_loop(self):
        """Main worker loop to handle commands and data from API server / direct clients

        Two-phase protocol with ACK-based ordering (prevents race conditions):
        1. Client → Rank 0: Send data via PUSH socket
        2. Rank 0 → Client: Send ACK via PULL socket (confirms data received)
        3. Client waits for ACK, then broadcasts execute signal via PUB/SUB
        4. All workers: Receive execute signal, participate in NCCL scatter/gather

        This ensures execute signal never arrives before data on rank 0.
        """
        logger.info(f"Rank {self.rank}: Starting ZMQ worker loop (two-phase protocol enabled)")

        # Set up polling for both subscriber and data_puller
        poller = zmq.Poller()
        poller.register(self.subscriber, zmq.POLLIN)
        poller.register(self.data_puller, zmq.POLLIN)

        # Log once at start
        poll_count = 0
        last_debug_log = time.time()

        while True:
            try:
                # Poll for messages with a short timeout
                socks = dict(poller.poll(timeout=100))  # 100ms timeout
                poll_count += 1

                # Log polling status every 10 seconds for debugging
                now = time.time()
                if now - last_debug_log > 10:
                    logger.debug(
                        f"[WORKER-{self.rank}] Polling status: {poll_count} polls, sockets ready: {len(socks)}, pending_ops: {len(self.pending_ops)}"
                    )
                    last_debug_log = now

                # Log when any socket has data
                if socks:
                    socket_names = []
                    if self.subscriber in socks:
                        socket_names.append("SUB")
                    if self.data_puller in socks:
                        socket_names.append("PULL")
                    logger.debug(
                        f"[WORKER-{self.rank}] Poll returned {len(socks)} sockets: {socket_names}"
                    )

                # Check for messages via SUB socket (commands and execute signals)
                if self.subscriber in socks and socks[self.subscriber] == zmq.POLLIN:
                    logger.info(
                        f"[WORKER-{self.rank}] SUB socket has message available, receiving..."
                    )
                    message = self.subscriber.recv_pyobj()
                    logger.info(
                        f"[WORKER-{self.rank}] SUB message: phase={message.get('phase')}, op_id={message.get('op_id')}"
                    )

                    # TWO-PHASE PROTOCOL: Phase 2 - Execute signal
                    if message.get("phase") == "execute":
                        op_id = message.get("op_id")
                        logger.info(
                            f"[WORKER-{self.rank}] ★ Received execute signal for op_id={op_id}"
                        )
                        logger.info(
                            f"[WORKER-{self.rank}] Current pending_ops: {list(self.pending_ops.keys())}"
                        )

                        if op_id in self.pending_ops:
                            # Data already received (on rank 0) or execute signal arrived
                            pending = self.pending_ops.pop(op_id)
                            operation = pending.get("operation")
                            use_nccl_scatter = pending.get("use_nccl_scatter", False)

                            # Get data: rank 0 has it, others get None initially
                            data_proto = pending.get("data") if self.rank == 0 else None

                            # In NCCL mode, broadcast operation name from rank 0 to all workers
                            if use_nccl_scatter:
                                # Broadcast operation info
                                operation_info = [operation] if self.rank == 0 else [None]
                                torch.distributed.broadcast_object_list(operation_info, src=0)
                                operation = operation_info[0]

                                logger.info(
                                    f"[WORKER-{self.rank}] ► Executing {operation} for op_id={op_id} "
                                    f"(NCCL scatter/gather mode)"
                                )
                            else:
                                logger.info(
                                    f"[WORKER-{self.rank}] ► Executing {operation} for op_id={op_id} "
                                    f"(legacy mode)"
                                )

                            try:
                                # If using NCCL scatter, rank 0 scatters data to all workers
                                if use_nccl_scatter:
                                    logger.info(
                                        f"[WORKER-{self.rank}] 🔄 NCCL scatter starting from rank 0..."
                                    )
                                    data_proto = self._nccl_scatter_data(data_proto, src_rank=0)
                                    logger.info(
                                        f"[WORKER-{self.rank}] ✓ NCCL scatter complete, received data chunk"
                                    )

                                # All workers execute with their data chunk
                                if operation is None:
                                    raise ValueError(f"Operation is None for op_id={op_id}")
                                if data_proto is None:
                                    raise ValueError(f"Data is None for op_id={op_id}")
                                result = self._execute_operation(operation, data_proto)
                                logger.info(
                                    f"[WORKER-{self.rank}] ✓ Operation {operation} completed successfully"
                                )

                                # If using NCCL gather, all workers send results to rank 0
                                if use_nccl_scatter:
                                    logger.info(
                                        f"[WORKER-{self.rank}] 🔄 NCCL gather starting to rank 0..."
                                    )
                                    gathered_result = self._nccl_gather_result(result, dst_rank=0)

                                    if self.rank == 0:
                                        # Rank 0 sends gathered result to client
                                        logger.info(
                                            f"[WORKER-{self.rank}] ✓ NCCL gather complete, sending to client"
                                        )
                                        response = {
                                            "rank": self.rank,
                                            "op_id": op_id,
                                            "success": True,
                                            "result": gathered_result,
                                        }
                                        self.pusher.send_pyobj(response)
                                        logger.info(
                                            f"[WORKER-{self.rank}] ✓ Result sent successfully"
                                        )
                                    else:
                                        # Other ranks don't send results (already gathered to rank 0)
                                        logger.info(
                                            f"[WORKER-{self.rank}] ✓ Result gathered to rank 0, no ZMQ send needed"
                                        )
                                else:
                                    # Legacy mode: all workers send results individually
                                    response = {
                                        "rank": self.rank,
                                        "op_id": op_id,
                                        "success": True,
                                        "result": result,
                                    }
                                    self.pusher.send_pyobj(response)
                                    logger.info(f"[WORKER-{self.rank}] ✓ Result sent successfully")

                            except Exception as e:
                                tb = traceback.format_exc()
                                logger.error(
                                    f"[WORKER-{self.rank}] ✗ Operation {operation} failed: {e}"
                                )
                                logger.error(f"[WORKER-{self.rank}] Traceback:\n{tb}")

                                # Always send error responses (even in NCCL mode)
                                response = {
                                    "rank": self.rank,
                                    "op_id": op_id,
                                    "success": False,
                                    "error": str(e),
                                    "traceback": tb,
                                }
                                self.pusher.send_pyobj(response)
                        else:
                            # Data not in pending_ops
                            if self.rank == 0:
                                # BUG: With ACK-based ordering, execute should never arrive before data on rank 0
                                logger.error(
                                    f"[WORKER-{self.rank}] 🐛 BUG: Execute signal arrived before data for op_id={op_id}. "
                                    f"This violates ACK-based ordering! Client should wait for data ACK before sending execute."
                                )
                                continue

                            # Non-rank-0 workers: This is EXPECTED in NCCL scatter mode
                            # Only rank 0 receives data via PULL socket, others participate via NCCL
                            logger.info(
                                f"[WORKER-{self.rank}] ⚠ Execute signal received, participating in NCCL scatter for op_id={op_id}"
                            )

                            # Broadcast operation name from rank 0
                            operation_info = [None]
                            torch.distributed.broadcast_object_list(operation_info, src=0)
                            operation = operation_info[0]

                            logger.info(
                                f"[WORKER-{self.rank}] ► Executing {operation} for op_id={op_id} "
                                f"(NCCL scatter/gather mode)"
                            )

                            try:
                                # NCCL scatter data from rank 0
                                logger.info(
                                    f"[WORKER-{self.rank}] 🔄 NCCL scatter starting from rank 0..."
                                )
                                data_proto = self._nccl_scatter_data(None, src_rank=0)
                                logger.info(
                                    f"[WORKER-{self.rank}] ✓ NCCL scatter complete, received data chunk"
                                )

                                # Execute operation
                                if operation is None:
                                    raise ValueError(f"Operation is None for op_id={op_id}")
                                if data_proto is None:
                                    raise ValueError(
                                        f"Data is None after NCCL scatter for op_id={op_id}"
                                    )
                                result = self._execute_operation(operation, data_proto)
                                logger.info(
                                    f"[WORKER-{self.rank}] ✓ Operation {operation} completed successfully"
                                )

                                # NCCL gather results to rank 0
                                logger.info(
                                    f"[WORKER-{self.rank}] 🔄 NCCL gather starting to rank 0..."
                                )
                                gathered_result = self._nccl_gather_result(result, dst_rank=0)

                                # Other ranks don't send results (already gathered to rank 0)
                                logger.info(
                                    f"[WORKER-{self.rank}] ✓ Result gathered to rank 0, no ZMQ send needed"
                                )

                            except Exception as e:
                                tb = traceback.format_exc()
                                logger.error(
                                    f"[WORKER-{self.rank}] ✗ Operation {operation} failed: {e}"
                                )
                                logger.error(f"[WORKER-{self.rank}] Traceback:\n{tb}")

                                # Send error response
                                response = {
                                    "rank": self.rank,
                                    "op_id": op_id,
                                    "success": False,
                                    "error": str(e),
                                    "traceback": tb,
                                }
                                self.pusher.send_pyobj(response)

                        continue

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
                        "update_actor_with_distillation",
                        "generate_sequences",
                        "compute_ref_log_prob",
                    ]:
                        # Legacy broadcast path is removed - use two-phase protocol via PULL socket
                        logger.error(
                            f"Rank {self.rank}: Received legacy broadcast operation '{operation}' via SUB socket. "
                            f"This is no longer supported. Use DirectZMQTrainServiceClient with two-phase protocol."
                        )
                        response = {
                            "rank": self.rank,
                            "success": False,
                            "error": f"Legacy broadcast operations are no longer supported. Use DirectZMQTrainServiceClient.",
                        }
                        self.pusher.send_pyobj(response)

                # Check for data messages via PULL socket (direct from client)
                if self.data_puller in socks and socks[self.data_puller] == zmq.POLLIN:
                    logger.info(
                        f"[WORKER-{self.rank}] PULL socket has data available, receiving..."
                    )
                    message = self.data_puller.recv_pyobj()
                    logger.info(
                        f"[WORKER-{self.rank}] Received message with keys: {list(message.keys())}, phase={message.get('phase')}"
                    )

                    # Handle ping message for connectivity verification
                    if message.get("phase") == "ping":
                        op_id = message.get("op_id")
                        logger.info(
                            f"[WORKER-{self.rank}] Received ping for op_id={op_id}, sending pong"
                        )
                        pong_response = {
                            "rank": self.rank,
                            "op_id": op_id,
                            "pong": True,
                            "success": True,
                            "timestamp": time.time(),
                        }
                        self.pusher.send_pyobj(pong_response)
                        logger.info(f"[WORKER-{self.rank}] ✓ Pong sent for op_id={op_id}")
                        continue

                    # TWO-PHASE PROTOCOL: Phase 1 - Store data, don't execute yet
                    if message.get("phase") == "data":
                        recv_time = time.time()
                        op_id = message.get("op_id")
                        operation = message.get("operation")
                        data_proto = message.get("data")
                        use_nccl_scatter = message.get("use_nccl_scatter", False)
                        send_timestamp = message.get("timestamp", 0)

                        # Calculate receive latency to help diagnose network/serialization issues
                        if send_timestamp > 0:
                            recv_latency = recv_time - send_timestamp
                            logger.info(
                                f"[WORKER-{self.rank}] Received data message for op_id={op_id} "
                                f"(network+deserialization latency: {recv_latency:.2f}s)"
                            )
                        else:
                            logger.info(
                                f"[WORKER-{self.rank}] Received data message for op_id={op_id}"
                            )

                        # Check for duplicate data message
                        if op_id in self.pending_ops:
                            logger.error(
                                f"[WORKER-{self.rank}] 🐛 BUG DETECTED: Received duplicate data for op_id={op_id}! "
                                f"Already have operation='{self.pending_ops[op_id].get('operation')}', "
                                f"new message has operation='{operation}'. "
                                f"This indicates a client-side bug or duplicate send."
                            )
                            # Overwrite with new data (fail-safe behavior)

                        # Store in pending_ops, wait for execute signal
                        self.pending_ops[op_id] = {
                            "operation": operation,
                            "data": data_proto,
                            "use_nccl_scatter": use_nccl_scatter,
                        }
                        logger.info(
                            f"[WORKER-{self.rank}] ✓ Stored data for op_id={op_id}, operation={operation}, "
                            f"NCCL mode={use_nccl_scatter}, waiting for execute signal (pending_ops count: {len(self.pending_ops)})"
                        )

                        # Send ACK to client so it knows data was received
                        # Client will wait for this before broadcasting execute signal
                        # This prevents race condition where execute arrives before data
                        ack_start = time.time()
                        ack_response = {
                            "rank": self.rank,
                            "op_id": op_id,
                            "phase": "data_ack",
                            "success": True,
                            "timestamp": time.time(),
                        }
                        self.pusher.send_pyobj(ack_response)
                        ack_send_time = time.time() - ack_start
                        logger.info(
                            f"[WORKER-{self.rank}] ✓ Sent data ACK for op_id={op_id} "
                            f"(ACK send took {ack_send_time:.3f}s)"
                        )

                        continue

                    # Message without phase="data" - this is an error in the new architecture
                    logger.error(
                        f"Rank {self.rank}: Received message without 'phase' field on PULL socket. "
                        f"Only two-phase protocol is supported. Message keys: {list(message.keys())}"
                    )
                    response = {
                        "rank": self.rank,
                        "success": False,
                        "error": "Invalid message format. Use two-phase protocol with phase='data' and phase='execute'.",
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
            return worker.compute_log_prob(data_proto)
        elif operation == "update_actor":
            return worker.update_actor(data_proto)
        elif operation == "update_actor_with_distillation":
            return worker.update_actor_with_distillation(data_proto)
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
