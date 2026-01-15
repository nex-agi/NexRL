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

import concurrent.futures
import io
import json
import logging
import os
import random
import shutil
import tempfile
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import IO, Any, cast

import numpy as np
import torch
import torch.distributed
from filelock import FileLock

# Optional import for internal-only weight provider feature
try:
    from north_checkpoint.server import WeightProvider

    HAS_WEIGHT_PROVIDER = True
except ImportError:
    WeightProvider = None  # type: ignore
    HAS_WEIGHT_PROVIDER = False

from torch import Tensor
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner, ReadItem
from torch.distributed.fsdp import (
    FullStateDictConfig,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.tensor import DTensor
from torch.futures import Future
from transformers import PreTrainedTokenizer

# ======================= Utility Classes and Functions =======================


class TensorType(Enum):
    NON_TP = "non_tp"
    ROW_SHARDED_TP = "row_sharded_tp"
    COLUMN_SHARDED_TP = "column_sharded_tp"


@dataclass
class TensorInfo:
    tensor: torch.Tensor
    # metadata
    offsets: list[int]
    global_shape: list[int]


@dataclass
class WeightProviderInfo:
    host: str
    port: int
    address: str
    tp_rank: int = 0
    tp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1
    parameter_names: list[str] | None = None

    def __post_init__(self):
        if self.parameter_names is None:
            self.parameter_names = []
        self.address = f"{self.host}:{self.port}"


def xor8_digest_chunked(t: torch.Tensor, chunk_bytes: int = 1 << 20) -> int:
    """
    One‑byte XOR checksum, processed in `chunk_bytes` blocks.
    • No modification of `t`.
    • Peak extra memory ≤ `chunk_bytes` (defaults to 1 MiB).
    • Works on CPU or GPU, PyTorch ≥ 1.10.

    Returns: int in [0, 255]
    """
    byte_view = t.view(torch.uint8).clone().flatten().contiguous()
    parity = torch.zeros((), dtype=torch.uint8, device=t.device)

    # --- fallback path for older PyTorch -------------------------------
    def _reduce_chunk(buf: torch.Tensor) -> torch.Tensor:
        """
        In‑place power‑of‑two fold (same idea as earlier, but confined
        to this small chunk).  Uses ≤ `buf.numel()` bytes of workspace.
        """
        n = buf.numel()
        while n > 1:
            if n & 1:  # odd length → fold last byte in
                buf[0] ^= buf[n - 1]
                n -= 1
            half = n >> 1
            buf[:half] = torch.bitwise_xor(buf[:half], buf[half:n])
            n = half
        return buf[0]

    for start in range(0, byte_view.numel(), chunk_bytes):
        end = min(start + chunk_bytes, byte_view.numel())
        # clone only this small slice → keeps original tensor pristine
        chunk = byte_view[start:end]
        parity ^= _reduce_chunk(chunk)

    return int(parity.item())


class ThreadedFileSystemReader(FileSystemReader):
    def __init__(self, weight_path: str, num_threads: int = 16):
        super().__init__(weight_path)
        self.num_threads = num_threads
        self.commit_lock = threading.Lock()

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        s = time.time()
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        def process_file(file_info: tuple[str, list[ReadItem]]) -> None:
            relative_path, reqs = file_info
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(file_slice.read(item_md.length))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                    else:
                        tensor = cast(
                            Tensor,
                            torch.load(
                                cast(IO[bytes], file_slice),
                                map_location="cpu",
                                weights_only=True,
                            ),
                        )
                        tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                        target_tensor = planner.resolve_tensor(req).detach()
                        target_tensor.copy_(tensor)

                        with self.commit_lock:
                            planner.commit_tensor(req, target_tensor)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(executor.map(process_file, per_file.items()))

        fut: Future = Future()
        fut.set_result(None)
        print(f"ThreadedFileSystemReader read_data time: {time.time() - s:.2f} s")
        return fut


# Megatron imports removed - Megatron support removed

# ======================= Base Checkpoint Manager =======================


class BaseCheckpointManager:
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        tokenizer: PreTrainedTokenizer,
    ):
        self.previous_global_step: int | None = None
        self.previous_save_local_path: str | None = None

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.world_size = (
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        )

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def remove_previous_save_local_path(self):
        if not self.previous_save_local_path:
            return

        abs_path = os.path.abspath(self.previous_save_local_path)
        print(f"Checkpoint manager remove previous save local path: {abs_path}")
        if not os.path.exists(abs_path):
            return

        # remove previous local_path
        shutil.rmtree(abs_path, ignore_errors=True)

    @staticmethod
    def local_mkdir(path):
        if not os.path.isabs(path):
            working_dir = os.getcwd()
            path = os.path.join(working_dir, path)

        with FileLock(os.path.join(tempfile.gettempdir(), path.replace("/", "_") + ".lock")):
            # make a new dir
            os.makedirs(path, exist_ok=True)

        return path

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        torch.cuda.set_rng_state(rng_state["cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])


# WPMegatronCheckpointManager and MegatronCheckpointManager removed - Megatron support removed


# ======================= DCP Checkpoint Manager =======================


class DCPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        tokenizer: PreTrainedTokenizer,
        *args,
        **kwargs,
    ):
        super().__init__(model, optimizer, lr_scheduler, tokenizer)

        # TODO(sunpeng): add model class
        self.rollout_tp = kwargs.get("rollout_tp", 1)
        self.use_weight_provider = kwargs.get("use_weight_provider", False)
        if self.use_weight_provider:
            if not HAS_WEIGHT_PROVIDER:
                raise ImportError(
                    "Weight provider requested (use_weight_provider=True) but 'north_checkpoint' "
                    "is not installed. This is an internal-only feature. "
                    "Please set use_weight_provider=False in your recipe configuration, "
                    "or use a different weight sync_method (e.g., 'disk' instead of 'network')."
                )

            # for unit test ingestion
            mock_weight_provider = kwargs.get("mock_weight_provider", None)
            if mock_weight_provider is not None:
                self.provider = mock_weight_provider
                self.provider_info = self.provider.get_provider_info()
                print(f"use mock weight provider")
                return

            self.provider = WeightProvider()
            self.provider_info = self.provider.get_provider_info()

            self.thread = threading.Thread(target=self.provider.wait_for_server_close, daemon=True)
            self.thread.start()
            print(f"WeightProvider started, info: {self.provider_info}")

    def load_checkpoint(
        self, path=None, del_local_after_load=True, load_weight_only=False, *args, **kwargs
    ):
        if path is None:
            return

        # every rank will load its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

        with FSDP.state_dict_type(
            self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
        ):
            fs_storage_reader = ThreadedFileSystemReader(path)
            load_state_dict = self.gen_state_dict()
            load_state_dict["model"] = self.model.state_dict()

            if self.optimizer is not None:
                load_state_dict["optimizer"] = self.optimizer.state_dict()

            if self.lr_scheduler is not None:
                lr_scheduler_state_dict = (
                    self.lr_scheduler.state_dict() if self.lr_scheduler else None
                )
                load_state_dict["extra_state_dict"] = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                }

            torch.distributed.checkpoint.load_state_dict(
                state_dict=load_state_dict,
                storage_reader=fs_storage_reader,
            )
            self.model.load_state_dict(load_state_dict["model"])

            if not load_weight_only:
                self.optimizer.load_state_dict(load_state_dict["optimizer"])
                self.lr_scheduler.load_state_dict(
                    load_state_dict["extra_state_dict"]["lr_scheduler"]
                )
                self.load_rng_state(load_state_dict["extra_state_dict"]["rng"])

        # wait for everyone to finish loading
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def gen_state_dict(self):
        return {"model": None, "optimizer": None, "extra_state_dict": None}

    def get_tensor_type(self, model_id: str, key: str) -> TensorType:
        assert model_id == "qwen2" or model_id == "llama3.2", "only support qwen2 and llama3.2 now"

        if model_id == "qwen2" or model_id == "llama3.2":
            if key.endswith(".bias") or "norm" in key or "lm_head" in key or "embed_tokens" in key:
                return TensorType.NON_TP
            if "down" in key or "o_proj" in key:
                return TensorType.COLUMN_SHARDED_TP

        return TensorType.ROW_SHARDED_TP

    def dtensor_to_tp_tensor(
        self, dtensor: DTensor, local_rank: int, tp_size: int, tensor_type: TensorType
    ) -> dict[int, TensorInfo]:
        """
        for now, dtensor is always row-sharded
        but we have column-sharded weights in vllm engine
        so we need to convert dtensor to tp_tensor.
        For row parallel (simplier case):
            we need to check which tp rank the dtensor should be sent to based on the offset
        For column parallel (more complex case):
            all tensors should be sharded again based on the column dim
            so we will have TP size tensors
        Returns:
            dict[int, torch.Tensor]: target inference engine tp rank, to local tensor
                -1 means non tp tensor
        """
        assert type(dtensor) == DTensor, "only support dtensor now"

        global_shape = dtensor.shape
        local_tensor = dtensor.to_local().to(torch.bfloat16)
        mesh_size = dtensor.device_mesh.size()
        placement = dtensor.placements[0]
        if hasattr(placement, "dim"):
            shard_dim = placement.dim
        else:  # for unit test
            shard_dim = 0
        assert shard_dim == 0, "only support row-sharded dtensor now"
        assert (
            global_shape[shard_dim] % mesh_size == 0
        ), "global shape should be divisible by mesh size"
        assert (
            mesh_size >= tp_size and mesh_size % tp_size == 0
        ), "mesh should be divisible by tp size"

        ranks_per_tp = mesh_size // tp_size
        shard_offsets = [0 for _ in range(len(global_shape))]
        per_shard_size = global_shape[shard_dim] // mesh_size
        shard_offsets[shard_dim] = local_rank * per_shard_size

        if tensor_type == TensorType.NON_TP:
            return {
                -1: TensorInfo(
                    tensor=local_tensor, offsets=shard_offsets, global_shape=global_shape
                )
            }
        elif tensor_type == TensorType.ROW_SHARDED_TP:
            assert len(global_shape) == 2, "only support 2D tensor for row parallel"
            target_tp_rank = local_rank // ranks_per_tp
            return {
                target_tp_rank: TensorInfo(
                    tensor=local_tensor, offsets=shard_offsets, global_shape=global_shape
                )
            }
        elif tensor_type == TensorType.COLUMN_SHARDED_TP:
            assert len(global_shape) == 2, "only support 2D tensor for column parallel"
            column_size = global_shape[1]
            assert column_size % tp_size == 0, "column size should be divisible by tp size"
            column_size_per_tp = column_size // tp_size

            result = {}
            for tp_rank in range(tp_size):
                result[tp_rank] = TensorInfo(
                    tensor=local_tensor[
                        :, tp_rank * column_size_per_tp : (tp_rank + 1) * column_size_per_tp
                    ],
                    offsets=[shard_offsets[0], tp_rank * column_size_per_tp],
                    global_shape=global_shape,
                )
            return result
        else:
            raise ValueError(f"Unsupported tensor type: {tensor_type}")

    def compose_state_dict(self, all_local_tensors: dict[int, dict[str, TensorInfo]]):
        """
        Returns:
        2 dicts:
        - state_dict: key -> tensor
        - metadata: key -> metadata
        Example metadata:
        {
            "keys": list(state_dict.keys()),
            "shapes": [t.shape for t in state_dict.values()],
            "numels": [t.numel() for t in state_dict.values()],
            "dtypes": [str(t.dtype) for t in state_dict.values()],
            'sharded_metadata': shared_tensor_metadata,
        }
        """
        state_dict_results: dict[int, dict[str, Any]] = defaultdict(dict)
        metadata_results: dict[int, dict[str, Any]] = defaultdict(dict)

        for tp_rank, tensor_infos in all_local_tensors.items():
            state_dict_this_tp = OrderedDict()
            offsets_this_tp = OrderedDict()
            global_shapes_this_tp = OrderedDict()
            for k, v in tensor_infos.items():
                state_dict_this_tp[k] = v.tensor
                offsets_this_tp[k] = v.offsets
                global_shapes_this_tp[k] = v.global_shape

            metadata_this_tp = {
                "keys": list(state_dict_this_tp.keys()),
                "shapes": [t.shape for t in state_dict_this_tp.values()],
                "numels": [t.numel() for t in state_dict_this_tp.values()],
                "dtypes": [str(t.dtype) for t in state_dict_this_tp.values()],
                "offsets": list(offsets_this_tp.values()),
                "global_shapes": list(global_shapes_this_tp.values()),
                "digests": [xor8_digest_chunked(t) for t in state_dict_this_tp.values()],
            }

            state_dict_results[tp_rank] = state_dict_this_tp
            metadata_results[tp_rank] = metadata_this_tp

        return state_dict_results, metadata_results

    def save_checkpoint(
        self,
        local_path: str,
        global_step: int,
        remove_previous_ckpt=False,
        save_weight_only=False,
        saved_fully_shared_ckpt=True,
        *args,
        **kwargs,
    ):
        # record the previous global step
        self.previous_global_step: int | None = global_step

        # remove previous local_path
        # TODO: shall we remove previous ckpt every save?
        if remove_previous_ckpt:
            self.remove_previous_save_local_path()
        local_path = self.local_mkdir(local_path)

        print(
            f"enter save_checkpoint, save_weight_only: {save_weight_only}, saved_fully_shared_ckpt: {saved_fully_shared_ckpt}!!"
        )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

        local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if self.use_weight_provider and global_step == 0:
            fsdp_start_time = time.time()
            with FSDP.state_dict_type(
                self.model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                state_dict = self.model.state_dict()
                got_weight_end_time = time.time()
                print(f"got weight took: {got_weight_end_time - fsdp_start_time} s")

                prepare_time = time.time()

                # TP_rank -> key -> tensor_info
                all_local_tensors: dict[int, dict[str, TensorInfo]] = defaultdict(dict)

                for k, v in state_dict.items():
                    tensor_type = self.get_tensor_type("qwen2", k)
                    tensors = self.dtensor_to_tp_tensor(v, local_rank, self.rollout_tp, tensor_type)
                    for tp_rank, tensor_info in tensors.items():
                        all_local_tensors[tp_rank][k] = tensor_info
                state_dict_results, metadata_results = self.compose_state_dict(all_local_tensors)

                prepare_end_time = time.time()
                print(f"prepare took: {prepare_end_time - prepare_time} s")

                start_time = time.time()
                non_tp_state_dicts = state_dict_results.pop(-1) if -1 in state_dict_results else {}
                non_tp_metadata = metadata_results.pop(-1) if -1 in metadata_results else {}
                self.provider.update_weights(
                    non_tp_state_dicts, state_dict_results, non_tp_metadata, metadata_results
                )

                end_time = time.time()
                print(f"update weights took: {end_time - start_time} s")

                with open(
                    os.path.join(
                        local_path, f"non_tp_tensors_server_rank_{local_rank}.wp_metadata.json"
                    ),
                    "w",
                ) as f:
                    json.dump(
                        {
                            "host_info": {"address": self.provider_info.address},
                            "metadata": non_tp_metadata,
                        },
                        f,
                    )

                for tp_rank, metadata in metadata_results.items():
                    with open(
                        os.path.join(
                            local_path,
                            f"tp_tensors_{tp_rank}_server_rank_{local_rank}.wp_metadata.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(
                            {
                                "host_info": {
                                    "address": self.provider_info.address,
                                },
                                "metadata": metadata,
                            },
                            f,
                        )

        elif saved_fully_shared_ckpt:
            with FSDP.state_dict_type(
                self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
            ):
                fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(
                    local_path, thread_count=4
                )
                saved_state_dict = self.gen_state_dict()
                saved_state_dict["model"] = self.model.state_dict()

                if not save_weight_only:
                    optimizer_state_dict = self.optimizer.state_dict() if self.optimizer else None
                    lr_scheduler_state_dict = (
                        self.lr_scheduler.state_dict() if self.lr_scheduler else None
                    )
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    if optimizer_state_dict is not None:
                        saved_state_dict["optimizer"] = optimizer_state_dict
                    if extra_state_dict is not None:
                        saved_state_dict["extra_state_dict"] = extra_state_dict

                torch.distributed.checkpoint.save_state_dict(
                    state_dict=saved_state_dict,
                    storage_writer=fs_storage_writer,
                )
        else:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                state_dict = self.model.state_dict()
                if torch.distributed.get_rank() == 0:
                    self.model.save_pretrained(save_directory=local_path, state_dict=state_dict)
                    self.tokenizer.save_pretrained(local_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.previous_save_local_path: str | None = local_path


# ======================= Checkpoint Manager Factory =======================


def get_checkpoint_manager(manager_type: str):
    """
    Factory function to get checkpoint manager classes by type.

    Args:
        manager_type (str): Type of checkpoint manager ('dcp', etc.)

    Returns:
        Checkpoint manager class
    """
    managers = {
        "dcp": DCPCheckpointManager,
    }

    if manager_type not in managers:
        available_types = ", ".join(managers.keys())
        raise ValueError(
            f"Unknown checkpoint manager type '{manager_type}'. Available types: {available_types}"
        )

    return managers[manager_type]


# ======================= Utility Functions =======================
