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


"""Debug and performance utilities for NexTrainer"""

import logging
from typing import Optional

import psutil
import torch
import torch.distributed as dist


def log_gpu_memory_usage(
    head: str, logger: Optional[logging.Logger] = None, level=logging.DEBUG, rank: int = 0
):
    """
    Log GPU and CPU memory usage statistics.

    Args:
        head: Header message for the log
        logger: Optional logger instance (if None, prints to stdout)
        level: Logging level (default: logging.DEBUG)
        rank: Only log if this rank matches the current rank (None to always log)
    """
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3

        cpu_percent = psutil.cpu_percent()
        cpu_memory_used = psutil.virtual_memory().used / 1024**3
        cpu_memory_total = psutil.virtual_memory().total / 1024**3

        # get pagecache usage
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
                lines = meminfo.split("\n")
                cached_line = [line for line in lines if "Cached:" in line][0]
                cached_kb = int(cached_line.split()[1])
                cached_gb = cached_kb / 1024**2
        except:
            cached_gb = 0.0

        if torch.distributed.get_rank() % 8 == 0:
            message = (
                f"{head}, Rank {torch.distributed.get_rank()}, "
                f"GPU memory allocated (GB): {memory_allocated:.2f}, "
                f"GPU memory reserved (GB): {memory_reserved:.2f}, "
                f"CPU usage: {cpu_percent:.2f}%, "
                f"CPU memory used (GB): {cpu_memory_used:.2f}/{cpu_memory_total:.2f}, "
                f"PageCache (GB): {cached_gb:.2f}"
            )
            if logger is None:
                print(message)
            else:
                logger.log(msg=message, level=level)
