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
Model utilities for NexTrainer
"""

import os
import warnings

import torch
from transformers import AutoTokenizer, GenerationConfig
from transformers.utils import is_remote_url


class PrecisionType:
    """Precision type utilities"""

    @staticmethod
    def to_dtype(dtype_str):
        """Convert string to torch dtype"""
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.float32)


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None."""
    import warnings

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}")


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer."""
    import warnings

    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107."
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def get_generation_config(model_path, trust_remote_code=False):
    """Get generation config from model"""
    try:
        return GenerationConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except:
        return None


def print_model_size(model):
    """Print model parameter count"""
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters ({param_count/1e9:.2f}B)")


def update_model_config(config, override_config_kwargs):
    """Update model config with override kwargs"""
    for key, value in override_config_kwargs.items():
        setattr(config, key, value)


def copy_local_path_from_hdfs(
    src: str, cache_dir=None, filelock=".file.lock", verbose=False
) -> str:
    """
    Copy src from hdfs to local if src is on hdfs or directly return src.
    """

    _HDFS_PREFIX = "hdfs://"

    def is_non_local(path):
        return path.startswith(_HDFS_PREFIX)

    assert (
        src[-1] != "/"
    ), f"Make sure the last char in src is not / because it will cause error. Got {src}"

    if is_non_local(src):
        # For HDFS paths, would need hdfs_io - for now raise error
        raise NotImplementedError("HDFS paths not supported in NexTrainer. Use local paths only.")
    else:
        # local path, directly return
        return src


def import_external_libs(external_lib_config):
    """Import external libraries - placeholder"""
    pass


def compute_position_id_with_mask(attention_mask):
    """Compute position IDs from attention mask"""
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


class FlopsCounter:
    """FLOPS counter - simplified placeholder"""

    def __init__(self, model_config):
        pass

    def estimate_flops(self, num_tokens, time_delta):
        return 0, 1e12


def get_checkpoint_manager(manager_type):
    """Get checkpoint manager"""
    from .checkpoint_manager import get_checkpoint_manager as _get_checkpoint_manager

    return _get_checkpoint_manager(manager_type)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps):
    """Get constant schedule with warmup"""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)
