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
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings

import torch
import torch.distributed
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh

# Import simplified base classes from NexTrainer
from ..distributed.worker_process import Worker
from ..utils import core_utils as nx_utils
from ..utils.dist_utils import FSDPUlyssesShardingManager
from ..utils.protocol import DataProto

# Removed unused decorator imports - not needed for standalone worker implementation


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("NX_PPO_LOGGING_LEVEL", "WARN"))
from typing import Tuple

from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        raise ValueError(
            "HSDP is not supported yet because it produces incorrect results for now. Please set fsdp_size=-1"
        )
        assert world_size % fsdp_size == 0
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(
            f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2"
        )
    return sharding_strategy


class ModelWorker(Worker):
    """
    This worker is a standalone model worker for the actor role.
    """

    def __init__(self, config: DictConfig, role: str = "actor", reward_fn=None):
        super().__init__()

        self.config = config
        self.apply_transformers_monkey_patch = (
            self.config.get("ulysses_sequence_parallel_size", 1) > 1
        )
        self.reward_fn = reward_fn
        self.role = role

        import torch.distributed

        if not torch.distributed.is_initialized():
            import datetime

            torch.distributed.init_process_group(
                backend="nccl", timeout=datetime.timedelta(seconds=1800)
            )

        self.enable_memory_tracing = self.config.get("enable_memory_tracing", False)
        if self.enable_memory_tracing:
            torch.cuda.memory._record_memory_history()

        # Enable expandable segments
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        print(f"role:{role}, world_size: {world_size}, {torch.distributed.GroupMember.WORLD}")

        self.device_mesh = create_device_mesh(
            world_size=world_size, fsdp_size=self.config.fsdp_config.fsdp_size
        )

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.use_ring_sequence_parallel = self.config.get("use_ring_sequence_parallel", False)
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # Configure offload settings
        self._is_offload_param = self.config.fsdp_config.get("param_offload", False)
        self._is_offload_grad = self.config.fsdp_config.get("grad_offload", False)
        self._is_offload_optimizer = self.config.fsdp_config.get("optimizer_offload", False)

        # Normalize config for actor
        self.config.ppo_mini_batch_size *= self.config.rollout.n
        dp_size = self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size
        self.config.ppo_mini_batch_size //= dp_size
        error_str = (
            f"is zero, you need increase batch size so that is is divisible by DP size({dp_size})"
        )
        assert self.config.ppo_mini_batch_size != 0, f"ppo_mini_batch_size {error_str}"

        # micro bsz
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= dp_size
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            assert self.config.ppo_micro_batch_size_per_gpu is not None
            assert (
                self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0
            ), f"ppo_mini_batch_size {self.config.ppo_mini_batch_size} is not divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            assert (
                self.config.ppo_micro_batch_size_per_gpu != 0
            ), f"ppo_micro_batch_size_per_gpu {error_str}"

        # Keep rollout config reads for compatibility
        if hasattr(self.config, "rollout") and hasattr(
            self.config.rollout, "log_prob_micro_batch_size"
        ):
            if self.config.rollout.log_prob_micro_batch_size is not None:
                self.config.rollout.log_prob_micro_batch_size //= dp_size
                self.config.rollout.log_prob_micro_batch_size_per_gpu = (
                    self.config.rollout.log_prob_micro_batch_size
                )
                assert (
                    self.config.rollout.log_prob_micro_batch_size_per_gpu != 0
                ), f"log_prob_micro_batch_size_per_gpu {error_str}"

        # Keep ref config reads for compatibility
        if hasattr(self.config, "ref") and hasattr(self.config.ref, "log_prob_micro_batch_size"):
            if self.config.ref.log_prob_micro_batch_size is not None:
                self.config.ref.log_prob_micro_batch_size //= dp_size
                self.config.ref.log_prob_micro_batch_size_per_gpu = (
                    self.config.ref.log_prob_micro_batch_size
                )
                assert (
                    self.config.ref.log_prob_micro_batch_size_per_gpu != 0
                ), f"log_prob_micro_batch_size_per_gpu {error_str}"

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        rollout_standalone=False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        from ..utils.dist_utils import (
            get_fsdp_wrap_policy,
            get_init_weight_context_manager,
            init_fn,
        )
        from ..utils.model_utils import (
            PrecisionType,
            copy_local_path_from_hdfs,
            get_generation_config,
            hf_tokenizer,
            print_model_size,
            update_model_config,
        )

        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32  # Actor uses float32 by default
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )

        self.generation_config = get_generation_config(
            local_path, trust_remote_code=trust_remote_code
        )

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                actor_module_class = AutoModelForCausalLM

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if use_remove_padding or self.ulysses_sequence_parallel_size > 1:
                from ..utils.monkey_patch import apply_monkey_patch

                if self.use_ring_sequence_parallel:
                    apply_monkey_patch(model=actor_module, use_ring_sequence_parallel=True)
                else:
                    apply_monkey_patch(model=actor_module, use_ring_sequence_parallel=False)

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            # some parameters may not in torch_dtype
            actor_module.to(torch_dtype)

            # if torch.distributed.get_rank() == 0:
            #     torch.save(actor_module.model.state_dict(), "fsdp_init_load.pkl")

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module, config=fsdp_config.get("wrap_policy", None)
        )

        print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None

        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=True,
            backward_prefetch=torch.distributed.fsdp.BackwardPrefetch.BACKWARD_PRE,
        )

        # Create optimizer and scheduler for actor
        from ..utils.model_utils import get_constant_schedule_with_warmup

        actor_optimizer = optim.AdamW(
            actor_module_fsdp.parameters(),
            lr=optim_config.lr,
            betas=optim_config.get("betas", (0.9, 0.999)),
            weight_decay=optim_config.get("weight_decay", 1e-2),
        )
        total_steps = optim_config.get("total_training_steps", 0)
        num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        actor_lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
        )

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def init_model(self):
        from ..utils.model_utils import FlopsCounter, get_checkpoint_manager, import_external_libs
        from .fsdp_actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )

        use_remove_padding = self.config.model.get("use_remove_padding", False)

        optim_config = self.config.optim
        fsdp_config = self.config.fsdp_config

        # Build actor model
        (
            self.actor_module_fsdp,
            self.actor_optimizer,
            self.actor_lr_scheduler,
            self.actor_model_config,
        ) = self._build_model_optimizer(
            model_path=self.config.model.path,
            fsdp_config=fsdp_config,
            optim_config=optim_config,
            override_model_config=override_model_config,
            use_remove_padding=use_remove_padding,
            enable_gradient_checkpointing=self.config.model.get(
                "enable_gradient_checkpointing", False
            ),
            trust_remote_code=self.config.model.get("trust_remote_code", False),
            use_liger=self.config.model.get("use_liger", False),
            rollout_standalone=False,
        )

        # get the original unwrapped module
        self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

        # Offloading removed - use torch.cuda.empty_cache() if needed
        if self._is_offload_param or self._is_offload_optimizer:
            torch.cuda.empty_cache()

        # Initialize actor
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.use_remove_padding = use_remove_padding
        self.actor = DataParallelPPOActor(
            config=self.config,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
            role="actor",
        )

        self.flops_counter = FlopsCounter(self.actor_model_config)
        self.checkpoint_manager = get_checkpoint_manager(self.config.checkpoint_manager)(
            model=self.actor_module_fsdp,
            optimizer=self.actor.actor_optimizer,
            lr_scheduler=self.actor_lr_scheduler,
            tokenizer=self.tokenizer,
            use_weight_provider=self.config.rollout.use_weight_provider,
            rollout_tp=self.config.rollout.tensor_model_parallel_size,
        )

        torch.cuda.empty_cache()

    def update_actor(self, data: DataProto):
        data = data.to("cuda")
        # Offloading removed - model/optimizer should remain on GPU

        data.batch = data.batch.cuda()

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            from ..utils.core_utils import Timer

            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time
            )
            metrics["mfu/actor"] = (
                estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
            )

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr

            output = DataProto(meta_info={"metrics": metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)

            output = output.to("cpu")

        # Offloading removed
        torch.cuda.empty_cache()
        # Profiler removed
        return output

    def compute_log_prob(self, data: DataProto):
        # Offloading removed - model should remain on GPU
        data = data.to("cuda")
        # Set meta info for actor-only compute_log_prob
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.ppo_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            temperature = (
                self.config.rollout.temperature
                if hasattr(self.config, "rollout") and hasattr(self.config.rollout, "temperature")
                else 1.0
            )
            output = DataProto.from_dict(
                tensors={"old_log_probs": output}, meta_info={"temperature": temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)

        # Offloading removed
        # clear kv cache
        torch.cuda.empty_cache()

        return output

    def update_actor_with_distillation(self, data: DataProto):
        """Update actor using on-policy distillation with reverse KL loss"""
        print(
            f"[fsdp_workers.update_actor_with_distillation] Starting, data batch_size={len(data)}"
        )
        data = data.to("cuda")
        data.batch = data.batch.cuda()
        print(f"[fsdp_workers.update_actor_with_distillation] Data moved to cuda")

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            print(
                f"[fsdp_workers.update_actor_with_distillation] Data preprocessed, calling actor.update_policy_with_distillation"
            )

            # Perform distillation training
            from ..utils.core_utils import Timer

            with Timer(name="update_policy_with_distillation", logger=None) as timer:
                metrics = self.actor.update_policy_with_distillation(data=data)
            delta_time = timer.last
            print(
                f"[fsdp_workers.update_actor_with_distillation] Training completed in {delta_time:.2f}s"
            )

            global_num_tokens = data.meta_info.get(
                "global_token_num", len(data) * 512
            )  # Rough estimate
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time
            )
            distillation_epochs = data.meta_info.get("distillation_epochs", self.config.ppo_epochs)
            metrics["mfu/actor"] = (
                estimated_flops * distillation_epochs / promised_flops / self.world_size
            )

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["distill/lr"] = lr

            output = DataProto(meta_info={"metrics": metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")
            print(f"[fsdp_workers.update_actor_with_distillation] Completed, returning metrics")

        # Offloading removed
        torch.cuda.empty_cache()
        return output

    def load_checkpoint(self, path, del_local_after_load=True, load_weight_only=False):
        # Offloading removed - model should remain on GPU
        self.checkpoint_manager.load_checkpoint(
            path=path, del_local_after_load=del_local_after_load, load_weight_only=load_weight_only
        )

    def save_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        global_step=0,
        saved_fully_shared_ckpt=True,
        save_weight_only=False,
        remove_previous_ckpt=True,
    ):
        import torch

        # Offloading removed - model should remain on GPU
        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            saved_fully_shared_ckpt=saved_fully_shared_ckpt,
            save_weight_only=save_weight_only,
            remove_previous_ckpt=remove_previous_ckpt,
        )

        torch.distributed.barrier()

    def convert_ckpt_to_huggingface(self, local_path):
        # It is only used for checkpoint convert!
        with FSDP.state_dict_type(
            self.actor.actor_module,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = self.actor.actor_module.state_dict()
            if torch.distributed.get_rank() == 0:
                self.actor_module.save_pretrained(save_directory=local_path, state_dict=state_dict)
                self.tokenizer.save_pretrained(local_path)

    def barrier(self):
        torch.distributed.barrier()

    def destroy(self):
        """Minimal cleanup before process termination

        We do NOTHING except clear CUDA cache. Calling destroy_process_group() or
        deleting objects causes segfaults because FSDP modules hold NCCL communicator
        references that become invalid. The OS will clean up everything on process exit.
        """
        logger.info(f"Rank {self.rank}: Preparing for graceful shutdown...")

        try:
            # Only clear CUDA cache to free GPU memory quickly
            torch.cuda.empty_cache()
            logger.info(f"Rank {self.rank}: CUDA cache cleared, ready for process exit")
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Error clearing CUDA cache: {e}")
