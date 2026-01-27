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
Data Dumper for collecting training debug data.

This module provides a unified interface for dumping training data including:

Self-Hosted Mode:
- Trajectories (training inputs)
- Old log probabilities (from GRPO preparation)
- Forward data (logprobs, entropy from forward pass)
- Loss data (loss, pg_loss, entropy_loss from backward pass)
- Param data (parameter values after optimizer step)
- Gradient data (gradients after backward pass)

Remote API Mode:
- Prepared trajectories (after GRPO advantage computation)
- Datums (converted data for Weaver service)
- Training metrics (returned from Weaver forward_backward)
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class DataDumper:
    """
    Thread-safe data dumper for training debug data.

    Supports both Trainer-side (single process) and Worker-side (multi-process) dumping.
    Files are organized by step and worker rank.
    """

    _instances: dict[str, "DataDumper"] = {}
    _lock = threading.Lock()

    def __init__(self, config: DictConfig | dict | None = None, rank: int = 0):
        """
        Initialize the DataDumper.

        Args:
            config: Configuration containing debug settings
            rank: Worker rank (0 for Trainer-side, actual rank for Worker-side)
        """
        self._rank = rank
        self._write_lock = threading.Lock()

        # Extract debug config
        if config is None:
            self._enabled = False
            self._dump_dir = None
            self._options: dict[str, bool] = {}
            self._format = "pt"
            self._dump_every_n = 1
            return

        # Handle both DictConfig and dict.
        # NOTE: In NexRL, modules like Trainer are often instantiated with a *sub-config*
        # (e.g., config.trainer). In that case, `debug` lives at the root level and is
        # accessible via OmegaConf parent chain. We try to resolve that here so callers
        # don't need to re-plumb root config everywhere.
        debug_config: Any = {}
        if isinstance(config, DictConfig):
            debug_config = config.get("debug", {}) or {}
            if not debug_config:
                parent = getattr(config, "_parent", None)
                if parent is not None:
                    try:
                        debug_config = parent.get("debug", {}) or {}
                    except Exception:  # pylint: disable=broad-exception-caught
                        debug_config = {}
        else:
            debug_config = config.get("debug", {}) or {}

        # Resolve interpolations if any (e.g., ${oc.env:EXPERIMENT_PATH,...})
        if isinstance(debug_config, DictConfig):
            debug_config = OmegaConf.to_container(debug_config, resolve=True) or {}

        self._enabled = debug_config.get("enable_data_dump", False)

        if not self._enabled:
            self._dump_dir = None
            self._options = {}
            self._format = "pt"
            self._dump_every_n = 1
            return

        # Set up dump directory
        dump_dir = debug_config.get("dump_dir", "debug_dump")
        self._dump_dir = Path(dump_dir)
        self._dump_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for both self_hosted and remote_api modes
        subdirs = [
            # Self-hosted mode
            "trajectory",
            "old_log_probs",
            "forward_data",
            "loss",
            "param",
            "gradient",
            # Remote API mode
            "prepared_trajectories",
            "datums",
            "training_metrics",
        ]
        for subdir in subdirs:
            (self._dump_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Remote API specific options
        self._remote_api_options = debug_config.get("remote_api_dump_options", {})

        # Target param pattern for param/gradient dump
        self._target_param_pattern = debug_config.get("target_param_pattern", "")

        self._options = debug_config.get("dump_options", {})
        self._format = debug_config.get("dump_format", "pt")
        self._dump_every_n = debug_config.get("dump_every_n_steps", 1)

        logger.info(
            f"DataDumper initialized: enabled={self._enabled}, "
            f"dump_dir={self._dump_dir}, format={self._format}, "
            f"dump_every_n={self._dump_every_n}, rank={self._rank}"
        )

    @classmethod
    def get_instance(
        cls, config: DictConfig | dict | None = None, rank: int = 0, key: str = "default"
    ) -> "DataDumper":
        """
        Get or create a DataDumper instance (singleton per key).

        Args:
            config: Configuration containing debug settings
            rank: Worker rank
            key: Instance key for different contexts

        Returns:
            DataDumper instance
        """
        instance_key = f"{key}_{rank}"
        with cls._lock:
            if instance_key not in cls._instances:
                cls._instances[instance_key] = cls(config, rank)
            return cls._instances[instance_key]

    @classmethod
    def reset_instances(cls):
        """Reset all instances (for testing)."""
        with cls._lock:
            cls._instances.clear()

    @property
    def enabled(self) -> bool:
        """Check if dumping is enabled."""
        return self._enabled

    def should_dump(self, data_type: str, step: int) -> bool:
        """
        Check if data should be dumped for the given type and step.

        Args:
            data_type: Type of data
                Self-hosted: "trajectory", "old_log_probs", "forward_data", "loss", "param", "gradient"
                Remote API: "prepared_trajectories", "datums", "training_metrics"
            step: Current training step

        Returns:
            True if data should be dumped
        """
        if not self._enabled:
            return False
        if step % self._dump_every_n != 0:
            return False

        # Check self_hosted options first
        if self._options.get(data_type, False):
            return True

        # Check remote_api options
        remote_api_options = getattr(self, "_remote_api_options", {})
        return remote_api_options.get(data_type, False)

    def dump_trajectory(self, step: int, trajectories: list) -> None:
        """
        Dump trajectories to file.

        Args:
            step: Current training step
            trajectories: List of Trajectory objects
        """
        if not self.should_dump("trajectory", step):
            return

        data = {
            "step": step,
            "num_trajectories": len(trajectories),
            "trajectories": [self._serialize_trajectory(t) for t in trajectories],
        }

        self._write_file("trajectory", f"step_{step:06d}", data)
        logger.info(f"Dumped {len(trajectories)} trajectories for step {step}")

    def dump_old_log_probs(
        self,
        step: int,
        old_log_probs: torch.Tensor,
        batch_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Dump old log probabilities from GRPO preparation.

        Args:
            step: Current training step
            old_log_probs: Old log probabilities tensor
            batch_info: Optional additional batch information
        """
        if not self.should_dump("old_log_probs", step):
            return

        data = {
            "step": step,
            "old_log_probs": old_log_probs.detach().cpu(),
            "shape": list(old_log_probs.shape),
        }
        if batch_info:
            data["batch_info"] = batch_info

        self._write_file("old_log_probs", f"step_{step:06d}", data)
        logger.info(f"Dumped old_log_probs for step {step}, shape={old_log_probs.shape}")

    def dump_forward_data(
        self,
        step: int,
        micro_idx: int,
        log_probs: torch.Tensor,
        entropy: torch.Tensor,
        response_mask: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Dump forward pass data (logprobs, entropy).

        Args:
            step: Current training step
            micro_idx: Micro-batch index
            log_probs: Log probabilities tensor
            entropy: Entropy tensor
            response_mask: Response mask tensor
            input_ids: Optional input IDs tensor
            extra_info: Optional additional information
        """
        if not self.should_dump("forward_data", step):
            return

        data = {
            "step": step,
            "rank": self._rank,
            "micro_idx": micro_idx,
            "log_probs": log_probs.detach().cpu(),
            "entropy": entropy.detach().cpu(),
            "response_mask": response_mask.detach().cpu(),
            "log_probs_shape": list(log_probs.shape),
            "entropy_shape": list(entropy.shape),
        }
        if input_ids is not None:
            data["input_ids"] = input_ids.detach().cpu()
        if extra_info:
            data["extra_info"] = extra_info

        filename = f"step_{step:06d}_rank{self._rank}_micro{micro_idx}"
        self._write_file("forward_data", filename, data)
        logger.debug(f"Dumped forward_data for step {step}, rank {self._rank}, micro {micro_idx}")

    def dump_loss(
        self,
        step: int,
        micro_idx: int,
        loss: float,
        pg_loss: float,
        entropy_loss: float,
        metrics: dict[str, Any] | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Dump loss data from backward pass.

        Args:
            step: Current training step
            micro_idx: Micro-batch index
            loss: Total loss value
            pg_loss: Policy gradient loss value
            entropy_loss: Entropy loss value
            metrics: Optional loss metrics dictionary
            extra_info: Optional additional information
        """
        if not self.should_dump("loss", step):
            return

        data: dict[str, Any] = {
            "step": step,
            "rank": self._rank,
            "micro_idx": micro_idx,
            "loss": float(loss),
            "pg_loss": float(pg_loss),
            "entropy_loss": float(entropy_loss),
        }
        if metrics:
            # Convert any tensor values to Python types
            data["metrics"] = {
                k: float(v) if isinstance(v, (torch.Tensor, float)) else v
                for k, v in metrics.items()
            }
        if extra_info:
            data["extra_info"] = extra_info

        filename = f"step_{step:06d}_rank{self._rank}_micro{micro_idx}"
        self._write_file("loss", filename, data)
        logger.debug(
            f"Dumped loss for step {step}, rank {self._rank}, micro {micro_idx}: "
            f"loss={loss:.6f}, pg_loss={pg_loss:.6f}, entropy_loss={entropy_loss:.6f}"
        )

    def dump_param(
        self,
        step: int,
        param_name: str,
        param: torch.Tensor,
    ) -> None:
        """
        Dump parameter after optimizer step.

        Args:
            step: Current training step
            param_name: Name of the parameter
            param: Parameter tensor (after optimizer step)
        """
        if not self.should_dump("param", step):
            return

        param_cpu = param.detach().cpu().float()
        data = {
            "step": step,
            "rank": self._rank,
            "param_name": param_name,
            "param": param.detach().cpu(),
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "norm": float(param_cpu.norm().item()),
            "mean": float(param_cpu.mean().item()),
            "std": float(param_cpu.std().item()),
        }

        filename = f"step_{step:06d}_rank{self._rank}"
        self._write_file("param", filename, data)
        logger.debug(f"Dumped param {param_name} for step {step}, rank {self._rank}")

    def dump_gradient(
        self,
        step: int,
        param_name: str,
        gradient: torch.Tensor,
    ) -> None:
        """
        Dump gradient after backward pass.

        Args:
            step: Current training step
            param_name: Name of the parameter
            gradient: Gradient tensor (after backward)
        """
        if not self.should_dump("gradient", step):
            return

        grad_cpu = gradient.detach().cpu().float()
        data = {
            "step": step,
            "rank": self._rank,
            "param_name": param_name,
            "gradient": gradient.detach().cpu(),
            "shape": list(gradient.shape),
            "dtype": str(gradient.dtype),
            "norm": float(grad_cpu.norm().item()),
            "mean": float(grad_cpu.mean().item()),
            "std": float(grad_cpu.std().item()),
        }

        filename = f"step_{step:06d}_rank{self._rank}"
        self._write_file("gradient", filename, data)
        logger.debug(f"Dumped gradient {param_name} for step {step}, rank {self._rank}")

    @property
    def target_param_pattern(self) -> str:
        """Get target param pattern for param/gradient dump."""
        return getattr(self, "_target_param_pattern", "")

    # =========================================================================
    # Remote API Mode Dump Methods
    # =========================================================================

    def dump_prepared_trajectories(
        self,
        step: int,
        trajectories: list,
        advantages: list[float] | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Dump trajectories after GRPO advantage computation (remote_api mode).

        Args:
            step: Current training step
            trajectories: List of Trajectory objects with computed advantages
            advantages: Optional list of computed advantages
            extra_info: Optional additional information
        """
        if not self.should_dump("prepared_trajectories", step):
            return

        data = {
            "step": step,
            "num_trajectories": len(trajectories),
            "trajectories": [self._serialize_trajectory(t) for t in trajectories],
        }
        if advantages is not None:
            data["advantages"] = advantages
        if extra_info:
            data["extra_info"] = extra_info

        self._write_file("prepared_trajectories", f"step_{step:06d}", data)
        logger.info(f"Dumped {len(trajectories)} prepared trajectories for step {step}")

    def dump_datums(
        self,
        step: int,
        datums_data: list[dict[str, Any]],
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Dump converted datums for Weaver service (remote_api mode).

        Args:
            step: Current training step
            datums_data: List of datum dictionaries to be sent to Weaver
            extra_info: Optional additional information
        """
        if not self.should_dump("datums", step):
            return

        data = {
            "step": step,
            "num_datums": len(datums_data),
            "datums": [self._serialize_datum(d) for d in datums_data],
        }
        if extra_info:
            data["extra_info"] = extra_info

        self._write_file("datums", f"step_{step:06d}", data)
        logger.info(f"Dumped {len(datums_data)} datums for step {step}")

    def dump_training_metrics(
        self,
        step: int,
        metrics: dict[str, Any],
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Dump training metrics from Weaver forward_backward response (remote_api mode).

        Args:
            step: Current training step
            metrics: Metrics dictionary returned from Weaver
            extra_info: Optional additional information
        """
        if not self.should_dump("training_metrics", step):
            return

        data = {
            "step": step,
            "metrics": self._convert_metrics(metrics),
        }
        if extra_info:
            data["extra_info"] = extra_info

        self._write_file("training_metrics", f"step_{step:06d}", data)
        logger.info(f"Dumped training metrics for step {step}: {list(metrics.keys())}")

    def _serialize_datum(self, datum: dict[str, Any]) -> dict[str, Any]:
        """
        Serialize a datum dictionary, converting tensors to serializable format.

        Args:
            datum: Datum dictionary

        Returns:
            Serialized dictionary
        """
        result: dict[str, Any] = {}
        for k, v in datum.items():
            if isinstance(v, dict):
                result[k] = self._serialize_datum(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [self._to_serializable(x) for x in v]
            else:
                result[k] = self._to_serializable(v)
        return result

    def _convert_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Convert metrics dictionary to serializable format.

        Args:
            metrics: Metrics dictionary

        Returns:
            Serializable metrics dictionary
        """
        result: dict[str, Any] = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 0:
                    result[k] = float(v.item())
                else:
                    result[k] = v.detach().cpu().tolist()
            elif isinstance(v, dict):
                result[k] = self._convert_metrics(v)
            elif isinstance(v, (float, int, str, bool, type(None))):
                result[k] = v
            else:
                try:
                    result[k] = float(v)
                except (TypeError, ValueError):
                    result[k] = str(v)
        return result

    def _write_file(self, subdir: str, filename: str, data: dict[str, Any]) -> None:
        """
        Write data to file in the specified format.

        Args:
            subdir: Subdirectory name
            filename: Base filename (without extension)
            data: Data dictionary to write
        """
        if self._dump_dir is None:
            return

        with self._write_lock:
            if self._format == "pt":
                filepath = self._dump_dir / subdir / f"{filename}.pt"
                torch.save(data, filepath)
            elif self._format == "jsonl":
                filepath = self._dump_dir / subdir / f"{filename}.jsonl"
                # Convert tensors to lists for JSON serialization
                json_data = self._to_json_serializable(data)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(json.dumps(json_data) + "\n")
            else:
                raise ValueError(f"Unsupported dump format: {self._format}")

    def _serialize_trajectory(self, traj) -> dict[str, Any]:
        """
        Serialize a Trajectory object to a dictionary.

        Args:
            traj: Trajectory object

        Returns:
            Serialized dictionary
        """
        result = {
            "tokens": self._to_serializable(traj.tokens),
            "loss_mask": self._to_serializable(traj.loss_mask),
            "reward": float(traj.reward),
            "is_val": bool(traj.is_val),
        }
        # Add extra fields
        if hasattr(traj, "extra_fields") and traj.extra_fields:
            result["extra_fields"] = {
                k: self._to_serializable(v) for k, v in traj.extra_fields.items()
            }
        return result

    def _to_serializable(self, v: Any) -> Any:
        """
        Convert a value to a serializable format.

        Args:
            v: Value to convert

        Returns:
            Serializable value
        """
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().tolist()
        if hasattr(v, "item"):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [self._to_serializable(x) for x in v]
        if isinstance(v, dict):
            return {k: self._to_serializable(val) for k, val in v.items()}
        return v

    def _to_json_serializable(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Convert all values in a dictionary to JSON-serializable format.

        Args:
            data: Dictionary with potentially non-serializable values

        Returns:
            Dictionary with JSON-serializable values
        """
        result = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.tolist()
            elif isinstance(v, dict):
                result[k] = self._to_json_serializable(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [x.tolist() if isinstance(x, torch.Tensor) else x for x in v]
            else:
                result[k] = v
        return result


def get_data_dumper(
    config: DictConfig | dict | None = None, rank: int = 0, key: str = "default"
) -> DataDumper:
    """
    Get or create a DataDumper instance.

    Convenience function for getting a DataDumper instance.

    Args:
        config: Configuration containing debug settings
        rank: Worker rank
        key: Instance key for different contexts

    Returns:
        DataDumper instance
    """
    return DataDumper.get_instance(config, rank, key)
