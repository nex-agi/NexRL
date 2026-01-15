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
SelfHostedGrpoTrainer - Self-hosted trainer with integrated GRPO algorithm
"""

import logging
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig

from ..algorithm import core_algos
from ..nexrl_types import Batch
from .self_hosted_trainer import SelfHostedTrainer

logger = logging.getLogger(__name__)


class SelfHostedGrpoTrainer(SelfHostedTrainer):
    """
    Self-hosted trainer with integrated GRPO (Group Relative Policy Optimization) algorithm.

    Extends SelfHostedTrainer with GRPO-specific trajectory processing:
    1. Converts trajectories to batch format (inherited)
    2. Applies GRPO algorithm (advantage computation, KL penalty, etc.)
    3. Sends the processed batch to the train service (inherited)
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the self-hosted GRPO trainer

        Args:
            config: Configuration dictionary with training settings and GRPO parameters
        """
        super().__init__(config)

        # GRPO algorithm configuration
        self._algorithm_config = config.algorithm
        self._do_old_log_prob_compute = self._algorithm_config.get("do_old_log_prob_compute", True)
        self._use_kl_in_reward = self._algorithm_config.get("use_kl_in_reward", False)
        self._pad_token_id = self.tokenizer.pad_token_id

        # Initialize KL controller for reward computation
        if self._use_kl_in_reward:
            self._kl_ctrl_in_reward = self._init_kl_controller(self._algorithm_config)
        else:
            self._kl_ctrl_in_reward = None

        logger.info("SelfHostedGrpoTrainer initialized")

    # ========================================================================
    # Initialization Methods
    # ========================================================================

    def _init_kl_controller(self, config: DictConfig):
        """
        Initialize KL controller based on configuration

        Args:
            config: Configuration dictionary

        Returns:
            KL controller instance (AdaptiveKLController or FixedKLController)
        """
        if not hasattr(config, "critic") or not hasattr(config.critic, "kl_ctrl"):
            # Default to adaptive controller
            return core_algos.AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0, horizon=10000)
        kl_config = config.critic.kl_ctrl
        if kl_config.type == "fixed":
            return core_algos.FixedKLController(kl_reward_coef=kl_config.kl_reward_coef)
        elif kl_config.type == "adaptive":
            return core_algos.AdaptiveKLController(
                init_kl_coef=kl_config.kl_reward_coef,
                target_kl=kl_config.target_kl,
                horizon=kl_config.horizon,
            )
        else:
            raise NotImplementedError(f"KL controller type {kl_config.type} not implemented")

    # ========================================================================
    # Batch Preparation (Override from SelfHostedTrainer)
    # ========================================================================

    def _prepare_batch(self, batch: Batch) -> tuple[Batch, dict]:
        """
        Prepare batch using GRPO (Group Relative Policy Optimization) algorithm.

        Implements the abstract _prepare_batch method with GRPO-specific processing:
        1. Log rollout metrics
        2. Remove redundant padding
        3. Recompute old log probabilities
        4. Compute token-level rewards with KL penalty
        5. Calculate GRPO advantages
        6. Compute and log metrics

        Args:
            batch: Batch of trajectory data from rollout

        Returns:
            Tuple of (prepared_batch, metrics_dict)
        """
        logger.info("Begin GRPO batch preparation")

        # Step 0: Log rollout metrics
        self._log_rollout_metrics(batch)

        # Step 1: Prepare batch for training
        batch.metadata["global_token_num"] = torch.sum(
            batch.values["attention_mask"] * batch.values["loss_mask"], dim=-1
        ).tolist()
        if "loss_mask" in batch.values:
            batch.values["scoring_attention_mask"] = (
                batch.values["attention_mask"] * batch.values["loss_mask"]
            )
        else:
            batch.values["scoring_attention_mask"] = batch.values["attention_mask"].clone()

        batch = Batch.remove_redundant_left_padding(
            batch,
            pad_token_id=self._pad_token_id,
            anchor_field="input_ids",
            fields=[
                "input_ids",
                "prompts",
                "attention_mask",
                "position_ids",
                "scoring_attention_mask",
                "loss_mask",
            ],
        )

        batch = Batch.remove_redundant_right_padding(
            batch,
            pad_token_id=self._pad_token_id,
            anchor_field="input_ids",
            fields=[
                "input_ids",
                "responses",
                "attention_mask",
                "position_ids",
                "scoring_attention_mask",
                "loss_mask",
            ],
        )

        # Step 2: Recompute old logprobs
        if self._do_old_log_prob_compute:
            old_log_probs = self._compute_old_log_probs(batch)
            batch.values["old_log_probs"] = old_log_probs
        else:
            bsz = batch.metadata["batch_size"]
            old_log_probs = torch.tensor([[0.0]] * bsz, dtype=torch.float32)
            batch.values["old_log_probs"] = old_log_probs

        # Step 3: Calculate advantages
        metrics = {}
        # Compute token-level scores using reward function
        if "token_level_scores" not in batch.values:
            reward_tensor = self._reward_fn(batch)
            batch.values["token_level_scores"] = reward_tensor

        # Apply KL penalty to get token-level rewards
        if self._use_kl_in_reward:
            kl_penalty_type = getattr(self._algorithm_config, "kl_penalty", "kl")
            batch, kl_metrics = self._apply_kl_penalty(
                batch,
                kl_ctrl=self._kl_ctrl_in_reward,
                kl_penalty=kl_penalty_type,
            )
            metrics.update(kl_metrics)
        else:
            batch.values["token_level_rewards"] = batch.values["token_level_scores"]
            kl_metrics = self._cal_reward_kl(batch)
            metrics.update(kl_metrics)

        # Compute GRPO advantages
        batch = self._compute_advantage(batch)

        # Add GRPO std metrics
        if "grpo_std" in batch.metadata:
            grpo_std_lst = list(batch.metadata["grpo_std"].values())
            metrics["critic/grpo_std/mean"] = np.mean(grpo_std_lst)
            metrics["critic/grpo_std/min"] = float(min(grpo_std_lst))
            metrics["critic/grpo_std/max"] = float(max(grpo_std_lst))
            metrics["critic/grpo_num"] = len(grpo_std_lst)

        metrics.update(self._compute_data_metrics(batch, use_critic=False))
        self._activity_tracker.experiment_logger_post(backend="wandb", data=metrics)

        logger.info("GRPO batch preparation completed")

        return batch, metrics

    def _compute_old_log_probs(self, batch: Batch) -> torch.Tensor:
        """
        Recompute old log probs using the train service client.

        Args:
            batch: Batch to compute log probs for

        Returns:
            Old log probabilities tensor
        """
        with self._train_service_client.actor_context():
            ret = self._train_service_client.compute_log_prob(batch.to_nextrainer_batch())
        old_log_probs: torch.Tensor = ret["batch"]["old_log_probs"]
        return old_log_probs

    def _reward_fn(self, batch: Batch) -> torch.Tensor:
        """
        Reward function that computes token-level scores.

        Assigns the reward score to the last valid token position in each response.
        This follows the pattern where the reward is given only at the end of the response.

        Args:
            batch: Batch containing responses, attention_mask/loss_mask, and scores

        Returns:
            reward_tensor: Token-level reward tensor with same shape as responses
        """
        # Get responses and initialize reward tensor
        responses = batch.values["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)

        logger.info(f"Reward tensor shape: {reward_tensor.shape}")

        batch_size = len(batch)
        response_length = responses.size(-1)

        # If loss_mask is available, use it instead (for multi-turn environments)
        if "loss_mask" in batch.values:
            loss_mask = batch.values["loss_mask"]
            response_mask = loss_mask[:, -response_length:]
        else:
            attention_mask = batch.values["attention_mask"]
            response_mask = attention_mask[:, -response_length:]

        # Get scores from batch (trajectories store reward as 'reward')
        if "reward" in batch.values:
            scores = batch.values["reward"]
        else:
            logger.warning("'reward' field not found in batch.values, using zeros")
            scores = [0.0] * batch_size

        # Process each item in the batch
        for i in range(batch_size):
            # Extract score for this item
            if isinstance(scores, torch.Tensor):
                score = scores[i].item() if scores.dim() > 0 else scores.item()
            else:
                score = scores[i]

            # Get the response mask for this item
            mask = response_mask[i]

            # Find the last valid token position
            last_valid_pos = torch.where(mask > 0)[0]

            if len(last_valid_pos) == 0:
                logger.warning(f"No valid tokens in response for item {i}, skipping")
                continue

            last_valid_token_pos = last_valid_pos[-1].item()

            # Assign the score to the last valid token position
            reward_tensor[i, last_valid_token_pos] = score

        return reward_tensor

    def _apply_kl_penalty(
        self, batch: Batch, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty: str = "kl"
    ) -> tuple[Batch, Dict]:
        """
        Apply KL penalty to token-level scores to get token-level rewards

        Args:
            batch: Batch containing scores and log probabilities
            kl_ctrl: KL controller instance
            kl_penalty: Type of KL penalty

        Returns:
            Updated batch and metrics dictionary
        """
        responses = batch.values["responses"]
        response_length = responses.size(-1)
        token_level_scores = batch.values["token_level_scores"]
        batch_size = responses.shape[0]
        attention_mask = batch.values["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

        # Compute KL between ref_policy and current policy
        if "ref_log_prob" in batch.values:
            kld = core_algos.kl_penalty(
                batch.values["old_log_probs"],
                batch.values["ref_log_prob"],
                kl_penalty_type=kl_penalty,
            )
            kld = kld * response_mask
            beta = kl_ctrl.value
        else:
            beta = 0
            kld = torch.zeros_like(response_mask, dtype=torch.float32)

        token_level_rewards = token_level_scores - beta * kld

        current_kl = core_algos.masked_mean(kld, mask=response_mask, axis=-1)
        current_kl = torch.mean(current_kl).item()

        kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
        batch.values["token_level_rewards"] = token_level_rewards

        metrics = {
            "critic/kl": current_kl,
            "critic/kl_coeff": beta,
            "actor/reward_kl_penalty": current_kl,
            "actor/reward_kl_penalty_coeff": beta,
        }

        return batch, metrics

    def _cal_reward_kl(self, batch: Batch) -> Dict:
        """
        Calculate reward KL metrics when KL penalty is not used

        Args:
            batch: Batch containing responses and attention mask

        Returns:
            Metrics dictionary
        """
        beta = 0
        responses = batch.values["responses"]
        response_length = responses.size(-1)
        attention_mask = batch.values["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

        kld = torch.zeros_like(response_mask, dtype=torch.float32)
        current_kl = core_algos.masked_mean(kld, mask=response_mask, axis=-1)
        current_kl = torch.mean(current_kl).item()

        metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}

        return metrics

    def _compute_advantage(self, batch: Batch) -> Batch:
        """
        Compute GRPO advantages

        Args:
            batch: Batch containing rewards and other data

        Returns:
            Updated batch with advantages and returns
        """
        if "scoring_attention_mask" in batch.values:
            attention_mask = batch.values["scoring_attention_mask"]
        else:
            attention_mask = batch.values["attention_mask"]

        token_level_rewards = batch.values["token_level_rewards"]

        # Get group IDs for GRPO grouping
        if "uid" in batch.values:
            group_ids = batch.values["uid"]
        elif "group_id" in batch.values:
            group_ids = batch.values["group_id"]
        else:
            raise NotImplementedError("Batch must contain 'uid' or 'group_id'")

        responses = batch.values["responses"]
        response_length = responses.size(-1)
        response_mask = attention_mask[:, -response_length:]

        # Compute GRPO advantages
        index: Any = group_ids
        run_ids: Any = batch.values["run_id"]
        advantages, returns, id2std = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            run_ids=run_ids,
        )

        batch.values["advantages"] = advantages
        batch.values["returns"] = returns
        batch.metadata["grpo_std"] = id2std

        return batch

    # ========================================================================
    # Metrics and Logging Methods
    # ========================================================================

    def _log_rollout_metrics(self, batch: Batch):
        """
        Log rollout metrics from the batch to wandb.

        Args:
            batch: Batch containing rollout data including scores
        """
        if "score" not in batch.values:
            logger.warning(
                "'score' field not found in batch.values, skipping rollout metrics logging"
            )
            return

        scores = batch.values["score"]

        # Scores is a list of dicts, we need to compute mean for each key
        if not scores or len(scores) == 0:
            logger.warning("Empty scores list, skipping rollout metrics logging")
            return

        # Collect all keys from score dicts
        all_keys: set[str] = set()
        for score_dict in scores:
            if isinstance(score_dict, dict):
                all_keys.update(score_dict.keys())

        # Compute mean for each key
        metrics_data = {}
        for key in all_keys:
            values = []
            for score_dict in scores:
                if isinstance(score_dict, dict) and key in score_dict:
                    # Convert boolean to float if needed
                    val = score_dict[key]
                    if isinstance(val, bool):
                        val = float(val)
                    elif isinstance(val, (int, float)):
                        val = float(val)
                    else:
                        continue
                    values.append(val)

            if values:
                mean_value = np.mean(values)
                metrics_data[f"rollout/{key}"] = mean_value

        # Log to wandb if we have metrics
        if metrics_data:
            self._activity_tracker.experiment_logger_post(backend="wandb", data=metrics_data)

    @staticmethod
    def _compute_response_info(batch: Batch) -> dict:
        """
        Compute response length and mask information.

        Args:
            batch: Batch containing attention masks

        Returns:
            Dictionary with response_mask, prompt_length, and response_length
        """
        response_length = batch.values["responses"].shape[-1]

        prompt_mask = batch.values["attention_mask"][:, :-response_length]
        response_mask = batch.values["attention_mask"][:, -response_length:]

        prompt_length = prompt_mask.sum(-1).float()
        response_length = response_mask.sum(-1).float()  # (batch_size,)

        return {
            "response_mask": response_mask,
            "prompt_length": prompt_length,
            "response_length": response_length,
        }

    def _compute_data_metrics(self, batch: Batch, use_critic: bool = True) -> dict:
        """
        Compute comprehensive metrics for logging.

        Args:
            batch: Batch containing all computed values
            use_critic: Whether to include critic-specific metrics

        Returns:
            Dictionary of metrics
        """
        sequence_score = batch.values["token_level_scores"].sum(-1)
        sequence_reward = batch.values["token_level_rewards"].sum(-1)

        advantages = batch.values["advantages"]
        returns = batch.values["returns"]

        max_response_length = batch.values["responses"].shape[-1]

        prompt_mask = batch.values["attention_mask"][:, :-max_response_length].bool()
        response_mask = batch.values["attention_mask"][:, -max_response_length:].bool()

        max_prompt_length = prompt_mask.size(-1)

        response_info = self._compute_response_info(batch)
        prompt_length = response_info["prompt_length"]
        response_length = response_info["response_length"]

        valid_adv = torch.masked_select(advantages, response_mask)
        valid_returns = torch.masked_select(returns, response_mask)

        if use_critic:
            values = batch.values["values"]
            valid_values = torch.masked_select(values, response_mask)
            return_diff_var = torch.var(valid_returns - valid_values)
            return_var = torch.var(valid_returns)

        metrics = {
            # score
            "critic/score/mean": torch.mean(sequence_score).detach().item(),
            "critic/score/max": torch.max(sequence_score).detach().item(),
            "critic/score/min": torch.min(sequence_score).detach().item(),
            # accuracy
            "critic/accuracy/mean": torch.mean(torch.round(sequence_score)).detach().item(),
            "critic/accuracy/max": torch.max(torch.round(sequence_score)).detach().item(),
            "critic/accuracy/min": torch.min(torch.round(sequence_score)).detach().item(),
            # reward
            "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
            "critic/rewards/max": torch.max(sequence_reward).detach().item(),
            "critic/rewards/min": torch.min(sequence_reward).detach().item(),
            # adv
            "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
            "critic/advantages/max": torch.max(valid_adv).detach().item(),
            "critic/advantages/min": torch.min(valid_adv).detach().item(),
            # returns
            "critic/returns/mean": torch.mean(valid_returns).detach().item(),
            "critic/returns/max": torch.max(valid_returns).detach().item(),
            "critic/returns/min": torch.min(valid_returns).detach().item(),
            **(
                {
                    # values
                    "critic/values/mean": torch.mean(valid_values).detach().item(),
                    "critic/values/max": torch.max(valid_values).detach().item(),
                    "critic/values/min": torch.min(valid_values).detach().item(),
                    # vf explained var
                    # pylint: disable=possibly-used-before-assignment
                    "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5))
                    .detach()
                    .item(),
                }
                if use_critic
                else {}
            ),
            # response length
            "response_length/mean": torch.mean(response_length).detach().item(),
            "response_length/max": torch.max(response_length).detach().item(),
            "response_length/min": torch.min(response_length).detach().item(),
            "response_length/clip_ratio": torch.mean(
                torch.eq(response_length, max_response_length).float()
            )
            .detach()
            .item(),
            # prompt length
            "prompt_length/mean": torch.mean(prompt_length).detach().item(),
            "prompt_length/max": torch.max(prompt_length).detach().item(),
            "prompt_length/min": torch.min(prompt_length).detach().item(),
            "prompt_length/clip_ratio": torch.mean(
                torch.eq(prompt_length, max_prompt_length).float()
            )
            .detach()
            .item(),
        }

        if "average_agent_acc" in batch.metadata:
            for k, v in batch.metadata["average_agent_acc"].items():
                metrics["critic/average_agent_acc/" + k] = v
        return metrics
