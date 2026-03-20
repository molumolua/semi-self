# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Ray single-controller trainer for DAPO / semi-self (extends VERL RayPPOTrainer).

Model init is Hugging Face–centric and agnostic to specific model families.
"""

import os
import uuid
from collections import defaultdict
from typing import Optional
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.dataset.inmemory_dataset import InMemoryRLHFDataset
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask




                
class RayDAPOTrainer(RayPPOTrainer):
    """
    DAPO-style semi-self trainer: driver coordinates rollout, curriculum (`update_problems`), merge, and PPO.

    Intended for single-node setups where the driver process orchestrates worker groups.
    """

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Recompute `old_log_prob` (+ ref if enabled), entropy metric, and union onto `batch`."""
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def fit(self):
        """
        Main training loop: each step calls `train_batch` (rollout, curriculum update, merge, PPO update).

        The driver orchestrates workers via RPC; advantage and light-weight metrics run on the driver.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.tracking_logger = logger

        self.global_steps = 0
        self.gen_steps = 0


        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics:{val_metrics}")
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        


        for epoch in range(self.config.trainer.total_epochs):
            # only one batch for InMemoryRLHFDataset
            for batch_dict in self.train_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps
                batch,metrics=self.train_batch(
                    batch_dict,
                    prev_step_profile,
                    curr_step_profile,
                    timing_raw,
                )
                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing


                if "acc" in batch.non_tensor_batch:
                    acc_vals = np.asarray(batch.non_tensor_batch["acc"], dtype=np.float32)
                    if acc_vals.size > 0:
                        metrics["reward_model/acc"] = float(np.mean(acc_vals))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics:{last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

                batch = None
        # check if last step checkpoint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)

    def get_item_from_all_train(self, problem_id):
        return self.train_dataset.dataframe[problem_id]


    def _merge_pending_generated_batch_into_train_batch(
        self, new_batch: DataProto, pending_generated_batch: Optional[DataProto]
    ) -> DataProto:
        """
        Append rows from `pending_generated_batch` (rollout from `_generate_problem_variants` in the
        same `train_batch`, before Pass 2) onto the Pass-2 batch `new_batch`, and set token-level
        rewards from Pass-2 fields.

        For each uid on `new_batch`, average `acc` over rollouts with that uid; for each appended
        generated row, reward is that average if `knowledge` is non-empty for that uid, else -0.5.
        """
        if pending_generated_batch is None:
            return new_batch

        # uid -> average acc on Pass-2 batch (for gen-step reward)
        uid_to_acc = {}
        uid_to_knowledge = {}
        acc_vals = new_batch.non_tensor_batch.get("acc", None)
        knowledge_vals = new_batch.non_tensor_batch.get("knowledge", None)
        uids = new_batch.non_tensor_batch.get("uid", None)
        if acc_vals is not None and uids is not None:
            acc_vals = np.asarray(acc_vals, dtype=np.float32)
            uid_to_acc_list = {}
            for i in range(len(uids)):
                uid = uids[i]
                if uid is not None and str(uid).strip() != "":
                    uid = str(uid)
                    if i < len(acc_vals):
                        uid_to_acc_list.setdefault(uid, []).append(float(acc_vals[i]))
            for uid, a_list in uid_to_acc_list.items():
                uid_to_acc[uid] = sum(a_list) / len(a_list)

        if knowledge_vals is not None and uids is not None:
            knowledge_vals = np.asarray(knowledge_vals, dtype=np.float32)
            for i in range(len(uids)):
                uid = uids[i]
                if uid is not None and str(uid).strip() != "":
                    uid = str(uid)
                    if i < len(knowledge_vals):
                        uid_to_knowledge[uid]=bool(len(knowledge_vals[i]) > 0)

        pending_batch = pending_generated_batch

        device = pending_batch.batch["attention_mask"].device
        response_length = int(pending_batch.batch["responses"].shape[1])
        token_level_rewards_list = []
        keep_indices = []
        for i, uid in enumerate(uids):
                parsed_success = uid_to_knowledge.get(uid, False)
                if not parsed_success:
                    r_gen = -0.5
                else:
                    if uid not in uid_to_acc:
                        raise ValueError(f"uid {uid} not found in uid_to_acc")
                    acc = uid_to_acc[uid]
                    r_gen = acc
                keep_indices.append(i)
                valid_response_len = int(pending_batch.batch["attention_mask"][i, -response_length:].sum().item())
                last_pos_in_response = valid_response_len - 1
                r_vec = torch.zeros(response_length, device=device, dtype=torch.float32)
                if last_pos_in_response >= 0:
                    r_vec[last_pos_in_response] = float(r_gen)
                token_level_rewards_list.append(r_vec.unsqueeze(0))

        if keep_indices:
                idx = torch.tensor(keep_indices, dtype=torch.long, device=device)
                sub_batch = {k: v.index_select(0, idx) for k, v in pending_batch.batch.items()}
                sub_non_tensor = {k: np.array(v)[keep_indices] for k, v in pending_batch.non_tensor_batch.items()}
                token_level_rewards = torch.cat(token_level_rewards_list, dim=0)
                sub_batch["rm_scores"] = token_level_rewards
                sub_batch["token_level_scores"] = token_level_rewards
                sub_batch["token_level_rewards"] = token_level_rewards
                sub_batch_td = TensorDict(source=sub_batch, batch_size=[len(keep_indices)])
                added_batch = DataProto(
                    batch=sub_batch_td,
                    non_tensor_batch=sub_non_tensor,
                    meta_info=dict(pending_batch.meta_info),
                )
                added_batch.non_tensor_batch["uid"] = np.array(
                    [f"gen_{i}" for i in range(len(keep_indices))], dtype=object
                )

                for k in new_batch.non_tensor_batch.keys():
                    if k not in added_batch.non_tensor_batch:
                        added_batch.non_tensor_batch[k] = np.array([None] * len(keep_indices), dtype=object)


                new_batch = DataProto.concat([new_batch, added_batch])

        return new_batch

    def _rollout_and_compute_reward(self, batch_dict, _metrics, timing_raw):
        """
        One rollout pass: `generate_sequences`, optional REMAX baseline, then RM + rule `compute_reward`,
        filling `token_level_scores` / `token_level_rewards`.

        `_metrics` is reserved for call-site symmetry with `train_batch` (unused here).
        """
        new_batch: DataProto = DataProto.from_single_dict(batch_dict)
        # pop those keys for generation
        if "multi_modal_data" in new_batch.non_tensor_batch.keys():
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
        else:
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )
        gen_batch_output = gen_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
        )

        # generate a batch
        with marked_timer("gen", timing_raw, "red"):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
            timing_raw.update(gen_batch_output.meta_info["timing"])
            gen_batch_output.meta_info.pop("timing", None)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            with marked_timer("gen_max", timing_raw, "red"):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["do_sample"] = False
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                # compute reward model score on new_batch
                rm_scores = None
                if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                    rm_scores = self.rm_wg.compute_rm_score(new_batch)
                    new_batch = new_batch.union(rm_scores)
                reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                keys_to_pop = set(gen_baseline_output.batch.keys())
                if rm_scores is not None:
                    keys_to_pop.update(rm_scores.batch.keys())
                new_batch.pop(batch_keys=list(keys_to_pop))

                new_batch.batch["reward_baselines"] = reward_baseline_tensor

                del rm_scores, gen_baseline_batch, gen_baseline_output

        new_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
        )
        # repeat to align with repeated responses in rollout
        new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        new_batch = new_batch.union(gen_batch_output)

        if self.config.algorithm.use_kl_in_reward:
            raise NotImplementedError("use kl in reward is not supported for semi-self.")

        reward_extra_infos_dict = {}
        with marked_timer("reward", timing_raw, "yellow"):
            # compute scores. Support both model and function-based.
            # We first compute the scores using reward model. Then, we call reward_fn to combine
            # the results from reward model and rule-based results.
            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                # we first compute reward model score
                reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                new_batch = new_batch.union(reward_tensor)

            # we combine with rule-based rm
            reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

            new_batch.batch["token_level_scores"] = reward_tensor

            if reward_extra_infos_dict:
                new_batch.non_tensor_batch.update(
                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                )

            # Semi-self does not support folding KL into the reward; token rewards == scores.
            if self.config.algorithm.use_kl_in_reward:
                raise NotImplementedError("use kl in reward is not supported for semi-self.")
            else:
                new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]



        return new_batch, reward_extra_infos_dict

    def _compute_advantage_and_backward(self, batch, metrics, timing_raw, reward_extra_infos_dict):
        """
        Optional DP balance, KL-related log-probs (when not using KL-in-reward), values, advantage,
        rollout-correction (if configured), critic/actor updates, optional rollout logging.
        """
        # === Updating ===
        # Optionally rebalance token counts across DP ranks (can reorder rows; affects minibatch grouping).
        # Padding to world_size divisibility was explored for uneven merged batches (e.g. concat Pass-1+2);
        # that path is disabled here—enable only if you restore pad/trim around advantage.
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # With KL-in-reward disabled, attach old (and optional ref) log-probs for the policy loss / KL terms.
        if not self.config.algorithm.use_kl_in_reward:
            batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

        # compute values
        if self.use_critic:
            with marked_timer("values", timing_raw, "cyan"):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch)
            # IS and off-policy metrics already have rollout_corr/ prefix
            metrics.update(is_metrics)

        with marked_timer("adv", timing_raw, "brown"):
            # compute advantages on the driver (paired with pad/trim above if padding is re-enabled)
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )

        # update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, "pink"):
                critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_output_metrics)

        # implement critic warmup
        if self.config.trainer.critic_warmup <= self.global_steps:
            # update actor
            with marked_timer("update_actor", timing_raw, "red"):
                # For normalize_update_by_reference_batch_size: so K-sample update matches T-sample effect
                batch.meta_info["actual_global_batch_size"] = batch.batch["attention_mask"].shape[0]
                batch.meta_info["reference_batch_size"] = (
                    self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                )
                # Log batch sizes used by normalize_update_by_reference_batch_size
                metrics["train/actual_global_batch_size"] = batch.meta_info["actual_global_batch_size"]
                metrics["train/reference_batch_size"] = batch.meta_info["reference_batch_size"]
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)

        # Log rollout generations if enabled
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)


        return batch, metrics

    def collate_single_batch_from_train_problems(self, train_problems):
        """
        Build a fresh InMemoryRLHFDataset + StatefulDataLoader with batch_size=len(train_problems)
        and return the single collated batch dict (one batch per epoch when dataset size matches).

        Used after update_problems to align rollout with the refreshed curriculum before merge/backward.
        """
        if not train_problems:
            raise ValueError("train_problems is empty")
        dl = self.createInmemoryDataLoader(train_problems)
        return next(iter(dl))

    def train_batch(
        self,
        batch_dict,
        prev_step_profile,
        curr_step_profile,
        timing_raw,
    ):
        """
        1) Pass 1: rollout + reward on the dataloader batch; `update_action` then `update_problems`
           (may run `_generate_problem_variants`, producing `pending_generated_batch` for merge).
        2) Pass 2: collate `train_problems` and rollout + reward again (`batch_with_knowledge`).
        3) Merge `pending_generated_batch` into Pass 2 (reward from Pass-2 uid acc / knowledge), then
           `batch_pass1.union(batch_with_knowledge)`.
        4) Advantage + PPO backward on the combined batch.
        """
        metrics = {}

        with marked_timer("start_profile", timing_raw):
            self._start_profiling(
                not prev_step_profile and curr_step_profile
                if self.config.global_profiler.profile_continuous_steps
                else curr_step_profile
            )

        pending_generated_batch = None
        action_counts = {}

        with marked_timer("step", timing_raw):
            with marked_timer("train_batch/pass1_rollout_reward", timing_raw, "red"):
                batch, reward_extra_infos_dict = self._rollout_and_compute_reward(batch_dict, metrics, timing_raw)

            with marked_timer("train_batch/update_action", timing_raw, "blue"):
                self.update_action(batch)

            with marked_timer("train_batch/update_problems", timing_raw, "blue"):
                (
                    train_problems,
                    action_counts,
                    pending_generated_batch,
                ) = self.update_problems(batch)

                metrics.update({f"semi_self/action_counts/{k}": v for k, v in action_counts.items()})

            # Pass 2: collate from updated `train_problems`, then same rollout + reward as Pass 1
            with marked_timer("train_batch/pass2_collate", timing_raw, "blue"):
                batch_dict_with_knowledge = self.collate_single_batch_from_train_problems(train_problems)

            with marked_timer("train_batch/pass2_rollout_reward", timing_raw, "red"):
                batch_with_knowledge, reward_extra_infos_dict = self._rollout_and_compute_reward(
                    batch_dict_with_knowledge, metrics, timing_raw
                )

            # attach generated variants + rewards, then union Pass-1 and Pass-2 batches
            with marked_timer("train_batch/merge_and_union", timing_raw, "blue"):
                batch_with_knowledge = self._merge_pending_generated_batch_into_train_batch(
                    batch_with_knowledge, pending_generated_batch
                )
                batch = batch.union(batch_with_knowledge)

            # advantage + PPO backward (inner stages also record into timing_raw)
            with marked_timer("train_batch/ppo_backward", timing_raw, "pink"):
                batch, metrics = self._compute_advantage_and_backward(
                    batch, metrics, timing_raw, reward_extra_infos_dict
                )

        return (
            batch,
            metrics
        )

    def createInmemoryDataLoader(self,train_problems):
        train_dataset = InMemoryRLHFDataset(
            data_list=train_problems,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data
        )
        from verl.trainer.main_ppo import  create_rl_sampler
        train_sampler = create_rl_sampler(self.config.data, train_dataset)
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
        collate_fn = default_collate_fn
        num_workers = self.config.data["dataloader_num_workers"]
        inmemory_dataloader=StatefulDataLoader(
            dataset=train_dataset,
            batch_size=len(train_problems),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        return inmemory_dataloader
    def update_problems(self, batch, num_variations_per_problem=8):
        """
        Build the next curriculum: for each uid with action `add_in_context_knowledge`, call
        `_generate_problem_variants` and expand `updated_problems` with new items (prompt + knowledge).

        Args:
            batch: DataProto after `update_action`; must include `uid`, `action`, `problem_id`.
            num_variations_per_problem: Variants per uid that requested in-context knowledge (M).

        Returns:
            (updated_problems, action_counts, pending_generated_batch)
            - updated_problems: list of problem dicts for Pass-2 collate (see `train_batch`)
            - action_counts: per-action tallies for logging
            - pending_generated_batch: rollout `DataProto` from `_generate_problem_variants`, or None
              if there was no `add_in_context_knowledge` work (consumed in the same `train_batch` merge).
        """
        # Get uids and actions from batch
        uids = batch.non_tensor_batch.get("uid", [])
        uids = np.asarray(uids) if not isinstance(uids, np.ndarray) else uids
        actions = batch.non_tensor_batch.get("action", [])
        problem_ids = batch.non_tensor_batch.get("problem_id", [])
        

        


        # Initialize counters for metrics
        action_counts = {
            'drop': 0,
            'add_in_context_knowledge': 0,
            'add_in_context_knowledge_success': 0,
            'new_problems_appended': 0,
        }

       
        # Aggregate actions by uid (since batch contains multiple rollouts per prompt)
        uid_to_action = {}
        uid_to_problem_id = {}
        
        for i, uid in enumerate(uids):
            if uid not in uid_to_action:
                uid_to_action[uid] = actions[i] if i < len(actions) else "keep"
                uid_to_problem_id[uid] = problem_ids[i]


        # Update counters based on unique uids
        for action in uid_to_action.values():
            if action == 'add_in_context_knowledge':
                action_counts['add_in_context_knowledge'] += 1
            else:
                action_counts['drop'] += 1

        # Phase 1: uids that need LLM-generated knowledge variants
        generation_tasks = []  # List of (uid, action)

        for uid, action in uid_to_action.items():
            if action == 'add_in_context_knowledge':
                generation_tasks.append((uid, action))

        # Phase 2: Batch generate variants for add_in_context_knowledge operations
        generated_variants = {}
        problem_indices = []
        generated_output = None

        if generation_tasks:
            all_problems = []
            for uid, action in generation_tasks:
                problem_id = uid_to_problem_id.get(uid, 0)
                original_problem_data = self.get_item_from_all_train(problem_id)
                generation_problem = {
                    'problem': original_problem_data.get('extra_info', {}).get('question', ''),
                    'action': action
                    }
                all_problems.append(generation_problem)
                problem_indices.append((uid, action))

            # Batch generate all variants at once (each prompt -> N variants in all_variants)
            if all_problems:
                all_variants, add_knowledge_success_count, generated_output, parse_success_list = self._generate_problem_variants(all_problems, num_variations_per_problem)
                action_counts['add_in_context_knowledge_success'] = add_knowledge_success_count
                num_prompts = len(problem_indices)
                for p in range(num_prompts):
                    uid, action = problem_indices[p]
                    start = p * num_variations_per_problem
                    end = start + num_variations_per_problem
                    # All N variants for this (uid, action)
                    generated_variants[uid] = [(action, v) for v in all_variants[start:end]]

        


        # Phase 3: Flatten variants into `updated_problems`; keep generation rollout for merge after Pass 2
        pending_generated_batch = generated_output
        updated_problems = []
        

        for uid, action in uid_to_action.items():
            problem_id = uid_to_problem_id.get(uid, 0)

            if action == 'add_in_context_knowledge':
                original_problem_data = self.get_item_from_all_train(problem_id)

                if uid in generated_variants:
                    # Keep all generated variants for this prompt.
                    for va, v in generated_variants[uid]:
                        variant_action, variant = va, v
                        if variant_action is not None and variant and variant.get('problem'):
                            new_item = {
                                **original_problem_data,
                                "prompt": [
                                    {
                                        "role": "system",
                                        "content": "Please reason step by step, and put your final answer within \\boxed{{}}.",
                                    },
                                    {
                                        "role": "user",
                                        "content": str(variant['problem']),
                                    }
                                ],
                                "action": "keep",
                                "knowledge": variant.get("knowledge", []),
                                "problem_id": problem_id,
                                "data_source": str(variant['uid']),
                            }
                            updated_problems.append(new_item)
                        # Invalid or empty variant: skip (no row appended for that variant).

                else:
                    raise ValueError(f"Unexpect error for {original_problem_data}")


        return updated_problems, action_counts,  pending_generated_batch
        

    def update_problems_simple(self, train_problems, next_problem_id):
        """
        Update all problems with consecutive IDs starting from next_problem_id.

        Args:
            train_problems: List of problem dictionaries
            next_problem_id: Starting ID for the problems

        Returns:
            tuple: (updated_train_problems, new_next_problem_id)
        """
        updated_problems = []
        current_id = next_problem_id
        

        for problem in train_problems:
            updated_problem = dict(self.train_dataset.dataframe[current_id])
            current_id +=1
            updated_problems.append(updated_problem)
        return updated_problems, current_id

    def update_action(self, batch: DataProto):
        """
        Update actions based on verifier accuracy (acc), aggregated by uid.

        Args:
            batch: DataProto containing non_tensor_batch["acc"] and uids
        """
        acc_vals = batch.non_tensor_batch.get("acc", None)
        uids = batch.non_tensor_batch.get("uid", [])

        if acc_vals is None or len(uids) == 0:
            return

        acc_vals = np.asarray(acc_vals, dtype=np.float32)

        # Aggregate acc by uid (each uid corresponds to one original prompt; batch may have multiple rollouts)
        uid_to_acc_list = {}
        for i, uid in enumerate(uids):
            if i >= len(acc_vals):
                continue
            uid_to_acc_list.setdefault(uid, []).append(float(acc_vals[i]))

        uid_to_avg_acc = {uid: sum(vals) / len(vals) for uid, vals in uid_to_acc_list.items()}

        # Optional: downstream or logging can read per-uid averages from meta_info
        batch.meta_info["uid_to_avg_acc"] = uid_to_avg_acc
        batch.meta_info["uid_to_avg_reward"] = uid_to_avg_acc  # backward compat: same values, acc-based

        # Since batch is expanded, we need to get unique uids and their corresponding actions
        unique_uids = list(uid_to_avg_acc.keys())
        batch_data_sources = batch.non_tensor_batch.get("data_source", [])
        uid_to_data_source = {}
        for i, uid in enumerate(uids):
            if uid not in uid_to_data_source:
                ds = batch_data_sources[i] if i < len(batch_data_sources) else "general-reasoner"
                if ds is None or (isinstance(ds, str) and str(ds).strip() == ""):
                    ds = "general-reasoner"
                uid_to_data_source[uid] = str(ds)

        add_knowledge_threshold = getattr(
            self.config.data,
            'add_in_context_knowledge_threshold',
            getattr(self.config.data, 'degrade_threshold', -0.2),
        )
        updated_actions = []

        # Process each unique uid
        for uid in unique_uids:
            avg_acc = uid_to_avg_acc[uid]
            data_source = uid_to_data_source.get(uid, "general-reasoner")
            # Update action based on average acc for this uid
            if data_source != "general-reasoner":
                raise ValueError(f"data_source {data_source} is not supported")
            elif avg_acc < add_knowledge_threshold:
                new_action = 'add_in_context_knowledge'
            else:
                new_action = 'drop'

            # Store per-uid results
            updated_actions.append(new_action)

        # Update the batch with new actions (per-uid)
        # Note: Since batch is expanded with multiple samples per uid, we need to
        # replicate the per-uid actions for all samples of each uid
        batch_updated_actions = []
        for uid in uids:
            if uid in unique_uids:
                idx = unique_uids.index(uid)
                batch_updated_actions.append(updated_actions[idx])
            else:
                batch_updated_actions.append("drop")

        # Add assertion to verify dimensions match
        assert len(batch_updated_actions) == len(uids), f"Action length mismatch: {len(batch_updated_actions)} vs {len(uids)}"

        batch.non_tensor_batch["action"] = np.array(batch_updated_actions, dtype=object)

    def _generate_problem_variants(self, original_problems, num_variations_per_problem=4):
        """
        Generate in-context knowledge snippets with the policy and parse JSON from the reply.

        Args:
            original_problems: List of dicts with at least 'problem' and 'action' (e.g. add_in_context_knowledge).
            num_variations_per_problem: Rollout rows per source prompt (M).

        Returns:
            (new_problems, add_knowledge_success_count, generated_output, parse_success_list)
            - new_problems: Parsed dicts with 'problem', 'knowledge', 'uid' (fallback rows if parse fails)
            - add_knowledge_success_count: Count where action is add_in_context_knowledge and parse succeeded
            - generated_output: DataProto from `generate_sequences` (merged into Pass 2 in the same `train_batch`)
            - parse_success_list: Per-row whether JSON list-of-strings parsed
        """
        import torch
        from verl import DataProto
        from tensordict import TensorDict
        import json
        import re

        # Create prompts for each original problem
        prompts = []
        for problem_data in original_problems:
            problem = problem_data['problem']
            label = problem_data.get('action', 'add_in_context_knowledge')

            if label == 'add_in_context_knowledge':
                prompt = [{
                    "role":"user",
                    "content":f"""You are given a problem.

Problem:
{problem}

Return valid JSON in this format:
```json
[
  "knowledge 1",
  "knowledge 2"
]
```

Each string should be a piece of knowledge that helps solve the problem.
Do not include any explanation or markdown.
"""}]
            else:
                continue  # Skip unknown labels

            prompts.append(prompt)

        if not prompts:
            return [], 0, None, []

        # Repeat each prompt M times
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * num_variations_per_problem)

        # Tokenize prompts using the same pipeline as InMemoryRLHFDataset / RLHFDataset.__getitem__
        # so that generate_sequences receives identical prompt format to the inmemory_dataloader path.
        data_cfg = self.config.data
        apply_kwargs = data_cfg.get("apply_chat_template_kwargs", {})
        max_prompt_length = data_cfg.get("max_prompt_length", 1024)
        truncation = data_cfg.get("truncation", "error")
        pad_token_id = self.tokenizer.pad_token_id

        input_ids_list = []
        attention_mask_list = []
        raw_prompt_ids_list = []

        for prompt_messages in repeated_prompts:
            raw_prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False,
                **apply_kwargs,
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            ids = model_inputs["input_ids"]
            mask = model_inputs["attention_mask"]
            ids, mask = verl_F.postprocess_data(
                input_ids=ids,
                attention_mask=mask,
                max_length=max_prompt_length,
                pad_token_id=pad_token_id,
                left_pad=True,
                truncation=truncation,
            )
            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > max_prompt_length:
                if truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
                elif truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
                elif truncation == "middle":
                    left_half = max_prompt_length // 2
                    right_half = max_prompt_length - left_half
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                elif truncation == "error":
                    raise RuntimeError(
                        f"Prompt length {len(raw_prompt_ids)} is longer than max_prompt_length {max_prompt_length}."
                    )
            raw_prompt_ids_list.append(raw_prompt_ids)

        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        position_ids = compute_position_id_with_mask(attention_mask)

        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(repeated_prompts),
        )

        # One uid per original prompt, repeated for each variation of that prompt
        num_prompts = len(prompts)
        uid_per_prompt = [str(uuid.uuid4()) for _ in range(num_prompts)]
        uid_array = np.array(
            [uid_per_prompt[i // num_variations_per_problem] for i in range(len(repeated_prompts))],
            dtype=object,
        )
        gen_batch = DataProto(
            batch=batch,
            non_tensor_batch={
                "uid": uid_array,
                "data_source": np.array(["update_problems"] * len(repeated_prompts)),
                "raw_prompt_ids": np.array(raw_prompt_ids_list, dtype=object),
            },
            meta_info={
                "do_sample": True,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
                "top_p": self.config.actor_rollout_ref.rollout.top_p,
                "top_k": self.config.actor_rollout_ref.rollout.get("top_k", -1),
                "max_new_tokens": self.config.data.max_response_length,
            },
        )
        pprint(f"Generating {len(repeated_prompts)} variants ({len(prompts)} problems x {num_variations_per_problem} variations)")
        # Generate new problems using the policy model
        generated_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        # Extract the generated text
        generated_sequences = generated_output.batch["input_ids"]
        # Prompt was left-padded; rollout returns [prompt_seq, new_tokens]. Use input sequence length
        # so we decode only the assistant's reply (after the prompt), not the prompt again.
        input_seq_len = gen_batch.batch["input_ids"].shape[1]

        new_problems = []
        batch_size = len(repeated_prompts)
        parse_success_count = 0
        add_knowledge_success_count = 0
        parse_success_list = []

        for i in range(batch_size):
            generated_tokens = generated_sequences[i, input_seq_len:]

            # Decode the generated text
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Try to parse as JSON list of strings
            parsed_problem = None
            try:
                json_block_pattern = re.compile(r'```json\s*([\s\S]*?)```')
                matches = json_block_pattern.findall(generated_text)
                json_str = matches[-1].strip() if matches else generated_text
                parsed_data = json.loads(json_str)

                if isinstance(parsed_data, list) and all(isinstance(item, str) for item in parsed_data):
                    uid_i = gen_batch.non_tensor_batch["uid"][i]
                    original_idx = i // num_variations_per_problem
                    original_problem_text = ""
                    if original_idx < len(original_problems):
                        original_problem_text = str(original_problems[original_idx].get('problem', ''))
                    knowledge_lines = [k.strip() for k in parsed_data if isinstance(k, str) and k.strip()]
                    combined_problem = original_problem_text
                    if knowledge_lines:
                        combined_problem = (
                            f"{original_problem_text}\n\n"
                            "Helpful knowledge:\n"
                            + "\n".join([f"- {k}" for k in knowledge_lines])
                        )
                    parsed_problem = {
                        'problem': combined_problem,
                        'knowledge': knowledge_lines,
                        'uid': uid_i,
                    }
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

            parse_success_list.append(parsed_problem is not None)

            # If JSON parsing succeeded, use the parsed data and count by action
            if parsed_problem:
                new_problems.append(parsed_problem)
                parse_success_count += 1
                original_idx = i // num_variations_per_problem
                if original_idx < len(original_problems):
                    action = original_problems[original_idx].get('action', 'add_in_context_knowledge')
                    if action == 'add_in_context_knowledge':
                        add_knowledge_success_count += 1
            else:
                # Fallback
                uid_i = gen_batch.non_tensor_batch["uid"][i]
                original_idx = i // num_variations_per_problem
                original_problem_text = ""
                if original_idx < len(original_problems):
                    original_problem_text = str(original_problems[original_idx].get('problem', ''))
                new_problems.append({
                    'problem': original_problem_text,
                    'knowledge': [],
                    'uid': uid_i,
                })
            # Log the first case as an example
            if i == 0:
                pprint(f"[generate prompt]: {repeated_prompts[0]}")
                pprint(f"[generate output]: {generated_text}")
                if parsed_problem:
                    pprint(f"[parsed problem]: {parsed_problem['problem']}")
                    pprint(f"[parsed knowledge]: {parsed_problem['knowledge']}")
                else:
                    pprint("FAILED - falling back to raw text")
        pprint(f"parse success:{parse_success_count},batch_size:{batch_size}")
        return new_problems, add_knowledge_success_count, generated_output, parse_success_list
