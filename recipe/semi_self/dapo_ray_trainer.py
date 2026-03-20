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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
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
from verl.utils.dataset import inmemory_dataset
from verl.utils.dataset.inmemory_dataset import InMemoryRLHFDataset
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask




                
class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
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
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
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
        original_batch=None
        added_batch = None
        num_prompt_in_batch = 0
        
        # TODO: 当难度出现反复时，
        self.all_train_object =[]
        for item in self.train_dataset.dataframe:
            self.all_train_object.append({
                "level":0,
                "dict":{
                    0:item
                    }
                })
        # Get train problems directly from the raw dataframe (not processed data)
        # since InMemoryRLHFDataset expects raw data with 'prompt' field
        train_problems = []
        for i in range(min(self.config.data.train_batch_size, len(self.train_dataset))):
            # Get raw data from dataframe, not processed data from __getitem__
            problem = dict(self.train_dataset.dataframe[i])
            train_problems.append({**problem,"action":"keep","keep_count":0,"problem_id":i,"level":0})


        inmemory_dataloader=self.createInmemoryDataLoader(train_problems)

        next_problem_id = self.config.data.train_batch_size
        self.pending_generated_samples = []
        self.pending_generated_batch = None
        
        self.pending_super_uid_to_new_item = {}

        for epoch in range(self.config.trainer.total_epochs):
            # only one batch for InMemoryRLHFDataset
            for batch_dict in inmemory_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps
                original_batch, added_batch, metrics = self.train_batch(batch_dict, prev_step_profile, curr_step_profile, timing_raw)
                # Combined batch for metrics (original + added)
                batch = (
                    DataProto.concat([original_batch, added_batch])
                    if added_batch is not None and original_batch is not None
                    else (original_batch if original_batch is not None else added_batch)
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

                # update train_problems and inmemory_dataloader
                # Use the current batch (which contains uids and updated actions from update_action)
                # The batch should already have the correct actions and keep_counts from update_action
                timing_raw = defaultdict(float)
                with marked_timer("update_problems", timing_raw, "blue"):
                    # original_batch 有一些是updated_problems,这些updated_problems如果不管控，那么batch_sizeh会越来越大。
                    train_problems, next_problem_id, action_counts, generated_samples, pending_generated_batch = self.update_problems(original_batch, train_problems, next_problem_id)
                    if self.config.trainer.get("generation_train",False):
                        self.pending_generated_samples = generated_samples
                        self.pending_generated_batch = pending_generated_batch
                    inmemory_dataloader=self.createInmemoryDataLoader(train_problems)

                batch = None

                # Log next_problem_id, update_problems timing, and action counts as metrics
                update_metrics = {
                    "train/next_problem_id": next_problem_id,
                    "timing/update_problems": timing_raw["update_problems"],
                    "train/action_drop": action_counts["drop"],
                    "train/action_add_in_context_knowledge": action_counts["add_in_context_knowledge"],
                    "train/action_add_in_context_knowledge_success": action_counts["add_in_context_knowledge_success"],
                    "train/new_problems_appended": action_counts["new_problems_appended"],
                }
            logger.log(data=update_metrics, step=self.global_steps)
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)

    def get_item_from_all_train(self,problem_id):
        object = self.all_train_object[problem_id]
        return object['dict'][object['level']]
    def update_item_for_all_train(self,problem_id,level,item):
        self.all_train_object[problem_id]['dict'][level]=item

    def train_batch(self, batch_dict, prev_step_profile, curr_step_profile,  timing_raw):
        metrics = {}

        with marked_timer("start_profile", timing_raw):
            self._start_profiling(
                not prev_step_profile and curr_step_profile
                if self.config.global_profiler.profile_continuous_steps
                else curr_step_profile
            )

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

        

        with marked_timer("step", timing_raw):
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
                # We need these metrics for apply_kl_penalty if using kl in reward
                new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                # otherwise, we will compute those after dynamic sampling

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

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    new_batch, kl_metrics = apply_kl_penalty(
                        new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(
                        kl_metrics
                    )  # TODO: This will be cleared if we use multiple genenration batches
                else:
                    new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

            # Update actions based on reward scores
            self.update_action(new_batch)

            # Compute super_uid -> average acc for samples from generated problems (for gen-step reward)
            super_uid_to_acc = {}
            acc_vals = new_batch.non_tensor_batch.get("acc", None)
            super_uids = new_batch.non_tensor_batch.get("super_uid", None)
            if acc_vals is not None and super_uids is not None:
                acc_vals = np.asarray(acc_vals, dtype=np.float32)
                super_uid_to_acc_list = {}
                for i in range(len(super_uids)):
                    sid = super_uids[i]
                    if sid is not None and str(sid).strip() != "":
                        sid = str(sid)
                        if i < len(acc_vals):
                            super_uid_to_acc_list.setdefault(sid, []).append(float(acc_vals[i]))
                for sid, a_list in super_uid_to_acc_list.items():
                    super_uid_to_acc[sid] = sum(a_list) / len(a_list)

            # Merge pending generated batch (from previous step's add_in_context_knowledge) with rewards for PPO
            if getattr(self, "pending_generated_batch", None) is not None and getattr(self, "pending_generated_samples", None):
                pending_batch = self.pending_generated_batch
                pending_samples = self.pending_generated_samples
                if len(pending_samples) > 0 and pending_batch.batch["attention_mask"].shape[0] == len(pending_samples):
                    device = pending_batch.batch["attention_mask"].device
                    dtype = pending_batch.batch["attention_mask"].dtype
                    # Reward-related tensors must be response_length only (match new_batch / fsdp _expand_to_token_level)
                    response_length = int(pending_batch.batch["responses"].shape[1])
                    token_level_rewards_list = []
                    keep_indices = []
                    for i, info in enumerate(pending_samples):
                        super_uid = info["super_uid"]
                        parsed_success = info["parsed_success"]
                        if not parsed_success:
                            r_gen = -0.5
                        else:
                            if super_uid not in super_uid_to_acc:
                                continue
                            acc = super_uid_to_acc[super_uid]
                            r_gen = 1.0 - 2.0 * abs(acc - 0.5)
                        keep_indices.append(i)
                        # Last valid position within response part only (not full sequence)
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
                        # Reward tensors: response_length only (same as new_batch / fsdp _expand_to_token_level)
                        sub_batch["rm_scores"] = token_level_rewards
                        sub_batch["token_level_scores"] = token_level_rewards
                        sub_batch["token_level_rewards"] = token_level_rewards
                        sub_batch_td = TensorDict(source=sub_batch, batch_size=[len(keep_indices)])
                        added_batch = DataProto(batch=sub_batch_td, non_tensor_batch=sub_non_tensor, meta_info=dict(pending_batch.meta_info))
                        added_batch.non_tensor_batch["uid"] = np.array([f"gen_{i}" for i in range(len(keep_indices))], dtype=object)
                        for k in new_batch.non_tensor_batch.keys():
                            if k not in added_batch.non_tensor_batch:
                                added_batch.non_tensor_batch[k] = np.array([None] * len(keep_indices), dtype=object)
                        # Mark source so we can split original vs added at return
                        n_orig = new_batch.batch["attention_mask"].shape[0]
                        new_batch.non_tensor_batch["_batch_source"] = np.array(["original"] * n_orig, dtype=object)
                        added_batch.non_tensor_batch["_batch_source"] = np.array(["added"] * len(keep_indices), dtype=object)
                        new_batch = DataProto.concat([new_batch, added_batch])
                    
                    self.pending_generated_batch = None
                    self.pending_samples = None
                else:
                    if len(pending_samples) > 0:
                        raise ValueError(f'{pending_batch.batch["attention_mask"].shape[0]} != {len(pending_samples)}')

            if not self.config.algorithm.filter_groups.enable:
                batch = new_batch
                if "_batch_source" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["_batch_source"] = np.array(
                        ["original"] * batch.batch["attention_mask"].shape[0], dtype=object
                    )
            else:
                raise NotImplementedError("Filter groups are not supported for semi-self.")
                            
        

            # === Updating ===
            # Balance the number of valid tokens across DP ranks.
            # NOTE: This usually changes the order of data in the `batch`,
            # which won't affect the advantage calculation (since it's based on uid),
            # but might affect the loss calculation (due to the change of mini-batching).
            # TODO: Decouple the DP balancing and mini-batching.
            # Skip when batch_size is not divisible by world_size (e.g. semi-self merged batch 2052 % 8).
            # world_size = self.actor_rollout_wg.world_size
            # batch_size = batch.batch["attention_mask"].shape[0]
            # pad_size = 0
            # if batch_size % world_size != 0:
            #     # Pad so DP chunk (e.g. compute_log_prob) can split evenly; trim before advantage.
            #     batch, pad_size = pad_dataproto_to_divisor(batch, world_size)
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

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
                # # Remove padding added for DP chunking so advantage is only over real samples.
                # if pad_size > 0:
                #     original_len = len(batch) - pad_size
                #     batch = batch.select_idxs(list(range(original_len)))
                #     batch.meta_info["global_token_num"] = batch.meta_info["global_token_num"][:original_len]
                # compute advantages, executed on the driver process
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
                    # 也把这两个信息打到 metrics 里，方便在 logger 中查看
                    metrics["train/actual_global_batch_size"] = batch.meta_info["actual_global_batch_size"]
                    metrics["train/reference_batch_size"] = batch.meta_info["reference_batch_size"]
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            # Log rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

            # Split batch into original and added (from pending generated) for separate return
            src = batch.non_tensor_batch.get("_batch_source", None)
            if src is not None:
                src = np.asarray(src)
                orig_idx = np.where(src == "original")[0]
                add_idx = np.where(src == "added")[0]
                device = batch.batch["attention_mask"].device
                if len(orig_idx) > 0:
                    o_idx = torch.tensor(orig_idx, dtype=torch.long, device=device)
                    orig_batch_dict = {
                        k: v.index_select(0, o_idx) for k, v in batch.batch.items()
                    }
                    orig_batch = TensorDict(source=orig_batch_dict, batch_size=[len(orig_idx)])
                    orig_non = {k: np.array(v)[orig_idx] for k, v in batch.non_tensor_batch.items() if k != "_batch_source"}
                    original_batch = DataProto(batch=orig_batch, non_tensor_batch=orig_non, meta_info=dict(batch.meta_info))
                else:
                    original_batch = None
                if len(add_idx) > 0:
                    a_idx = torch.tensor(add_idx, dtype=torch.long, device=device)
                    add_batch_dict = {
                        k: v.index_select(0, a_idx) for k, v in batch.batch.items()
                    }
                    add_batch = TensorDict(source=add_batch_dict, batch_size=[len(add_idx)])
                    add_non = {k: np.array(v)[add_idx] for k, v in batch.non_tensor_batch.items() if k != "_batch_source"}
                    added_batch = DataProto(batch=add_batch, non_tensor_batch=add_non, meta_info=dict(batch.meta_info))
                else:
                    added_batch = None
            else:
                original_batch = batch
                added_batch = None

            if original_batch is not None and "_batch_source" in original_batch.non_tensor_batch:
                original_batch.non_tensor_batch.pop("_batch_source", None)
            if added_batch is not None and "_batch_source" in added_batch.non_tensor_batch:
                added_batch.non_tensor_batch.pop("_batch_source", None)

            return original_batch, added_batch, metrics
        
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
    def update_problems(self, batch, train_problems, next_problem_id, num_variations_per_problem=8):
        """
        Update problems based on actions in the batch.

        Args:
            batch: DataProto batch containing actions aggregated by uid
            train_problems: Current list of training problems with their states
            next_problem_id: Starting ID for newly appended problems
            num_variations_per_problem: Number of new problems to generate per original problem (M)

        Returns:
            tuple: (updated_problems, new_next_problem_id, action_counts, generated_samples, pending_generated_batch)
            - generated_samples: list of dicts with super_uid, problem_id, level, parsed_success (for RL reward in next train_batch)
            - pending_generated_batch: DataProto from _generate_problem_variants (None if no generation)
        """
        # Get uids and actions from batch
        uids = batch.non_tensor_batch.get("uid", [])
        uids = np.asarray(uids) if not isinstance(uids, np.ndarray) else uids
        actions = batch.non_tensor_batch.get("action", [])
        problem_ids = batch.non_tensor_batch.get("problem_id", [])
        levels = batch.non_tensor_batch.get("level", [])
        super_uids = batch.non_tensor_batch.get("super_uid", [])
        

                
                
        uid_to_avg_reward = batch.meta_info.get("uid_to_avg_reward") or batch.non_tensor_batch.get("uid_to_avg_reward") or {}
        # Phase 0: Cleanup uids by data_source — keep all for "general-reasoner", else keep only highest-reward uid per gen
        data_sources = batch.non_tensor_batch.get("data_source", None)
        if data_sources is not None:
            data_sources = np.asarray(data_sources)
        else:
            data_sources = np.array(["general-reasoner"] * len(uids), dtype=object)

        data_source_to_uids = {}
        for i in range(len(uids)):
            ds = data_sources[i] if i < len(data_sources) else "general-reasoner"
            if ds is None or (isinstance(ds, str) and str(ds).strip() == ""):
                ds = "general-reasoner"
            ds = str(ds)
            if ds not in data_source_to_uids:
                data_source_to_uids[ds] = []
            data_source_to_uids[ds].append(uids[i])

        keep_uids = set()
        best_uids = set()
        for ds, uid_list in data_source_to_uids.items():
            unique_uids = list(dict.fromkeys(uid_list))
            if ds == "general-reasoner":
                keep_uids.update(unique_uids)
            else:
                best_uid = max(
                    unique_uids,
                    key=lambda u:  1-2*abs(uid_to_avg_reward.get(u,0.0)-0.5),
                )
                keep_uids.add(best_uid)
                best_uids.add(best_uid)
        


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
        uid_to_level = {}
        uid_to_super_uid ={}
        
        for i, uid in enumerate(uids):
            if uid not in uid_to_action and uid in keep_uids:
                uid_to_action[uid] = actions[i] if i < len(actions) else "keep"
                uid_to_problem_id[uid] = problem_ids[i]
                uid_to_level[uid] =int(levels[i])
                if len(super_uids) > 0:
                    uid_to_super_uid[uid]=super_uids[i]
                
        uids = keep_uids
        
        for uid in best_uids:
            problem_id = uid_to_problem_id[uid]
            level = uid_to_level[uid]
            super_uid = uid_to_super_uid[uid]
            self.update_item_for_all_train(problem_id,level,self.pending_super_uid_to_new_item[super_uid])
        
        self.pending_super_uid_to_new_item = {}
        
        

        # Update counters based on unique uids
        for action in uid_to_action.values():
            if action == 'add_in_context_knowledge':
                action_counts['add_in_context_knowledge'] += 1
            else:
                action_counts['drop'] += 1

        # We need original problem data - this should be stored in the batch
        # For now, assume we can reconstruct or access original problems
        # This might need to be adjusted based on how original problems are stored

        # Phase 1: Collect all generation tasks for batch processing (per uid)
        generation_tasks = []  # List of (uid, action)

        for uid, action in uid_to_action.items():
            if action == 'add_in_context_knowledge':
                generation_tasks.append((uid, action))

        # Phase 2: Batch generate variants for add_in_context_knowledge operations
        generated_variants = {}
        problem_indices = []
        generated_output = None
        parse_success_list = []
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

        


        # Phase 3: Build final updated_problems list (one per unique uid)
        # Build generated_samples and pending_generated_batch for RL reward in next train_batch
        pending_generated_batch = generated_output
        generated_samples = []
        updated_problems = []
        current_next_id = next_problem_id
        

        for uid, action in uid_to_action.items():
            problem_id = uid_to_problem_id.get(uid, 0)
            level = uid_to_level.get(uid, 0)

            if action == 'add_in_context_knowledge':
                original_problem_data = self.get_item_from_all_train(problem_id)

                if uid in generated_variants:
                    # Keep all generated variants for this prompt.
                    for va, v in generated_variants[uid]:
                        variant_action, variant = va, v
                        super_uid = str(uuid.uuid4())
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
                                "keep_count": 0,
                                "problem_id": problem_id,
                                "level": level,
                                "data_source": str(variant['uid']),
                                "super_uid": super_uid
                            }
                            updated_problems.append(new_item)
                            self.pending_super_uid_to_new_item[super_uid] = new_item

                            generated_samples.append({
                                "super_uid": super_uid,
                                "problem_id": uid_to_problem_id.get(uid, 0),
                                "level": level,
                                "parsed_success": bool(variant.get("knowledge")),
                            })
                        # If action doesn't need add_knowledge, that prompt is dropped entirely.
                        
                else:
                    raise ValueError(f"Unexpect error for {original_problem_data}")

        # After each round, append batch_size new problems from the dataset tail.
        batch_size_to_add = len(uid_to_action)
        for _ in range(batch_size_to_add):
            new_problem = self.get_item_from_all_train(current_next_id)
            updated_problems.append({
                **new_problem,
                "action": "keep",
                "keep_count": 0,
                "problem_id": current_next_id,
                "level": 0,
                "data_source": "general-reasoner",
                "super_uid": "none"
            })
            current_next_id += 1
        action_counts["new_problems_appended"] = batch_size_to_add

        return updated_problems, current_next_id, action_counts, generated_samples, pending_generated_batch
        

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
        Update actions for prompts based on their reward scores.
        Rewards are aggregated by uid since batch contains multiple rollouts per prompt.

        Args:
            batch: DataProto containing the batch data with rewards
        """
        # Get rewards from batch
        rewards = batch.batch.get("token_level_rewards", None)
        uids = batch.non_tensor_batch.get("uid", [])

        if rewards is None or len(uids) == 0:
            return

        # Calculate average reward for each sample (sum over sequence dimension)
        reward_sums = rewards.sum(dim=-1)  # (batch_size,)
        # Clamp negative rewards to 0 per sample
        sample_rewards = reward_sums.clamp(min=0)

        # Aggregate rewards by uid (each uid corresponds to one original prompt)
        uid_to_rewards = {}
        for i, uid in enumerate(uids):
            if uid not in uid_to_rewards:
                uid_to_rewards[uid] = []
            uid_to_rewards[uid].append(sample_rewards[i].item())


        # Calculate average reward per uid
        uid_to_avg_reward = {}
        for uid, reward_list in uid_to_rewards.items():
            uid_to_avg_reward[uid] = sum(reward_list) / len(reward_list)


        # Store for Phase 0 in update_problems (cleanup by data_source)
        batch.meta_info["uid_to_avg_reward"] = uid_to_avg_reward  # also in meta_info so it survives batch split

        # Since batch is expanded, we need to get unique uids and their corresponding actions
        unique_uids = list(uid_to_avg_reward.keys())

        add_knowledge_threshold = getattr(
            self.config.data,
            'add_in_context_knowledge_threshold',
            getattr(self.config.data, 'degrade_threshold', -0.2),
        )
        updated_actions = []

        # Process each unique uid
        for uid in unique_uids:
            avg_reward = uid_to_avg_reward[uid]
            # Update action based on average reward for this uid
            if avg_reward < add_knowledge_threshold:
                new_action = 'add_in_context_knowledge'
            else:
                new_action = 'drop'

            # Store per-uid results
            updated_actions.append(new_action)

        # Update the batch with new actions and keep_counts (per-uid)
        # Note: Since batch is expanded with multiple samples per uid, we need to
        # replicate the per-uid actions/keep_counts for all samples of each uid
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
        Generate in-context knowledge using LLM.

        Args:
            original_problems: List of dicts with 'problem', 'answer', and 'action'
            num_variations_per_problem: Number of variants per problem

        Returns:
            tuple: (new_problems, add_knowledge_success_count)
            - new_problems: List of dicts with 'problem', 'knowledge', and 'uid' keys
            - add_knowledge_success_count: Number of parsed knowledge generations
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
