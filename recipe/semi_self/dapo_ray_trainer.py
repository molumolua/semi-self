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


        for epoch in range(self.config.trainer.total_epochs):
            # only one batch for InMemoryRLHFDataset
            for batch_dict in inmemory_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps
                batch,metrics= self.train_batch(batch_dict,prev_step_profile,curr_step_profile,timing_raw)
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
                    train_problems, next_problem_id, action_counts = self.update_problems(batch, train_problems, next_problem_id)
                    inmemory_dataloader=self.createInmemoryDataLoader(train_problems)

                batch = None

                # Log next_problem_id, update_problems timing, and action counts as metrics
                update_metrics = {
                    "train/next_problem_id": next_problem_id,
                    "timing/update_problems": timing_raw["update_problems"],
                    "train/action_keep": action_counts["keep"],
                    "train/action_replace": action_counts["replace"],
                    "train/action_upgrade": action_counts["upgrade"],
                    "train/action_degrade": action_counts["degrade"],
                    "train/action_upgrade_success": action_counts["upgrade_success"],
                    "train/action_degrade_success": action_counts["degrade_success"]
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

            if not self.config.algorithm.filter_groups.enable:
                batch = new_batch
            else:
                raise NotImplementedError("Filter groups are not supported for semi-self.")
                            
        

            # === Updating ===
            # Balance the number of valid tokens across DP ranks.
            # NOTE: This usually changes the order of data in the `batch`,
            # which won't affect the advantage calculation (since it's based on uid),
            # but might affect the loss calculation (due to the change of mini-batching).
            # TODO: Decouple the DP balancing and mini-batching.
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
            
            return batch,metrics
        
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
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        return inmemory_dataloader
    def update_problems(self, batch, train_problems, next_problem_id, num_variations_per_problem=4):
        """
        Update problems based on actions in the batch.

        Args:
            batch: DataProto batch containing actions aggregated by uid
            train_problems: Current list of training problems with their states
            next_problem_id: Starting ID for new problems (used for replace action)
            num_variations_per_problem: Number of new problems to generate per original problem (M)

        Returns:
            tuple: (updated_problems, new_next_problem_id, action_counts) - same format as update_problems_simple
        """
        # Get uids, actions and keep_counts from batch
        uids = batch.non_tensor_batch.get("uid", [])
        actions = batch.non_tensor_batch.get("action", [])
        keep_counts = batch.non_tensor_batch.get("keep_count", [])
        problem_ids = batch.non_tensor_batch.get("problem_id",[])
        levels = batch.non_tensor_batch.get("level",[])

        # Initialize counters for metrics
        action_counts = {
            'keep': 0,
            'replace': 0,
            'upgrade': 0,
            'degrade': 0,
            'upgrade_success': 0,  # Actually successfully upgraded
            'degrade_success': 0   # Actually successfully degraded
        }

        # Aggregate actions by uid (since batch contains multiple rollouts per prompt)
        uid_to_action = {}
        uid_to_keep_count = {}
        uid_to_problem_id = {}
        uid_to_level = {}

        for i, uid in enumerate(uids):
            if uid not in uid_to_action:
                uid_to_action[uid] = actions[i] if i < len(actions) else "keep"
                uid_to_keep_count[uid] = keep_counts[i] if i < len(keep_counts) else 0
                uid_to_problem_id[uid] = problem_ids[i]
                uid_to_level[uid] =levels[i]

        # Update counters based on unique uids
        for action in uid_to_action.values():
            if action == 'upgrade':
                action_counts['upgrade'] += 1
            elif action == 'degrade':
                action_counts['degrade'] += 1
            elif action == 'replace':
                action_counts['replace'] += 1
            elif action == 'keep':
                action_counts['keep'] += 1

        # We need original problem data - this should be stored in the batch
        # For now, assume we can reconstruct or access original problems
        # This might need to be adjusted based on how original problems are stored

        # Phase 1: Collect all generation tasks for batch processing (per uid)
        generation_tasks = []  # List of (uid, action)

        for uid, action in uid_to_action.items():
            if action in ['upgrade', 'degrade']:
                generation_tasks.append((uid, action))

        # Phase 2: Batch generate variants for all upgrade/degrade operations
        generated_variants = {}  # index -> (action, variant)
        if generation_tasks:
            # Unified batch processing for both upgrade and degrade
            all_problems = []
            problem_indices = []
            

            for uid, action in generation_tasks:
                problem_id = uid_to_problem_id.get(uid, 0)
                original_problem_data = self.get_item_from_all_train(problem_id)
                generation_problem = {
                    'problem': original_problem_data.get('extra_info', {}).get('question', ''),
                    'answer': original_problem_data.get('extra_info', {}).get('answer', ''),
                    'action': action
                    }
                all_problems.append(generation_problem)
                problem_indices.append((uid, action))

            # Batch generate all variants at once
            if all_problems:
                all_variants, upgrade_success_count, degrade_success_count = self._generate_problem_variants(all_problems, 1)
                action_counts['upgrade_success'] = upgrade_success_count
                action_counts['degrade_success'] = degrade_success_count
                for i, variant in enumerate(all_variants):
                    if i < len(problem_indices):
                        uid, action = problem_indices[i]
                        generated_variants[uid] = (action, variant)

        # Phase 3: Build final updated_problems list (one per unique uid)
        updated_problems = []
        current_next_id = next_problem_id

       

        for uid, action in uid_to_action.items():
            keep_count = uid_to_keep_count.get(uid, 0)
            problem_id = uid_to_problem_id.get(uid, 0)
            level = uid_to_level.get(uid,0)

            if action == 'keep':
                original_problem_data = self.get_item_from_all_train(problem_id)
                updated_problems.append({
                    **original_problem_data,
                    "action": "keep",
                    "keep_count": keep_count,
                    "problem_id":problem_id,
                    "level":level
                })

            elif action == 'replace':
                new_problem = self.get_item_from_all_train(current_next_id)
                updated_problems.append({
                    **new_problem,
                    "action": "keep",
                    "keep_count": 0,
                    "problem_id":problem_id,
                    "level":0
                })
                current_next_id += 1

            elif action in ['upgrade', 'degrade']:
                original_problem_data = self.get_item_from_all_train(problem_id)

                if uid in generated_variants:
                    variant_action, variant = generated_variants[uid]
                    if variant['problem'] and variant['answer']:
                        new_level=level
                        if variant_action == 'upgrade':
                            new_level+=1
                        else:
                            new_level-=1
                        new_item ={
                            **original_problem_data,
                            "prompt": [
                                {
                                    "role": "system",
                                    "content": "Please reason step by step, and put your final answer within \\boxed{{}}.",
                                },
                                {
                                    "role": "user",
                                    "content": variant['problem'],
                                }
                            ],
                            "reward_model": {
                                "style": "rule",
                                "ground_truth": variant['answer'],
                            },
                            "extra_info": {
                                'split': original_problem_data['extra_info']['split'],
                                'index': original_problem_data['extra_info']['index'],
                                'answer': variant['answer'],
                                "question": variant['problem'],
                                'level': original_problem_data['extra_info']['level'],
                            },
                            "action": "keep",
                            "keep_count": 0,
                            "problem_id":problem_id,
                            "level":new_level
                        }
                        self.update_item_for_all_train(problem_id,level,new_item)
                        updated_problems.append(new_item)
                    else:
                        updated_problems.append({
                            **original_problem_data,
                            "action": "keep",
                            "keep_count": keep_count,
                            "problem_id":problem_id,
                            "level":level
                        })
                else:
                    updated_problems.append({
                        **original_problem_data,
                        "action": "keep",
                        "keep_count": keep_count,
                        "problem_id":problem_id,
                        "level":level
                    })

        return updated_problems, current_next_id, action_counts
        

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
        Actions and keep_count are stored directly in the batch data.
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

        # Get current actions and keep_counts (these should be per-uid, not per-sample)
        # Since batch is expanded, we need to get unique uids and their corresponding actions
        unique_uids = list(uid_to_avg_reward.keys())

        # For actions and keep_counts, we assume they are stored per-uid
        # If not available, initialize with defaults
        current_actions = {}
        current_keep_counts = {}

        # Try to get existing actions and keep_counts from batch
        batch_actions = batch.non_tensor_batch.get("action", [])
        batch_keep_counts = batch.non_tensor_batch.get("keep_count", [])

        # Map existing actions/keep_counts to uids (if available)
        for i, uid in enumerate(uids[:len(batch_actions)]):
            current_actions[uid] = batch_actions[i]
            current_keep_counts[uid] = batch_keep_counts[i] if i < len(batch_keep_counts) else 0

        upgrade_threshold = getattr(self.config.data, 'upgrade_threshold', 0.8)
        degrade_threshold = getattr(self.config.data, 'degrade_threshold', -0.2)
        keep_max = getattr(self.config.data, 'keep_max', 5)

        updated_actions = []
        updated_keep_counts = []

        # Process each unique uid
        for uid in unique_uids:
            avg_reward = uid_to_avg_reward[uid]
            current_action = current_actions.get(uid, "keep")
            current_keep_count = current_keep_counts.get(uid, 0)

            # Update action based on average reward for this uid
            if current_keep_count >= keep_max:
                    new_keep_count = 0
                    new_action = 'replace'
            elif avg_reward > upgrade_threshold:
                new_action = 'upgrade'
                new_keep_count = current_keep_count + 1
            elif avg_reward < degrade_threshold:
                new_action = 'degrade'
                new_keep_count = current_keep_count + 1
            else:
                # Keep action
                new_keep_count = current_keep_count + 1
                new_action = 'keep'

            # Store per-uid results
            updated_actions.append(new_action)
            updated_keep_counts.append(new_keep_count)

        # Update the batch with new actions and keep_counts (per-uid)
        # Note: Since batch is expanded with multiple samples per uid, we need to
        # replicate the per-uid actions/keep_counts for all samples of each uid
        batch_updated_actions = []
        batch_updated_keep_counts = []

        for uid in uids:
            if uid in unique_uids:
                idx = unique_uids.index(uid)
                batch_updated_actions.append(updated_actions[idx])
                batch_updated_keep_counts.append(updated_keep_counts[idx])
            else:
                # Fallback for uids not in our processing
                batch_updated_actions.append("keep")
                batch_updated_keep_counts.append(0)

        # Add assertion to verify dimensions match
        assert len(batch_updated_actions) == len(uids), f"Action length mismatch: {len(batch_updated_actions)} vs {len(uids)}"
        assert len(batch_updated_keep_counts) == len(uids), f"Keep count length mismatch: {len(batch_updated_keep_counts)} vs {len(uids)}"

        batch.non_tensor_batch["action"] = np.array(batch_updated_actions, dtype=object)
        batch.non_tensor_batch["keep_count"] = np.array(batch_updated_keep_counts, dtype=object)

    def _generate_problem_variants(self, original_problems, num_variations_per_problem=4):
        """
        Generate problem variants using LLM (upgrade/degrade logic).

        Args:
            original_problems: List of dicts with 'problem', 'answer', and 'action'
            num_variations_per_problem: Number of variants per problem

        Returns:
            tuple: (new_problems, upgrade_success_count, degrade_success_count)
            - new_problems: List of dicts with 'problem' and 'answer' keys
            - upgrade_success_count: Number of upgrade variants that were successfully parsed
            - degrade_success_count: Number of degrade variants that were successfully parsed
        """
        import torch
        from verl import DataProto
        from tensordict import TensorDict
        import json

        # Create prompts for each original problem
        prompts = []
        for problem_data in original_problems:
            problem = problem_data['problem']
            answer = problem_data['answer']
            label = problem_data.get('action', 'upgrade')  # Default to upgrade

            if label == 'upgrade':
                # Create prompt to make the problem harder
                prompt = [{
                    "role":"user",
                    "content":f"""You are given a problem in JSON format. Create a more difficult version of this problem.
Keep the core concept the same but increase the complexity, add more constraints, or require deeper understanding.

Input:
```json
{{
"problem": "{problem}",
"answer": "{answer}"
}}
```

Generate a harder version and output in the same JSON format:
```json
{{
"problem": "your harder problem here",
"answer": "corresponding answer here"
}}
```
"""}]
            elif label == 'degrade':
                # Create prompt to make the problem easier
                prompt = [{
                    "role":"user",
                    "content":f"""You are given a problem in JSON format. Create a simpler version of this problem.
Keep the core concept the same but reduce the complexity, simplify the requirements, or make it more straightforward.

Input:
```json
{{
"problem": "{problem}",
"answer": "{answer}"
}}
```
Generate an easier version and output in the same JSON format:
```json
{{
"problem": "your easier problem here",
"answer": "corresponding answer here"
}}
```
"""}]
            else:
                continue  # Skip unknown labels

            prompts.append(prompt)

        if not prompts:
            return [], 0, 0

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

        gen_batch = DataProto(
            batch=batch,
            non_tensor_batch={
                "uid": np.arange(len(repeated_prompts)),
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
        upgrade_success_count = 0
        degrade_success_count = 0

        for i in range(batch_size):
            generated_tokens = generated_sequences[i, input_seq_len:]

            # Decode the generated text
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Try to parse as JSON
            parsed_problem = None
            try:
                # Find JSON in the generated text
                start_idx = generated_text.find('{')
                end_idx = generated_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = generated_text[start_idx:end_idx]
                    parsed_data = json.loads(json_str)

                    # Extract problem and answer
                    if 'problem' in parsed_data and 'answer' in parsed_data:
                        parsed_problem = {
                            'problem': parsed_data['problem'],
                            'answer': parsed_data['answer']
                        }
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

            # If JSON parsing succeeded, use the parsed data and count by action
            if parsed_problem:
                new_problems.append(parsed_problem)
                parse_success_count += 1
                original_idx = i // num_variations_per_problem
                if original_idx < len(original_problems):
                    action = original_problems[original_idx].get('action', 'upgrade')
                    if action == 'upgrade':
                        upgrade_success_count += 1
                    elif action == 'degrade':
                        degrade_success_count += 1
            else:
                # Fallback
                new_problems.append({
                    'problem': "",
                    'answer': ''
                })
            # Log the first case as an example
            if i == 0:
                pprint(f"[generate prompt]: {repeated_prompts[0]}")
                pprint(f"[generate output]: {generated_text}")
                if parsed_problem:
                    pprint(f"[parsed problem]: {parsed_problem['problem']}")
                    pprint(f"[parsed answer]: {parsed_problem['answer']}")
                else:
                    pprint("FAILED - falling back to raw text")
        pprint(f"parse success:{parse_success_count},batch_size:{batch_size}")
        return new_problems, upgrade_success_count, degrade_success_count
