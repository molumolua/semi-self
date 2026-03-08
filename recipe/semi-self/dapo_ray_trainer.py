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

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
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
        num_gen_batches = 0
        

        # Get train problems directly from the raw dataframe (not processed data)
        # since InMemoryRLHFDataset expects raw data with 'prompt' field
        train_problems = []
        for i in range(min(self.config.data.train_batch_size, len(self.train_dataset))):
            # Get raw data from dataframe, not processed data from __getitem__
            problem = dict(self.train_dataset.dataframe[i])
            train_problems.append(problem)


        inmemory_dataloader=self.createInmemoryDataLoader(train_problems)

        next_problem_id = self.config.data.train_batch_size


        for epoch in range(self.config.trainer.total_epochs):
            # only one batch for InMemoryRLHFDataset
            for batch_dict in inmemory_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps
                metrics= self.train_batch(batch_dict,prev_step_profile,curr_step_profile,timing_raw)
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

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

            # update train_problems and inmemory_dataloader
            train_problems,next_problem_id = self.update_problems_simple(train_problems,next_problem_id)
            inmemory_dataloader=self.createInmemoryDataLoader(train_problems)

            # Log next_problem_id as a metric
            problem_id_metrics = {"train/next_problem_id": next_problem_id}
            logger.log(data=problem_id_metrics, step=self.global_steps)
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)

    
    def train_batch(self, batch_dict, prev_step_profile, curr_step_profile,  timing_raw):
        metrics = {}

        with marked_timer("start_profile", timing_raw):
            self._start_profiling(
                not prev_step_profile and curr_step_profile
                if self.config.global_profiler.profile_continuous_steps
                else curr_step_profile
            )

        new_batch: DataProto = DataProto.from_single_dict(batch_dict)
        num_gen_batches += 1
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
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            # Log rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
            
            return metrics
        
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
    def update_problems(self, original_problems, num_variations_per_problem=8):
        """
        Generate new problems by upgrading or downgrading the difficulty of original problems.

        Args:
            original_problems: List of dicts, each containing 'problem', 'answer', and 'difficulty_label'
                             where difficulty_label is either 'upgrade' (make harder) or 'downgrade' (make easier)
            num_variations_per_problem: Number of new problems to generate per original problem (M)

        Returns:
            List of dicts, each containing 'problem' and 'answer' keys for the generated problems
        """
        import torch
        from verl import DataProto
        from tensordict import TensorDict
        import numpy as np
        import json

        # Create prompts for each original problem
        prompts = []
        for problem_data in original_problems:
            problem = problem_data['problem']
            answer = problem_data['answer']
            label = problem_data['difficulty_label']

            if label == 'upgrade':
                # Create prompt to make the problem harder
                prompt = f"""You are given a problem in JSON format. Create a more difficult version of this problem.
Keep the core concept the same but increase the complexity, add more constraints, or require deeper understanding.

Input:
{{
"problem": "{problem}",
"answer": "{answer}"
}}

Generate a harder version and output in the same JSON format:
{{
"problem": "your harder problem here",
"answer": "corresponding answer here"
}}
"""
            elif label == 'downgrade':
                # Create prompt to make the problem easier
                prompt = f"""You are given a problem in JSON format. Create a simpler version of this problem.
Keep the core concept the same but reduce the complexity, simplify the requirements, or make it more straightforward.

Input:
{{
"problem": "{problem}",
"answer": "{answer}"
}}

Generate an easier version and output in the same JSON format:
{{
"problem": "your easier problem here",
"answer": "corresponding answer here"
}}

Output:"""
            else:
                raise ValueError(f"Invalid difficulty_label: {label}. Must be 'upgrade' or 'downgrade'")

            prompts.append(prompt)

        # Repeat each prompt M times
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * num_variations_per_problem)

        # Tokenize prompts
        tokenized_prompts = self.tokenizer(repeated_prompts, return_tensors='pt', padding=True, truncation=True)

        # Create attention mask
        attention_mask = tokenized_prompts['attention_mask']

        # Create position ids
        position_ids = torch.arange(attention_mask.size(1), dtype=torch.long).unsqueeze(0).expand(attention_mask.size(0), -1)

        # Create TensorDict batch
        batch = TensorDict({
            "input_ids": tokenized_prompts['input_ids'],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=len(repeated_prompts))

        # Create DataProto
        gen_batch = DataProto(
            batch=batch,
            non_tensor_batch={
                "uid": np.arange(len(repeated_prompts)),
                "data_source": np.array(["update_problems"] * len(repeated_prompts)),
            },
            meta_info={
                "do_sample": True,
                "temperature": 0.8,  # Add some creativity for problem generation
                "max_new_tokens": 512,
                "min_new_tokens": 50,
            }
        )

        # Generate new problems using the policy model
        generated_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        # Extract the generated text
        generated_sequences = generated_output.batch["input_ids"]  # This includes both prompt and generated text
        attention_masks = generated_output.batch["attention_mask"]

        new_problems = []
        batch_size = len(repeated_prompts)

        for i in range(batch_size):
            # Find where the prompt ends and generation begins
            prompt_length = attention_mask[i].sum().item()
            generated_tokens = generated_sequences[i, prompt_length:]

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
                # If JSON parsing fails, try to extract manually
                pass

            # If JSON parsing succeeded, use the parsed data
            if parsed_problem:
                new_problems.append(parsed_problem)
            else:
                # Fallback: use the raw text as problem, set answer as empty
                # This handles cases where JSON parsing failed
                new_problems.append({
                    'problem': generated_text,
                    'answer': ''
                })

        return new_problems

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
        return updated_problems, current_id
