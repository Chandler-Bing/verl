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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
import re
import json
import os
import decimal


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False,val = False,step = 0,config = None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        #reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        all_reward = {
            "total_reward": torch.zeros_like(data.batch["responses"], dtype=torch.float32),
        }
        reward_extra_info = defaultdict(list)

        solution_strs = []
        # ground_truths = []
        mob3d30s = []
        uids = []

        def extract(solution_str):
            try:
                json_str = re.findall(r'```json(.*?)```', solution_str, re.DOTALL)
                extract_str = json.loads(json_str[-1])
                risk_score = float(extract_str.get('risk_score'))
                assert 0 <= risk_score <= 100, f"Risk score {risk_score} is out of range [0, 100]"
                return risk_score
            except:
                return -100

        def rank_reward(solution_strs, mob3d30s, n):
            solution_strs = [extract(solution_str) / 100 for solution_str in solution_strs]
            # solution_strs = [float(solution_str) / 100 for solution_str in solution_strs]
            mob3d30s = [int(mob3d30) for mob3d30 in mob3d30s]

            print(f'{len(solution_strs)=}, {len(mob3d30s)=},{mob3d30s.count(1)=},{mob3d30s.count(0)=}')

            rewards = [0] * len(solution_strs)
            counts = [0] * len(solution_strs)
            group_avg = []
            for i in range(0, len(solution_strs), n):
                group_avg.append(sum(solution_strs[i:i + n]) / n)
            for i in range(len(solution_strs)):
                sample_mob3d30 = mob3d30s[i]
                sample_solution = solution_strs[i]
                group_index = i // n
                for j in range(len(group_avg)):
                    group_mob3d30 = mob3d30s[j * n]
                    if j == group_index or sample_mob3d30 == group_mob3d30:
                        continue
                    if sample_mob3d30 > group_mob3d30:
                        if sample_solution > group_avg[j]:
                            rewards[i] += 1
                    elif sample_mob3d30 < group_mob3d30:
                        if sample_solution < group_avg[j]:
                            rewards[i] += 1
                    else:
                        pass
                    counts[i] += 1

            rewards = [rewards[i] / counts[i] if counts[i] > 0 else 0 for i in range(len(rewards))]
            return rewards

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            solution_strs.append(response_str)
            # ground_truths.append(ground_truth)
            mob3d30s.append(extra_info['mob3d30'])
            # uids.append(data_item.non_tensor_batch['uid'])
            uids.append(data_item.non_tensor_batch['extra_info']['biz_no'])

        def if_sort_by_order(uids, n):
            for i in range(0, len(uids), n):
                group = uids[i:i + n]
                if len(set(group)) != 1:
                    return False
            return True

        if not val:
            assert if_sort_by_order(uids, config.actor_rollout_ref.rollout.n), f"uids are not sorted by pairs: {uids}"
            rank_rewards = rank_reward(solution_strs, mob3d30s, config.actor_rollout_ref.rollout.n)
        else:
            rank_rewards = [0 for i in range(len(solution_strs))]


        already_print_data_sources = {}
        y_trues = []
        y_preds = []
        prompts = []
        old_responses = []
        responses = []
        extra_infos = []
        details = []
        print(f'calulate reward for {len(data)=} samples')
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            old_response_ids = data_item.batch["old_responses"]
            old_valid_response_length = data_item.batch["old_attention_mask"][prompt_length:].sum()
            old_valid_response_ids = old_response_ids[:old_valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            old_response_str = self.tokenizer.decode(old_valid_response_ids, skip_special_tokens=True)
            prompts.append(prompt_str)
            responses.append(response_str)
            old_responses.append(old_response_str)

            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            extra_infos.append(extra_info)

            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})

            extra_info["rollout_reward_scores"] = rollout_reward_scores

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                rank_reward=rank_rewards[i],
                cur_step=step
            )
            details.append(result["details"])
            if result['pred'] is not None and result['pred'] != "":
                y_trues.append(float(ground_truth) / 100)
                y_preds.append(float(result['pred']))

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score
            for k, v in result['details'].items():
                if 'reward_' in k:
                    if k not in all_reward.keys():
                        all_reward[k] = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
                    all_reward[k][i, valid_response_length - 1] = v

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            all_reward['total_reward'][i, valid_response_length - 1] = reward
            #reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            if val:
                return prompts, responses, extra_infos, details, y_trues, y_preds, {
                    "reward_tensor": all_reward,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                rollout_txt_path = f'/data/oceanus_share/boruipeng/github/risk-val-logs/{config.trainer.experiment_name}/rollout_step_{step}.txt'
                os.makedirs(os.path.dirname(rollout_txt_path), exist_ok=True)
                v = 1
                while os.path.exists(rollout_txt_path):
                    rollout_txt_path = f'{rollout_txt_path}.{v}'
                    v += 1
                print(f'saving rollout results to {rollout_txt_path}')
                with open(rollout_txt_path, 'w', encoding='utf-8') as f:
                    for p, r, e, d,o_r in zip(prompts, responses, extra_infos, details,old_responses):
                        e = dict(e)
                        for k, v in e.items():
                            if type(v) is decimal.Decimal:
                                # print(f'befor,{type(v)}')
                                v = float(v)
                                # print(f'after,{type(v)}')
                                e[k] = v
                        f.write(
                            json.dumps({'prompt': p, 'response': r, 'old_response':o_r ,'extra_info': e, "details": d}, ensure_ascii=False))
                        f.write('\n')
                print(f'saving done....')
                return {
                    "reward_tensor": all_reward,
                    "reward_extra_info": reward_extra_info,
                }
        else:
            return all_reward