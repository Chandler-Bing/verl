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
import time
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
import re
import requests
import json
import copy
import concurrent.futures
from tqdm import tqdm
prompt_template = """你是一个风险评估审核专家。你的任务是判断给定的 Chain-of-Thought（COT）和最终决策（A/D）是否逻辑一致。输出格式为 **JSON**，内容包括判断结果和原因。请根据以下标准进行判断：
- **低风险 / 中低风险** → 应该决策为 **通过（A）**
- **高风险 / 中高风险** → 应该决策为 **拒绝（D）**
- **中等风险** → 既可以是 **A** 也可以是 **D**，但必须有合理解释，且风险得分必须在40-60之间。
请输出 JSON 格式如下：
```json
{{
  "consistency": "一致 / 不一致"
}}
```
需要判断的内容如下：
- Chain-of-Thought（COT）推理内容：
{cot}
- 最终决策（A/D）：
{decision}
- 风险得分（risk_score）：
{risk_score}
"""

def _generate(
        messages,
        model='x',
        url='http://192.168.194.137:40000/v1/chat/completions',
        temperature=0.6,
        max_tokens=128,
        top_p=1.0
):
    body = {
        'model': model,
        'messages': messages,
        'top_p': top_p,
        'top_k': -1,
        'temperature': temperature,
        'max_tokens': max_tokens,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    headers = {"content-type": "application/json; charset=UTF-8",
               'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjM2YzIzMWNhZDVlN2ZhMjMyNjVkMDQzNmEyY2UzOTdiMjlmOGZiYjAifQ.eyJqdGkiOiJMeVVFeTB5TDZfTXk5LVU0NHF3ZUFBIiwiaWF0IjoxNzQ0MjcyODIyLCJleHAiOjE3NDY4NjQ4MjIsIm5iZiI6MTc0NDI3Mjc2Miwic3ViIjoiQ2dFd0dncDJiMnhqWlc1bmFXNWwiLCJhdWQiOiJnY3ZyM3RxdmM3amFvdDZjdHZhMTAiLCJpc3MiOiJwYWFzdG9iLnBhYXMub2lkYyIsImFjY291bnRfaWQiOjIxMDQwNTE3MjcsIm5vbmNlIjoiY0FqYWxqYlBPVnRLR3dPaCJ9.hsQXvTuXcXkYd2Hz8d5e1eEl7dIl2Uyo_G5EuapTTTeARW_GpiddpTUonz65nxrX3GQQO4FWG4v3hSxlIv2lux3ZZjslA4TopeOM7xokfX2Axgk-4kaxhpizlxkmPE2CZYmg3yjryOn1b-MRuKaAMyAO79UsaZyMotcJNuYuGXRWI7E-d_Tc6BcCMjkTGiLuwM9sK5sTCLVJsBrvxWdl6_Xn94p_PpPbgmHeLAhq2kvdR7W7yGPxkGkKg62C52ZXDFUUx_AvCEbH3r-8-rid5IU0O_KS2RHsCGE8TjXZZGR7jGRPbn9vu6tLXiAqb11sKBK2GLhSPcq3JkPJ8L4LFA'}
    response = requests.post(url, json=body, headers=headers)
    try:
        if response.json().get("choices")[0].get("message", {}).get('reasoning_content', ''):
            return response.json().get("choices")[0].get("message", {}).get('reasoning_content', '') + '</think>' + \
                response.json().get("choices")[0].get("message", {}).get('content', '')
        else:
            return response.json().get("choices")[0].get("message", {}).get('content', '')
    except Exception as e:
        print(f'{response.status_code=}: {response.text=},{str(e)}')
        return str(e)


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

        def extract_solution(solution_str):
            try:
                json_str = re.findall(r'```json(.*?)```', solution_str, re.DOTALL)
                extract_str = json.loads(json_str[-1])
                decision = extract_str.get('decision')
                return decision
            except:
                return ""

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


        print(f'judge the score and cot consistency... start at {time.time()}')
        cots = [solution_str.split('</think>',1)[0].strip() for solution_str in solution_strs]
        cots = [cot.rsplit("\n", 1)[1].strip() for cot in cots]
        decisions = [extract_solution(solution_str) for solution_str in solution_strs]
        risk_scores = [extract(solution_str) for solution_str in solution_strs]
        all_messages_list = []
        for cot, decision,risk_score in zip(cots, decisions,risk_scores):
            prompt = prompt_template.format(cot=cot, decision=decision,risk_score=risk_score)
            all_messages_list.append([{"role": "user", "content": prompt}])

        all_results = [None for _ in range(len(all_messages_list))]
        with tqdm(total=len(all_messages_list)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4096) as executor:
                futures = {executor.submit(_generate, data): idx for idx, data in enumerate(all_messages_list)}
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    all_results[idx] = future.result()
                    pbar.update(1)
                    pbar.refresh()
        print(f'judge the score and cot consistency... done at {time.time()}')


        already_print_data_sources = {}
        y_trues = []
        y_preds = []
        prompts = []
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

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            prompts.append(prompt_str)
            responses.append(response_str)

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
                cur_step=step,
                consis=all_results[i],
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
                    for p, r, e, d in zip(prompts, responses, extra_infos, details):
                        e = dict(e)
                        for k, v in e.items():
                            if type(v) is decimal.Decimal:
                                # print(f'befor,{type(v)}')
                                v = float(v)
                                # print(f'after,{type(v)}')
                                e[k] = v
                        f.write(
                            json.dumps({'prompt': p, 'response': r, 'extra_info': e, "details": d}, ensure_ascii=False))
                        f.write('\n')
                print(f'saving done....')
                return {
                    "reward_tensor": all_reward,
                    "reward_extra_info": reward_extra_info,
                }
        else:
            return all_reward