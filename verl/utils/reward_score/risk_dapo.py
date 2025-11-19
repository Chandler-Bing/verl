import json
import re
import math
import numpy as np

def extract(solution_str):
    json_str = re.findall(r'```json(.*?)```', solution_str, re.DOTALL)
    extract = json.loads(json_str[-1])
    risk_score = float(extract.get('risk_score'))
    return risk_score

def compute_score(solution_str: str,
                  ground_truth: str,
                  extra_info: dict,
                  rank_reward: float,
                  cur_step:int
                  ) -> dict:

    # Limit solution length for efficiency
    solution_str = solution_str  # The longest answer in MATH-500 has 159 characters

    try:
        pred = extract(solution_str)
    except Exception as e:
        print(f'计算reward时出错: {str(e)}')
        pred = None

    # no answer no reward
    if pred is None or pred > 100 or pred < 0:
        return {
        "score": -2,
        "acc": -2,
        "pred": "",
        "details": {"pred":pred}
    }

    else:
        # pass
        # pred = round(pred,4)
        # pred = float(pred) / 100
        # mob3d30 = int(extra_info['mob3d30'])
        # reward = rank_reward

    #   单奖励方案，加权
        pred = round(pred,4)
        pred = float(pred) / 100
        # if pred == 0:
        #     pred = 0.01
        # if pred == 1:
        #     pred = 0.99
        mob3d30 = int(extra_info['mob3d30'])
        xgb = float(ground_truth) / 100
        # reward_xgb = 1 - 2*(pred - xgb) ** 2 # [-1,1]
        # reward_mob3d30 = 1 - 2*(pred - mob3d30) ** 2 # [-1,1]
        # #reward_mob3d30 = mob3d30*math.log(pred) + (1-mob3d30)*math.log(1-pred) # [-inf,0]
        # #reward = reward_xgb*0.3 + reward_mob3d30*0.4 + rank_reward*0.3
        # reward = reward_xgb*0.5 + reward_mob3d30*0.5
        reward = rank_reward

    # #  多奖励融合方案
    #     xgb = int(float(ground_truth))
    #     xgb_bin_a = (xgb // 10 * 10)
    #     xgb_bin_b = (xgb_bin_a + 10) / 100
    #     xgb_bin_a = xgb_bin_a / 100
    #     xgb = float(xgb) / 100
    #     mob3d30 = int(extra_info['mob3d30'])
    #     pred = round(pred)
    #     pred = float(pred) / 100

    #     reward_brier = 1 - 2*(pred - mob3d30)**2 # [-1,1]
    #     # reward_xgb_bin =  1 if pred >= xgb_bin_a and pred <= xgb_bin_b else -1 #[-1,0]
    #     pred_bin = (int(pred*100) // 10 * 10) / 100
    #     reward_xgb_bin =  1 - 2 * (pred_bin - xgb_bin_a) ** 2

    # reward =  reward_brier + reward_xgb_bin


    #   多奖励融合方案
    #     xgb = int(float(ground_truth))
    #     xgb_bin_a = (xgb // 10 * 10)
    #     xgb_bin_b = (xgb_bin_a + 10) / 100
    #     xgb_bin_a = xgb_bin_a / 100
    #     xgb = float(xgb) / 100
    #     mob3d30 = int(extra_info['mob3d30'])
    #     pred = round(pred)
    #     pred = float(pred) / 100
    #     if pred == 0:
    #         pred = 0.01
    #     if pred == 1:
    #         pred = 0.99

    #     reward_ce = mob3d30*math.log(pred) + (1-mob3d30)*math.log(1-pred) # [-inf,0]
    #     reward_ce = np.clip(reward_ce / 5.0, -1.0, 0.0) # [-1,0]
    #     reward_ce = reward_ce * 2 + 1 # [-1,1]
    #     reward_brier = 1 - 2*(pred - mob3d30)**2 # [-1,1]
    #     # reward_xgb_bin =  1 if pred >= xgb_bin_a and pred <= xgb_bin_b else -1 #[-1,0]
    #     pred_bin = (int(pred*100) // 10 * 10) / 100
    #     reward_xgb_bin =  1 - 2 * (pred_bin - xgb_bin_a) ** 2

    #     #print(pred, xgb_bin_a,xgb_bin_b)
    #     reward_rank = rank_reward #[-1,1]
    #     xgb_confidence = abs(xgb - 0.5) # [0,1]
    #     #print(f'{xgb_confidence=}')
    #     if xgb_confidence < 0.15:
    #         if (mob3d30 == 1 and pred > xgb) or (mob3d30 == 0 and pred < xgb):
    #             reward_explore = 2*abs(pred - xgb) # [-1,0]
    #         else:
    #             reward_explore = 0
    #     else:
    #         reward_explore = 0

    # alpha = cur_step / 200
    # if alpha >= 1:
    #     alpha = 1
    # reward = reward_ce * (0.1+0.2*alpha) + reward_brier * (0.1+0.2*alpha) + reward_xgb_bin * (0.6-0.4*alpha) + reward_rank * 0.1 + reward_explore * 0.1

    acc = reward
    #acc = pred

    return {
        "score": reward,
        "acc": acc,
        "pred": pred if pred is not None else "" ,
        "details": {
            "pred": pred,
            "mob3d30": mob3d30,
            "ground_truth": ground_truth,
            #"reward_xgb": reward_xgb,
            #"reward_mob3d30": reward_mob3d30,
            # "reward_ce": reward_ce,
            #"reward_brier": reward_brier,
            #"reward_xgb_bin": reward_xgb_bin,
            "reward_rank": rank_reward,
            # "reward_explore": reward_explore,
            # "cur_step": cur_step,
            # "alpha": alpha
        }
    }

if __name__ == '__main__':
    s = """
    ```json
    {
        "risk_score": 72
    } 
    ```
    """
    ground_truth = '51'
    print(compute_score(s,ground_truth,cur_step=2,extra_info={'mob3d30':1},rank_reward=1))
