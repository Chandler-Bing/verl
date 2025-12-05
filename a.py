import torch
import numpy as np


reward1 = [1,2,3,4,5,6,7,8]
reward2 = [1,100,3,400,5,600,7,800]


reward1 = torch.tensor(reward1)
adv1 = (reward1 - reward1.mean()) / (reward1.std() + 1e-8)
