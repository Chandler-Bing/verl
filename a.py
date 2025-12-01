import torch
import numpy as np

a = torch.tensor([1, 2, 3,np.nan])
print(a.sum(-1))
print(torch.min(a).detach().item())