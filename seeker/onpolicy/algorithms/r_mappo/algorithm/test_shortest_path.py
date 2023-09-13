# from pydpp.dpp import DPP
from onpolicy.algorithms.dpp.dpp import DPP
import torch
import numpy as np
X = torch.rand(3,16)
dpp = DPP(X)
dpp.compute_kernel(kernel_type = 'rbf', sigma= 0.4)   # use 'cos-sim' for cosine similarity
samples = dpp.sample()                   # samples := [1,7,2,5]
ksamlpes = dpp.sample_k(2)                # ksamples := [5,8,0]

print(ksamlpes) #