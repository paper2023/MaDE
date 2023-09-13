import numpy as np
import torch

from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# from onpolicy.scripts.train.offline_train_states.transition_model import make_transition_model

import importlib
new_imp = importlib.import_module('onpolicy.scripts.train.0_offline_train_states.transition_model')


from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
import math
import torch.distributions as D
import os

import setproctitle

setproctitle.setproctitle("bq-abstract-6")
USE_CUDA = True

class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP,self).__init__()    # 
        self.fc1 = torch.nn.Linear(input_size, hidden_size)  # 第一个隐含层  
        self.bn = nn.BatchNorm1d(hidden_size)
                
        self.fc2 = torch.nn.Linear(hidden_size, output_size)  # 第二个隐含层
        self.bn_2 = nn.BatchNorm1d(hidden_size)
        
        # init the weights
        self.fanin_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.bias.data.uniform_(-3e-3, 3e-3)
    
    def fanin_init(self, tensor):
        size = tensor.size()
        if len(size) == 2:
            fan_in = size[0]
        elif len(size) > 2:
            fan_in = np.prod(size[1:])
        else:
            raise Exception("Shape must be have dimension at least 2.")
        bound = 1. / np.sqrt(fan_in)
        return tensor.data.uniform_(-bound, bound)
    
    
    def forward(self, din):   
        din = self.fc1(din)
        din = self.bn(din)
        din = F.relu(din)   # 使用 relu 激活函数
        
        dout = self.fc2(din)
        # dout = self.bn(din)
        
        # dout = F.relu(self.fc2(dout))
        # dout = F.relu(self.fc2(dout))
        return dout
    
class MLP_GLOBAL(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_GLOBAL,self).__init__()    # 
        # self.bn = nn.BatchNorm2d(input_size)
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = torch.nn.Linear(input_size, hidden_size)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)  # 第二个隐含层
        self.fc3 = torch.nn.Linear(hidden_size, output_size)  # 第二个隐含层
        # init the weights
        self.fanin_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.bias.data.uniform_(-3e-3, 3e-3)
    
    def fanin_init(self, tensor):
        size = tensor.size()
        if len(size) == 2:
            fan_in = size[0]
        elif len(size) > 2:
            fan_in = np.prod(size[1:])
        else:
            raise Exception("Shape must be have dimension at least 2.")
        bound = 1. / np.sqrt(fan_in)
        return tensor.data.uniform_(-bound, bound)
    
    
    def forward(self, din):    
        dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        dout = F.relu(self.fc2(dout))
        dout = self.fc3(dout)
        return dout



class BisimAgent(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self, input_size, hidden_size, output_size, action_shape, transition_model_type):
        super(BisimAgent,self).__init__()    # 
        
        
        self.encoder = MLP(input_size, hidden_size, output_size)
        self.reward_decoder = nn.Sequential(
                    nn.Linear(output_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1))
        self.transition_model = new_imp.make_transition_model(
            transition_model_type, output_size, action_shape
        )
        if USE_CUDA:
            self.transition_model.to(torch.device("cuda:0"))
     
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-5)
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=1e-5, weight_decay=0.0
        )







class BisimAgent_GLOBAL(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self, input_size, hidden_size, output_size, action_shape, transition_model_type):
        super(BisimAgent_GLOBAL,self).__init__()    # 
        
        
        self.encoder = MLP_GLOBAL(input_size, hidden_size, output_size)
        self.reward_decoder = nn.Sequential(
                    nn.Linear(output_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1))
        self.transition_model = new_imp.make_transition_model(
            transition_model_type, output_size, action_shape
        )
        if USE_CUDA:
            self.transition_model.to(torch.device("cuda:0"))
     
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-5)
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=1e-5, weight_decay=0.0
        )




