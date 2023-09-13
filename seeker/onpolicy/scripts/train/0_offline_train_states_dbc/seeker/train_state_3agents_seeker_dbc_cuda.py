import numpy as np
import torch

from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import importlib
new_imp = importlib.import_module('onpolicy.scripts.train.0_offline_train_states_dbc.transition_model')

from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
import math
import torch.distributions as D
import os

import setproctitle

setproctitle.setproctitle("test")
USE_CUDA = True

class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP,self).__init__()    # 
        self.fc1 = torch.nn.Linear(input_size, hidden_size)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(hidden_size, output_size)  # 第二个隐含层
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

######################### parameter steup begins ----------------

# root_path = '/data2/xingdp/wangkaishen/bq/WeWar6-rjq-1108/wewar/bis_state_mapping_rjq/data-020/'
# path_files = [
# 's_a_r_s_prime-slices_0-num_1488022_add_s.npy', 's_a_r_s_prime-slices_1-num_1362676_add_s.npy',
# 's_a_r_s_prime-slices_2-num_1721191_add_s.npy', 's_a_r_s_prime-slices_3-num_1747981_add_s.npy',
# 's_a_r_s_prime-slices_4-num_1720319_add_s.npy', 's_a_r_s_prime-slices_5-num_1705104_add_s.npy'
# ]

# root_path = '/home/jqruan/data/clustering/on-policy-mpe-161-20220301/onpolicy/scripts/results/MPE/simple_spread/rmappo/3agents/run1/models/'
# add_data_bk_path = '/data2/jqruan/data/clustering/MPE/simple_spread/rmappo/3agents/run1/models/' ## 162
# path_files = [
#     'o_a_r_o_prime_done_s_s_prime_u-slices_1-num_10000.npy', 'o_a_r_o_prime_done_s_s_prime_u-slices_2-num_10000.npy',
#     'o_a_r_o_prime_done_s_s_prime_u-slices_3-num_10000.npy', 'o_a_r_o_prime_done_s_s_prime_u-slices_4-num_10000.npy',
#     'o_a_r_o_prime_done_s_s_prime_u-slices_5-num_10000.npy' ]


root_path = '/data/jqruan/clustering/on-policy-seeker-161-20230320/onpolicy/scripts/train/results_ours/seeker/3agents/3agents_mappo/seed_1_run1/models/'

add_data_bk_path = '/data/jqruan/data/clustering/seeker/mappo/3agents/' ## 162
path_files = [
    'o_a_r_o_prime_done_s_s_prime_u-slices_1-num_5000_random.npy',
    'o_a_r_o_prime_done_s_s_prime_u-slices_2-num_5000_random.npy',
    'o_a_r_o_prime_done_s_s_prime_u-slices_3-num_5000_random.npy',
    'o_a_r_o_prime_done_s_s_prime_u-slices_4-num_5000_random.npy',
    'o_a_r_o_prime_done_s_s_prime_u-slices_5-num_5000_random.npy',
]

# s, 1, 1, s, 1, 4s, 4s, 4
# s, 1, 1, s, 1, 4s, 4s, 4 = 247
num_agents = 3
action_shape = 5
input_size, hidden_size, output_size = 150, 128, 64

global_size = input_size * num_agents
joint_action_shape = action_shape * num_agents
batch_size = 128

USE_MI = False
SHUFFLE = False
## cuda0 : 1 1 ; cuda1: 1 0; cuda2 : 0 1; cuda3 : 0 0 

# USE_MI = 
# transition_model_type
# transition_model_type = 'deterministic'  #
# transition_model_type = 'probabilistic'  #
transition_model_type = 'ensemble'  #

# For more simple logs
cur_time = datetime.now() + timedelta(hours=0)
log_dir = root_path + 'train_state_logs_dbc/' + transition_model_type + '/'
log_dir += cur_time.strftime("[%m-%d]%H.%M.%S")
writer = SummaryWriter(logdir=log_dir)
######################### parameter steup end ----------------

bis_agent = BisimAgent(input_size, hidden_size, output_size, action_shape, transition_model_type).cuda()


train_step = 0
for i in path_files:
    # i = root_path + 'data/' + i
    i = add_data_bk_path + i
    data_np = np.load(i, allow_pickle=True) 
    # 读取数据
    # datas = DataLoader(torch.from_numpy(data_np), batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    data_np = data_np.reshape(-1, data_np.shape[-1])
    datas = DataLoader(torch.from_numpy(data_np), batch_size=batch_size, shuffle=SHUFFLE, drop_last=True, num_workers=2)
    
    for _ in range(10):
        for ind, data in enumerate(tqdm(datas)):
            if data.shape[0] < batch_size:
                continue
            # print(i)
            # if torch.sum(np.isnan(data)) > 0:
            #     import pdb
            #     pdb.set_trace()
            # if torch.sum(np.isinf(data)) > 0:
            #     import pdb
            #     pdb.set_trace()
                
            obs = data[:, :input_size].float().cuda() # obs
            act = data[:, [input_size]].long().cuda() # act
            rew = data[:, [input_size+1]].float().cuda() # rew
            next_obs = data[:, input_size+1:input_size+1+input_size].float().cuda() # next_obs
            
            global_s = data[:, -global_size-global_size-num_agents:-global_size-num_agents].float().cuda() # global_s
            next_global_s = data[:, -global_size-num_agents:-num_agents].float().cuda() # next_global_s
            joint_act = data[:, -num_agents:].long().cuda()
            
            action = torch.zeros(batch_size, action_shape).cuda().scatter_(1, act, 1) ## transform to one hot 
            joint_action = torch.cat(([torch.zeros(batch_size, action_shape).cuda().scatter_(1, joint_act[:,[i]], 1) for i in range(num_agents)]), 1)
            

            """
            ------------------------------------------------------------------
            the upfate of actor
            ------------------------------------------------------------------
            """
            #### update_encoder
            h = bis_agent.encoder(obs)
            # batch_size_now = h.shape[0]
            perm = np.random.permutation(batch_size)
            h2 = h[perm]
            
            with torch.no_grad():
                pred_next_latent_mu1, pred_next_latent_sigma1 = bis_agent.transition_model(torch.cat([h, action], dim=1))
                reward2 = rew[perm]
            if pred_next_latent_sigma1 is None:
                pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
            if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
                pred_next_latent_mu2 = pred_next_latent_mu1[perm]
                pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
            elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
                pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
                pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
            else:
                raise NotImplementedError

            z_dist = F.smooth_l1_loss(h, h2, reduction='none')
            r_dist = F.smooth_l1_loss(rew, reward2, reduction='none')
            
            if transition_model_type == '':
                transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none')
            else:
                transition_dist = torch.sqrt(
                    (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
                    (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
                )
            
            bisimilarity = r_dist + 0.99 * transition_dist
            encoder_loss = (z_dist - bisimilarity).pow(2).mean()
            
            #### update_transition_reward_model
            pred_next_latent_mu, pred_next_latent_sigma = bis_agent.transition_model(torch.cat([h, action], dim=1))
            if pred_next_latent_sigma is None:
                pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
                
    
            next_h = bis_agent.encoder(next_obs)
            diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
            transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
            
            pred_next_latent = bis_agent.transition_model.sample_prediction(torch.cat([h, action], dim=1))
            pred_next_reward = bis_agent.reward_decoder(pred_next_latent)
            reward_loss = F.mse_loss(pred_next_reward, rew)
            transition_reward_loss = transition_loss + reward_loss

            
            ##### update optimizer
            total_loss = 0.5 * encoder_loss + transition_reward_loss 
            bis_agent.encoder_optimizer.zero_grad()
            bis_agent.decoder_optimizer.zero_grad()
            total_loss.backward()
            bis_agent.encoder_optimizer.step()
            bis_agent.decoder_optimizer.step()
            train_step += 1

            writer.add_scalar('encoder_loss', encoder_loss.item(), train_step)
            writer.add_scalar('transition_loss', transition_loss.item(), train_step)
            writer.add_scalar('reward_loss', reward_loss.item(), train_step)
            
            writer.add_scalar('total_loss', total_loss.item(), train_step)

            
            # save_path = if USE_MI
            save_path = log_dir + '/models_' + transition_model_type # models_ensemble
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
             
            if train_step % 20000 == 0:
                torch.save(bis_agent.state_dict(), '%s/bis_agent._%s.pt' % (save_path, train_step))
            
    
    torch.save(bis_agent.state_dict(), '%s/bis_agent._%s.pt' % (save_path, train_step))
                
        
        


        # cd /data/jqruan/clustering/on-policy-seeker-161-20230320/onpolicy/scripts/train/0_offline_train_states_dbc/seeker/; conda activate pyg
        # CUDA_VISIBLE_DEVICES=0 python train_state_3agents_seeker_dbc_cuda.py
        
        
        
        
        
         
         
         
        
        
        
        
    
    
