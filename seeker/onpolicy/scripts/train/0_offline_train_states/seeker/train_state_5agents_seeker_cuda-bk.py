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

setproctitle.setproctitle("test")
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







######################### parameter steup begins ----------------

# root_path = '/home/jqruan/data/clustering/on-policy-seeker-161-20230320/onpolicy/scripts/train/results_ours/seeker/5agents/5agents_mappo/seed_1_run1/models/'
# root_path = '/home/jqruan/data/clustering/on-policy-seeker-161-20230320/onpolicy/scripts/train/results_ours/seeker/5agents/5agents_rmappo/seed_3_run1/models/'
# root_path = '/home/jqruan/data/clustering/on-policy-seeker-161-20230320/onpolicy/scripts/train/results_ours/seeker/8agents/8agents_rmappo/seed_1_run1/models/'
# root_path = '/home/jqruan/data/clustering/on-policy-seeker-161-20230320/onpolicy/scripts/train/results_ours/seeker/12agents/12agents_rmappo/seed_1_run1/models/'
root_path = '/data/jqruan/clustering/on-policy-seeker-161-20230320/onpolicy/scripts/train/results_ours/seeker/5agents/5agents_mappo/seed_1_run1/models/'


# input_size, hidden_size, output_size = 150, 64, 128  ### 
# input_size, hidden_size, output_size = 200, 64, 128
# input_size, hidden_size, output_size = 275, 64, 128
# input_size, hidden_size, output_size = 375, 64, 128

# input_size, hidden_size, output_size = 150, 128, 64
input_size, hidden_size, output_size = 200, 128, 64

add_data_bk_path = '/data/jqruan/data/clustering/seeker/mappo/5agents/data_all/' ## 162
path_files = [
    'o_a_r_o_prime_done_s_s_prime_u-slices_0-num_5000_random.npy',
    'o_a_r_o_prime_done_s_s_prime_u-slices_1-num_5000_random.npy',
    'o_a_r_o_prime_done_s_s_prime_u-slices_2-num_3000_random.npy',
    'o_a_r_o_prime_done_s_s_prime_u-slices_3-num_3000_random.npy',
    'o_a_r_o_prime_done_s_s_prime_u-slices_4-num_3000_random.npy',
    'o_a_r_o_prime_done_s_s_prime_u-slices_5-num_3000_random.npy'
]

# s, 1, 1, s, 1, 4s, 4s, 4 = 247
num_agents = 5
action_shape = 5
global_size = input_size * num_agents
joint_action_shape = action_shape * num_agents
batch_size = 128
USE_MI = True
SHUFFLE = True

# USE_MI = True
# SHUFFLE = False

# USE_MI = False
# SHUFFLE = True

# USE_MI = False
# SHUFFLE = False

# USE_GLOBAL = True
# USE_GLOBAL = False

# cuda0 : 1 1 ; cuda1: 1 0; cuda2 : 0 1; cuda3 : 0 0 

# USE_MI = 
# transition_model_type
# transition_model_type = 'deterministic'  #
# transition_model_type = 'probabilistic'  #
transition_model_type = 'ensemble'  #

# For more simple logs
cur_time = datetime.now() + timedelta(hours=0)
# log_dir = root_path + 'train_state_logs/emb_' + str(output_size) + '_global_addmlp/' + transition_model_type + '_use_MI_'  if USE_MI else \
#     root_path + 'train_state_logs/emb_' + str(output_size) + '_global_addmlp/' + transition_model_type + '_' 

log_dir = root_path + 'train_state_logs/emb_' + str(output_size) + '/' + transition_model_type + '_use_MI_'  if USE_MI else \
    root_path + 'train_state_logs/emb_' + str(output_size) + '/' + transition_model_type + '_' 


log_dir += '_shuffle_' if SHUFFLE else "_"


log_dir += cur_time.strftime("[%m-%d]%H.%M.%S")
writer = SummaryWriter(logdir=log_dir)
######################### parameter steup end ----------------

bis_agent = BisimAgent(input_size, hidden_size, output_size, action_shape, transition_model_type).cuda()

bis_agent_global = BisimAgent_GLOBAL(global_size, hidden_size, output_size, joint_action_shape, transition_model_type).cuda()


train_step = 0
for _ in range(500):
    
    for i in path_files:
        i = add_data_bk_path + i
        data_np = np.load(i, allow_pickle=True) 
        # 读取数据
        # datas = DataLoader(torch.from_numpy(data_np), batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
        data_np = data_np.reshape(-1, data_np.shape[-1])
        datas = DataLoader(torch.from_numpy(data_np), batch_size=batch_size, shuffle=SHUFFLE, drop_last=True, num_workers=2)

        for ind, data in enumerate(tqdm(datas)):
            if data.shape[0] < batch_size:
                continue
                
            obs = data[:, :input_size].float().cuda() # obs 
            act = data[:, [input_size]].long().cuda() # act
            # print(act)
            # import pdb
            # pdb.set_trace()
            rew = data[:, [input_size+1]].float().cuda() # rew
            next_obs = data[:, input_size+1:input_size+1+input_size].float().cuda() # next_obs
            
            done_ = data[:, [input_size+1+input_size]].float().cuda()
            action = torch.zeros(batch_size, action_shape).cuda().scatter_(1, act, 1) ## transform to one hot 
            
            if USE_GLOBAL:
                global_s = data[:, -global_size-global_size-num_agents:-global_size-num_agents].float().cuda() # global_s
                next_global_s = data[:, -global_size-num_agents:-num_agents].float().cuda() # next_global_s
                joint_act = data[:, -num_agents:].long().cuda()
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

            
            if USE_GLOBAL:
                global_h = bis_agent_global.encoder(global_s)
                pred_next_latent_mu_global, pred_next_latent_sigma_global = bis_agent_global.transition_model(torch.cat([global_h, joint_action], dim=1))
                pred_next_latent_mu_global_, pred_next_latent_sigma_global_ = pred_next_latent_mu_global.detach(), pred_next_latent_sigma_global.detach()
                ######################### p 去拟合 q，即：local 去拟合 global的
                ######################### 数据分布 归一化之后 近似成概率分布
                ######################### kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum') 
                if USE_MI:
                    ne, bs, msg_d = pred_next_latent_mu_global.shape
                    t = 1
                    
                    da = D.Normal(pred_next_latent_mu[:,None].permute(1,0,2,3).reshape(t, 1, bs*ne, msg_d), pred_next_latent_sigma[:,None].permute(1,0,2,3).reshape(t, 1, bs*ne, msg_d).exp())
                    # print(torch.min(pred_next_latent_mu_global), torch.max(pred_next_latent_mu_global))
                    # print(torch.min(pred_next_latent_sigma_global), torch.max(pred_next_latent_sigma_global))
                    db = D.Normal(pred_next_latent_mu_global_[:,None].permute(1,0,2,3).reshape(t, 1, bs*ne, msg_d), \
                        pred_next_latent_sigma_global_[:,None].permute(1,0,2,3).reshape(t, 1, bs*ne, msg_d).exp())
                    # da db : Normal(loc: torch.Size([1, 1, 640, 16]), scale: torch.Size([1, 1, 640, 16])) # t,1,bs*ne, msg_d
                    z=da.sample() # torch.Size([1, 1, 640, 16])
                    
                    z=z.reshape(t, bs*ne, 1, msg_d) 
                    logits = db.log_prob(z)  # torch.Size([1, 640, 640, 16])
                    logits = logits.sum(-1)  # torch.Size([1, 640, 640])
                    
                    ince = D.Categorical(logits=logits) # Categorical(logits: torch.Size([1, 640, 640]))
                    inds = torch.arange(bs*ne).cuda().unsqueeze(0).repeat(t,1) # torch.Size([1, 640])
                    
                    mi_loss = - 0.01 * ince.log_prob(inds).mean() ## args.mi_weight = 0.01
                    
                
                kl_loss = F.kl_div(h.softmax(dim=-1).log(), global_h.softmax(dim=-1).detach(), reduction='sum') 
            
            

            ##### update optimizer
            if USE_GLOBAL and USE_MI:
                total_loss = 0.5 * encoder_loss + transition_reward_loss + kl_loss + mi_loss
                writer.add_scalar('mi_loss', mi_loss.item(), train_step)
                writer.add_scalar('kl_loss', kl_loss.item(), train_step)
                
            elif USE_GLOBAL:
                total_loss = 0.5 * encoder_loss + transition_reward_loss + kl_loss
                writer.add_scalar('kl_loss', kl_loss.item(), train_step)
                
            else:
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


            
            if USE_GLOBAL:
                """
                ------------------------------------------------------------------
                the upfate of global 
                ------------------------------------------------------------------
                """
                #### update_encoder
                # h = bis_agent_global.encoder(global_s)
                # batch_size_now = h.shape[0]
                perm = np.random.permutation(batch_size)
                global_h2 = global_h[perm]
                
                with torch.no_grad():
                    pred_next_latent_mu1, pred_next_latent_sigma1 = pred_next_latent_mu_global.clone(), pred_next_latent_sigma_global.clone()
                    # bis_agent_global.transition_model(torch.cat([h, action], dim=1))
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

                z_dist = F.smooth_l1_loss(global_h, global_h2, reduction='none')
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
                
                if pred_next_latent_sigma_global is None:
                    pred_next_latent_sigma_global = torch.ones_like(pred_next_latent_sigma_global)
                    
        
                next_h = bis_agent_global.encoder(next_global_s)
                diff = (pred_next_latent_mu_global - next_h.detach()) / pred_next_latent_sigma_global
                transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma_global))
                
                pred_next_latent = bis_agent_global.transition_model.sample_prediction(torch.cat([global_h, joint_action], dim=1))
                pred_next_reward = bis_agent_global.reward_decoder(pred_next_latent)
                reward_loss = F.mse_loss(pred_next_reward, rew)
                transition_reward_loss = transition_loss + reward_loss

                ##### update optimizer
                total_loss = 0.5 * encoder_loss + transition_reward_loss
                bis_agent_global.encoder_optimizer.zero_grad()  # [torch.sum(torch.isnan(p.data)) for p in bis_agent_global.encoder.parameters()]
                bis_agent_global.decoder_optimizer.zero_grad()  # [torch.sum(torch.isnan(p.data)) for p in bis_agent_global.reward_decoder.parameters()]
                # with autograd.detect_anomaly():
                #     total_loss.backward()
                bis_agent_global.encoder_optimizer.step()
                bis_agent_global.decoder_optimizer.step()
                train_step += 1
                
                writer.add_scalar('global_encoder_loss', encoder_loss.item(), train_step)
                writer.add_scalar('global_transition_loss', transition_loss.item(), train_step)
                writer.add_scalar('global_reward_loss', reward_loss.item(), train_step)
                writer.add_scalar('global_total_loss', total_loss.item(), train_step)

            
            # save_path = if USE_MI
            save_path = log_dir + '/models_' + transition_model_type # models_ensemble
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
             
            if train_step % 20000 == 0:
                torch.save(bis_agent.state_dict(), '%s/bis_agent._%s.pt' % (save_path, train_step))
                if USE_GLOBAL:
                    torch.save(bis_agent_global.state_dict(), '%s/bis_agent_global._%s.pt' % (save_path, train_step))
            
    # torch.save(bis_agent.state_dict(), '%s/bis_agent._%s.pt' % (root_path + 'models', train_step))
    torch.save(bis_agent.state_dict(), '%s/bis_agent._%s.pt' % (save_path, train_step))
    if USE_GLOBAL:
        torch.save(bis_agent_global.state_dict(), '%s/bis_agent_global._%s.pt' % (save_path, train_step))
                
        
        
        
        
        
        
        
         
         
         
        
        
        
        
    
    
