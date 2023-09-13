import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from onpolicy.algorithms.dpp.dpp import DPP

from torch.optim import Adam


import networkx as nx  # 导入 NetworkX 工具包
from sklearn.preprocessing import MinMaxScaler

import math
try:
    import PYNVML
    _pynvml_exist = True
except ModuleNotFoundError:
    _pynvml_exist = False

def remaining_memory(self):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if self._pynvml_exist:
      pynvml.nvmlInit()
      gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
      remaining = info.free
    else:
      remaining = torch.cuda.memory_allocated()
    return remaining
def euc_sim(a, b):
    return 2 * a @ b.transpose(-2, -1) -(a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]
def max_sim(a, b): ## a=X, b =centroids
    device = a.device.type
    batch_size = a.shape[0]
    sim_func = euc_sim

    if device == 'cpu':
      sim = sim_func(a, b)
      max_sim_v, max_sim_i = sim.max(dim=-1)
      return max_sim_v, max_sim_i
    else:
      if a.dtype == torch.float:
        expected = a.shape[0] * a.shape[1] * b.shape[0] * 4
      elif a.dtype == torch.half:
        expected = a.shape[0] * a.shape[1] * b.shape[0] * 2
      ratio = math.ceil(expected / remaining_memory())
      subbatch_size = math.ceil(batch_size / ratio)
      msv, msi = [], []
      for i in range(ratio):
        if i*subbatch_size >= batch_size:
          continue
        sub_x = a[i*subbatch_size: (i+1)*subbatch_size]
        sub_sim = sim_func(sub_x, b)
        sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
        del sub_sim
        msv.append(sub_max_sim_v)
        msi.append(sub_max_sim_i)
      if ratio == 1:
        max_sim_v, max_sim_i = msv[0], msi[0]
      else:
        max_sim_v = torch.cat(msv, dim=0)
        max_sim_i = torch.cat(msi, dim=0)
      return max_sim_v, max_sim_i
  
  
  
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Cluster(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.env_params = env_params
        # elbo_beta lr_cluster cluster_std_reg
        self.args = args
        self.n_mix = args.n_latent_landmarks
        self.z_dim = args.embed_size
        self.comp_mean = nn.Parameter(torch.randn(self.n_mix, self.z_dim) * np.sqrt(1.0 / self.n_mix))
        self.comp_logstd = nn.Parameter(torch.randn(1, self.z_dim) * 1 / np.e, requires_grad=True)
        self.mix_logit = nn.Parameter(torch.ones(self.n_mix), requires_grad=args.learned_prior)
        
        self.edges = None
        self.G = nx.Graph()  # 创建：空的 无向图
        
        # self.c_optim = Adam(self.parameters(), lr=args.lr_cluster)
    
    def component_log_prob(self, x):
        if x.ndim == 1:
            x = x.repeat(1, self.n_mix, 1)
        elif x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, self.n_mix, 1)
        assert x.ndim == 3 and x.size(1) == self.n_mix and x.size(2) == self.z_dim
        # comp_logstd = torch.sigmoid(self.comp_logstd) * (LOG_STD_MAX - LOG_STD_MIN) + LOG_STD_MIN
        comp_logstd = torch.clamp(self.comp_logstd, LOG_STD_MIN, LOG_STD_MAX)
        comp_dist = Normal(self.comp_mean, torch.exp(comp_logstd))
        comp_log_prob = comp_dist.log_prob(x).sum(dim=-1)  # (nbatch, n_mix)
        return comp_log_prob
    
    def forward(self, x, with_elbo=True):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        assert x.ndim == 2 and x.size(1) == self.z_dim
        log_mix_probs = torch.log_softmax(self.mix_logit, dim=-1).unsqueeze(0)  # (1, n_mix)
        assert log_mix_probs.size(0) == 1 and log_mix_probs.size(1) == self.n_mix
        
        prior_prob = torch.softmax(self.mix_logit, dim=0).unsqueeze(0)
        log_comp_probs = self.component_log_prob(x)  # (nbatch, n_mix)
        
        log_prob_x = torch.logsumexp(log_mix_probs + log_comp_probs, dim=-1, keepdim=True)  # (nbatch, 1)
        log_posterior = log_comp_probs + log_mix_probs - log_prob_x  # (nbatch, n_mix)
        posterior = torch.exp(log_posterior)
        if with_elbo:
            kl_from_prior = kl_divergence(Categorical(probs=posterior), Categorical(probs=prior_prob))
            return posterior, dict(
                comp_log_prob=log_comp_probs,
                log_data=(posterior * log_comp_probs).sum(dim=-1),
                kl_from_prior=kl_from_prior)
        else:
            return posterior
    
    def get_centroids(self):
        with torch.no_grad():
            return self.comp_mean.clone().detach()
    # def edges(self):
    #     with torch.no_grad():
    #         return self.edges.clone().detach()
    
    def circles(self):
        with torch.no_grad():
            return torch.exp(self.comp_logstd).clone().expand_as(self.comp_mean).detach()
    
    def std_mean(self):
        return torch.exp(self.comp_logstd).mean()
    
    def assign_centroids(self, x):
        self.comp_mean.data.copy_(x)
        
    
    def get_cluster_id(self, x):
        return max_sim(a=x.cpu(), b=self.get_centroids().cpu())[1]
    #################################
    
    def initialize_cluster_edges(self):
        path = self.args.cluster_root_path + str(self.args.num_agents) + self.args.cluster_last_path
        x = np.load(path + self.args.cluster_centroids_path, allow_pickle=True)
        edge = np.load(path + self.args.cluster_edges_path, allow_pickle=True)
        self.assign_centroids(torch.from_numpy(x))
        self.edges_count = torch.from_numpy(edge)
        # print(self.edges_count)
        self.update_G()
        
    def design_final_goal(self, x):
        """
        x : (n_agents, obs_shape)
        return final goal: (1, embding_size)
        """
        # print(x.shape)
        # np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
        #                             0 1                   2 3    |  4 + n*2-1 | 4 + n*2 : 4 + n*2 + (n-1)*2-1  
        N = x.shape[0]
        l_pos = [x[0][2:4] + x[0][4+2*j:4+2*j+2] for j in range(N)]
        for i in range(N):
            x[i][2:4] = l_pos[i]
            tmp_l_pos = l_pos.copy()
            tmp_l_pos.pop(i)
            for j in range(N-1):
                x[i][4+N*2+j*2:4+N*2+j*2+2] = tmp_l_pos[j]-l_pos[i]
        
        return x
            
            
    def normalization(self, edges):
        """
        归一化处理
        :return: 
        """
        # 归一化到[-1,1]
        mm = MinMaxScaler((-1,0))
        self.weighted_edges = -mm.fit_transform(edges)
        # print(data)
        return self.weighted_edges
    
    def construct_graph(self, edges):
        
        edge_lst = []
        for i in range(len(edges)):
            for j in range(len(edges)):
                if edges[i][j] > 0 and edges[i][j] < 0.95: # 删去不可达的路径
                    edge_lst.append((i, j, edges[i][j]))

        self.G.add_weighted_edges_from(edge_lst)  # 向图中添加多条赋权边: (node1,node2,weight)
        
        # self.get_path(0,7)
        
    def update_G(self):
        self.weighted_edges = self.normalization(self.edges_count)
        self.construct_graph(self.weighted_edges)
        
    def get_path(self, start, end):
        if start == end:
            return [start]
        else:
            minWPath = nx.dijkstra_path(self.G, source=start, target=end)
            return minWPath
             
    def dpp_sampling(self, feature, k=None):
        """
        feature : n_agents, feature_shape
        return : [0 1 ...]
        """
        dpp = DPP(feature)
        dpp.compute_kernel(kernel_type = 'rbf', sigma= 0.4)   # use 'cos-sim' for cosine similarity
        if k == None:
            samples = dpp.sample()                   # samples := [1,7,2,5]
        else:
            samples = dpp.sample_k(k)                # ksamples := [5,8,0]
        return samples

    def construct_groups(self, obs):
        all_index = np.arange(obs.shape[0])
        group_leader = self.dpp_sampling(obs.cpu().detach(), self.args.group_num)  ### leader index
        other_index = np.delete(all_index, group_leader)                     ### other index
    
        nearest_id = max_sim(a=obs[other_index].cpu(), b=obs[group_leader].cpu())[1]     ### other id
        
        ind_ = 0  ### 首先先把其他index对应的组号赋值
        for i in other_index:
            all_index[i] = nearest_id[ind_]
            ind_ += 1
        
        ind_ = 0 ### 其次 再把leader的组号按顺序赋值
        for i in group_leader:
            all_index[i] = ind_
            ind_ += 1
               
        return all_index
        
        
    @staticmethod
    def _has_nan(x):
        return torch.any(torch.isnan(x)).cpu().numpy() == True
    
    def embed_loss(self, embedding):
        # self.args.cluster_update_freq = 2
        posterior, elbo = self.forward(embedding, with_elbo=True)
        log_data = elbo['log_data']
        kl_from_prior = elbo['kl_from_prior']
        if self._has_nan(log_data) or self._has_nan(kl_from_prior):
            pass
        loss_elbo = - (log_data - self.args.elbo_beta * kl_from_prior).mean()
        std_mean = self.std_mean()
        loss_std = self.args.cluster_std_reg * std_mean
        loss_embed_total = loss_elbo + loss_std
        # self.monitor.store(
        #     Loss_elbo=loss_elbo.item(),
        #     Loss_cluster_std=loss_std.item(),
        #     Loss_embed_total=loss_embed_total.item(),
        # )
        # monitor_log = dict(
        #     Cluster_log_data=log_data,
        #     Cluster_kl=kl_from_prior,
        #     Cluster_post_std=posterior.std(dim=-1),
        #     Cluster_std_mean=std_mean,
        # )
        # self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_embed_total
        
   
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=3)
    parser.add_argument('--n_latent_landmarks', type=int, default=10)
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--learned_prior', action='store_true')
    parser.add_argument('--elbo_beta', type=float, default=1.0)
    parser.add_argument('--lr_cluster', type=float, default=3e-4)
    parser.add_argument('--cluster_std_reg', type=float, default=0.0)
    parser.add_argument('--root_path', type=str, default='/home/jqruan/data/clustering/on-policy-mpe/onpolicy/scripts/results/MPE/simple_spread/rmappo/')
    parser.add_argument('--last_path', type=str, default='agents/run1/models/train_state_logs/ensemble_use_MI__shuffle_[02-18]08.50.55/models_ensemble/MI_shuffle')
    parser.add_argument('--centroids_path', type=str, default='sample_data_2w_centroids_10.npy')
    parser.add_argument('--edges_path', type=str, default='sample_data_2w_edges_10_10.npy')
    
    
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = get_args()
    cluster = Cluster(args)
    
    pts = torch.rand(10, 64)
    edges = torch.zeros(10, 10)        
    
    cluster.initialize_cluster_edges(pts, edges) 
    
    
    
    
