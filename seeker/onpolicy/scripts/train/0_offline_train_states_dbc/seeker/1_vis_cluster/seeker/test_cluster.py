import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from torch.optim import Adam


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
        
        self.c_optim = Adam(self.parameters(), lr=args.lr_cluster)
    
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
    
    def centroids(self):
        with torch.no_grad():
            return self.comp_mean.clone().detach()
    
    def circles(self):
        with torch.no_grad():
            return torch.exp(self.comp_logstd).clone().expand_as(self.comp_mean).detach()
    
    def std_mean(self):
        return torch.exp(self.comp_logstd).mean()
    
    def assign_centroids(self, x):
        self.comp_mean.data.copy_(x)
        
    
    def get_cluster_id(self, x):
        return max_sim(a=x, b=self.centroids)[1]
    #################################
    
    def initialize_cluster_edges(self, x, edge):
        self.assign_centroids(x)
        self.edges = edge
        
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
    
    
    
    
