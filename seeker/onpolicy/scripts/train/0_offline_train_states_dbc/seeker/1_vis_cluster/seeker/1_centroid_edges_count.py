from torch.utils.data import DataLoader
from tqdm import tqdm

# from transition_model import make_transition_model
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
# from _train_state import BisimAgent


import numpy as np
import torch
import os

from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
# from transition_model import make_transition_model
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta

import random

# kmeans
from fast_pytorch_kmeans import KMeans
### vis
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class DeterministicTransitionModel(nn.Module):

    def __init__(self, encoder_feature_dim, action_shape, layer_width):
        super().__init__()
        self.fc = nn. Linear(encoder_feature_dim + action_shape, layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        print("Deterministic transition model chosen.")

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        return mu


class ProbabilisticTransitionModel(nn.Module):

    def __init__(self, encoder_feature_dim, action_shape, layer_width, announce=True, max_sigma=1e1, min_sigma=1e-4):
        super().__init__()
        self.fc = nn. Linear(encoder_feature_dim + action_shape, layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        self.fc_sigma = nn.Linear(layer_width, encoder_feature_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert(self.max_sigma >= self.min_sigma)
        if announce:
            print("Probabilistic transition model chosen.")

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        # sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = F.softplus(self.fc_sigma(x))  #
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps


class EnsembleOfProbabilisticTransitionModels(object):

    def __init__(self, encoder_feature_dim, action_shape, layer_width, ensemble_size=5):
        self.models = [ProbabilisticTransitionModel(encoder_feature_dim, action_shape, layer_width, announce=False)
                       for _ in range(ensemble_size)]
        print("Ensemble of probabilistic transition models chosen.")

    def __call__(self, x):
        mu_sigma_list = [model.forward(x) for model in self.models]
        mus, sigmas = zip(*mu_sigma_list)
        mus, sigmas = torch.stack(mus), torch.stack(sigmas)
        return mus, sigmas

    def sample_prediction(self, x):
        model = random.choice(self.models)
        return model.sample_prediction(x)

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def parameters(self):
        list_of_parameters = [list(model.parameters()) for model in self.models]
        parameters = [p for ps in list_of_parameters for p in ps]
        return parameters


_AVAILABLE_TRANSITION_MODELS = {'': DeterministicTransitionModel,
                                'deterministic': DeterministicTransitionModel,
                                'probabilistic': ProbabilisticTransitionModel,
                                'ensemble': EnsembleOfProbabilisticTransitionModels}


def make_transition_model(transition_model_type, encoder_feature_dim, action_shape, layer_width=512):
    assert transition_model_type in _AVAILABLE_TRANSITION_MODELS
    return _AVAILABLE_TRANSITION_MODELS[transition_model_type](
        encoder_feature_dim, action_shape, layer_width
    )


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
        self.transition_model = make_transition_model(
            transition_model_type, output_size, action_shape
        )
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=1e-3, weight_decay=0.0
        )






######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------


class chj_data(object):
        def __init__(self, data, target):
            self.data=data
            self.target=target
            
def chj_load_file(h, label):
    feature = h
    target = label
    res = chj_data(feature, target)
    return res

######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
#### ######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
######################### parameter steup begins ----------------
#### 

###### 3 agents
# dirs = ['ensemble__[02-18]08.51.17', 'ensemble__shuffle_[02-18]08.51.14',
#     'ensemble_use_MI__[02-18]08.51.08', 'ensemble_use_MI__shuffle_[02-18]08.50.55']
# log_names = ['no_MI_no_shuffle', 'no_MI_shuffle', 'MI_no_shuffle', 'MI_shuffle']
# model_nums = [1120000, 1000000, 920000, 960000]


# dirs = ['ensemble__[02-18]00.49.11', 'ensemble__shuffle_[02-18]00.49.05',
#     'ensemble_use_MI__[02-18]00.48.57', 'ensemble_use_MI__shuffle_[02-18]00.48.59']
# log_names = ['no_MI_no_shuffle', 'no_MI_shuffle', 'MI_no_shuffle', 'MI_shuffle']
# model_nums = [820000, 700000, 980000, 800000]

###### 6 agents
# dirs = ['ensemble__[02-18]14.36.42', 'ensemble__shuffle_[02-18]14.36.36',
#     'ensemble_use_MI__[02-18]14.36.31', 'ensemble_use_MI__shuffle_[02-18]14.36.25']
# log_names = ['no_MI_no_shuffle', 'no_MI_shuffle', 'MI_no_shuffle', 'MI_shuffle']
# model_nums = [4240000, 3440000, 3780000, 2820000]


###### 3 agents: data_load_index=0
###### 4 agents: data_load_index=1 
###### 6 agents: data_load_index=2
###### 8 agents: data_load_index=3
###### 10 agents: data_load_index=4


###### 4 agents: data_load_index=5  ### emb 32
###### 6 agents: data_load_index=6  ### emb 32

###### 4 agents: data_load_index=5  ### emb 32
###### 6 agents: data_load_index=6  ### emb 32
###### 8 agents: data_load_index=7  ### emb 32
###### 10 agents: data_load_index=8  ### emb 64

# data_load_index = 0  #######
root_path = '/data/jqruan/clustering/on-policy-seeker-161-20230320/onpolicy/scripts/train/results_ours/seeker'

for data_load_index in range(0, 4, 1):
    
    dirs_all = [
        [
        '/3agents/3agents_mappo/seed_1_run1/models/train_state_logs_dbc/ensemble/[06-02]15.58.45/',
      ],
        [
            '/5agents/5agents_mappo/seed_1_run1/models/train_state_logs_dbc/ensemble/[06-02]16.04.47/',
        ],
        [
            '/8agents/8agents_rmappo/seed_1_run1/models/train_state_logs_dbc/ensemble/[06-02]16.05.01', 
        ],
        [
            '/12agents/12agents_rmappo/seed_1_run1/models/train_state_logs_dbc/ensemble/[06-02]16.05.09',
        ]
        
    ]


    log_names_all = [
        ['no_MI_no_shuffle'],
        ['no_MI_no_shuffle'],
        ['no_MI_no_shuffle'],
        ['no_MI_no_shuffle'],
    ]

    model_nums_all = [  
        [
            [100000, 300000, 500000, 720000],
        ],
        [
            [100000, 200000, 400000, 600000], # 'no_MI_no_shuffle', 
        ],
        [
            [20000, 40000],  # 'no_MI_no_shuffle', 
        ],
        [
            [20000, 80000, 160000, 320000],  # 'no_MI_no_shuffle', 
        ]
        
    ]
    dirs, log_names, model_nums = dirs_all[data_load_index], log_names_all[data_load_index], model_nums_all[data_load_index]
    
    dones_ = [
        '/data/jqruan/data/clustering/seeker/mappo/3agents/o_a_r_o_prime_done_s_s_prime_u-slices_1-num_5000_random.npy',
        '/data/jqruan/data/clustering/seeker/mappo/5agents/data_all/o_a_r_o_prime_done_s_s_prime_u-slices_0-num_5000_random.npy',
        '/data/jqruan/data/clustering/seeker/mappo/8agents/o_a_r_o_prime_done_s_s_prime_u-slices_1-num_2000_random.npy',
        '/data/jqruan/data/clustering/seeker/mappo/12agents/data_all/o_a_r_o_prime_done_s_s_prime_u-slices_1-num_500_random.npy'
        ]
        
    
    
    agents_ = [3, 5, 8, 12]
    input_sizes_ = [150, 200, 275, 375]

    num_agents = agents_[data_load_index]    # 3
    action_shape = 5
    input_size, hidden_size, output_size = input_sizes_[data_load_index], 128, 64 # 4:(24, 64, 16)
    global_size = input_size * num_agents
    joint_action_shape = action_shape * num_agents
    batch_size = 128
    transition_model_type = 'ensemble'  
    bis_agent = BisimAgent(input_size, hidden_size, output_size, action_shape, transition_model_type)

    # root_path = root_path + '/models_ensemble/'
    
    all_data = np.load(dones_[data_load_index], allow_pickle=True)

    a,b,c,d = all_data.shape
    all_data = all_data.reshape(a*b*c, d)

    sample_num = 5000
    all_data_size = all_data.shape[0]
    sample_index = np.random.choice(all_data_size, sample_num)
    sample_data = all_data[sample_index]
    
    del all_data

    index = [0]
    # index = [4]
    ## no MI + no shuffle ; ## no MI + shuffle; ## MI + no shuffle; ## MI + shuffle

    for i_ in index:
        print('-----------------', i_)
        
        model_nums_ = model_nums[i_]
        
        for m_n in model_nums_:
            path_model = root_path + dirs[i_] + '/models_ensemble/' + '/bis_agent._' + str(m_n) + '.pt'
            print(path_model)
            
            bis_agent.load_state_dict(torch.load(path_model))


            h = bis_agent.encoder(torch.from_numpy(sample_data[:, :input_size]).float())

            # n_clusters = num_agents
            n_clusters = 10
            
            kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=1)
            labels = kmeans.fit_predict(h)


            iris = chj_load_file(h.detach().numpy(), labels)

            X_tsne = TSNE(n_components=2, init='pca', perplexity=60, learning_rate=100).fit_transform(iris.data)

            fig = plt.figure()
            plt.axis('off')  # 去掉坐标轴
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target, edgecolors='k', marker='o', linewidths=1.0, cmap='Spectral')
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0.03, 0.03)
            
            
            save_path = root_path + dirs[i_] + '/models_ensemble/' + log_names[i_] 
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            
            fig.savefig(save_path+'/kmeans_{}_samples_{}_clusters_{}_model.png'.format(sample_num, n_clusters, m_n), dpi=600, format='png')
            
            #### 存储聚类中心
            centroids = kmeans.centroids.cpu().detach().numpy()
            np.save(save_path + '/sample_data_2w_centroids_10_{}.npy'.format(m_n), centroids, allow_pickle=True)
            
            #### 存储边
            edges = np.zeros((n_clusters, n_clusters))
            
            h_2 = bis_agent.encoder(torch.from_numpy(sample_data[:, input_size+2:input_size+2+input_size]).float())
            labels_2 = kmeans.predict(h_2)
            
            # print(labels.shape)
            # print(labels_2.shape)
            
            for i in range(sample_num):
                edges[labels[i].item()][labels_2[i].item()] += 1
            
            np.save(save_path + '/sample_data_2w_edges_10_10_{}.npy'.format(m_n), edges, allow_pickle=True)
            
                
            
        
        
        
