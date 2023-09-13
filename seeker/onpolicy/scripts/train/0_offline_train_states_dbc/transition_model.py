import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-5)
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=1e-5, weight_decay=0.0
        )


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