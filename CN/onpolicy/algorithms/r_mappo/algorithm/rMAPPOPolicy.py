import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic, R_Actor_Graph, R_Critic_Graph
from onpolicy.algorithms.bis_dbc.dbc_agent import *
from gym.spaces.box import Box


from onpolicy.utils.util import update_linear_schedule

from onpolicy.algorithms.utils.util import init, check

from onpolicy.algorithms.r_mappo.algorithm.clustering import Cluster
import numpy as np

import importlib
imp = importlib.import_module('onpolicy.scripts.train.0_offline_train_states.transition_model')
# from onpolicy.scripts.train.0_offline_train_states.transition_model import *

from onpolicy.algorithms.dpp.dpp import DPP

from gym import spaces

# R_MAPPO_GRAPH_Policy

class R_MAPPO_GRAPH_Policy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.args = args
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor_Graph(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic_Graph(args, self.share_obs_space, self.device)
        self.tpdv = dict(dtype=torch.float32, device=device)
                
        if isinstance(self.share_obs_space, spaces.Box):
            self.global_obs_size = self.share_obs_space.shape[0]
            self.joint_action_shape = self.act_space.n * self.args.num_agents
            self.obs_size = self.obs_space.shape[0]
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        if self.args.use_graph:
            print(' ---------------use ours-------------')
            self.cluster = Cluster(args).to(self.device)
            self.cluster.initialize_cluster_edges()
            self.cluster_optimizer = torch.optim.Adam(self.cluster.parameters(),
                                                 lr=self.args.cluster_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
            
            transition_model_type = 'ensemble'  #
            hidden_size, output_size, num_actions = 64, self.args.embed_size, 5
            
            self.bis_agent_global = imp.BisimAgent(self.global_obs_size, hidden_size, output_size, self.joint_action_shape, transition_model_type) 
            self.bis_agent_local = imp.BisimAgent(self.obs_size, hidden_size, output_size, num_actions, transition_model_type) 
            ###### bis_agents_path
            path_ = self.args.cluster_root_path + str(self.args.num_agents) + self.args.cluster_last_path
            self.bis_agent_global.load_state_dict(torch.load(path_+self.args.global_bis_agents_path))
            self.bis_agent_local.load_state_dict(torch.load(path_+self.args.local_bis_agents_path))
            
            self.bis_agent_global.to(self.device)
            self.bis_agent_local.to(self.device)
            self.bis_agent_global.transition_model.to(self.device)
            self.bis_agent_local.transition_model.to(self.device)
            
            # self.bis_optimizer = torch.optim.Adam(self.cluster.parameters(),
            #                                      lr=self.args.cluster_lr,
            #                                      eps=self.opti_eps,
            #                                      weight_decay=self.weight_decay)
            ###### 
            

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, final_obs, cur_group_lst, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """        
        local_h = self.bis_agent_local.encoder(check(obs).to(**self.tpdv)).reshape(self.args.n_rollout_threads, self.args.num_agents, -1) # 64.3.16
        global_obs = obs.reshape(self.args.n_rollout_threads, -1) 
        final_obs = final_obs.reshape(self.args.n_rollout_threads, -1) 
        
        graph_h_cur = self.bis_agent_global.encoder(check(global_obs).to(**self.tpdv))
        graph_h_final = self.bis_agent_global.encoder(check(final_obs).to(**self.tpdv))
        
        cur = self.cluster.get_cluster_id(graph_h_cur)
        end = self.cluster.get_cluster_id(graph_h_final)
        
        should_change = torch.nonzero(cur!=end)[:,0]
        for i in should_change:
            cur[i] = self.cluster.get_path(cur[i].item(), end[i].item())[1]
            cur_group_lst[i] = self.cluster.construct_groups(local_h[i])
            
        centroids = torch.repeat_interleave(self.cluster.get_centroids()[None], self.args.n_rollout_threads, 0)  # 64, 10, 16
        index = torch.repeat_interleave(cur[:,None,None], centroids.shape[-1], -1)  # 64, 1, 16
        intention_tensors = torch.gather(centroids.cpu(), 1, index) # 64, 1, 16
        intention_tensors = torch.repeat_interleave(intention_tensors, self.args.num_agents, 1) #  --> 64, 3, 16

        # group_tensors = [(local_h*np.repeat((np.array(cur_group_lst) == i)[:,:,None], local_h.shape[-1], -1)).mean(1, keepdim=True) for i in range(self.args.group_num)] # [64.1.16, 64.1.16]
        group_tensor = torch.cat([(local_h.cpu()*np.repeat((np.array(cur_group_lst) == i)[:,:,None], local_h.shape[-1], -1)).mean(1, keepdim=True) for i in range(self.args.group_num)], 1) # 64.2.16
        group_tensors = torch.gather(group_tensor, 1, torch.tensor(cur_group_lst)[:,:,None].repeat(1,1,local_h.shape[-1])) ### 
        
        intention_tensors_, group_tensors_ = intention_tensors.reshape(-1, self.args.embed_size), group_tensors.reshape(-1, self.args.embed_size)
        
        actions, action_log_probs, rnn_states_actor = self.actor(obs, intention_tensors_, group_tensors_,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)
        cent_obs_cat = np.concatenate((cent_obs, intention_tensors_, group_tensors_), axis=-1)
        values, rnn_states_critic = self.critic(cent_obs_cat, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, cur_group_lst, intention_tensors_, group_tensors_

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.args = args
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        
        if self.args.use_dbc:
            hidden_size, output_size, action_shape, transition_model_type = 64, 16, 5, 'ensemble'  
            input_size = self.obs_space.shape[0]
            self.bis_model = BisimAgent(input_size, hidden_size, output_size, action_shape, transition_model_type).to(self.device)
            
            obs_space_after = Box(low=-np.inf, high=np.inf, shape=(input_size+output_size, ), dtype=np.float32)
            share_obs_space_after = Box(low=-np.inf, high=np.inf, shape=(self.share_obs_space.shape[0]+output_size*self.args.num_agents, ), dtype=np.float32)

            self.actor = R_Actor(args, obs_space_after, self.act_space, self.device)
            self.critic = R_Critic(args, share_obs_space_after, self.device)
        
        if isinstance(self.share_obs_space, spaces.Box):
            self.global_obs_size = self.share_obs_space.shape[0]
            self.joint_action_shape = self.act_space.n * self.args.num_agents
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        
    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                rnn_states_actor,
                                                                masks,
                                                                available_actions,
                                                                deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
