# -*- coding:utf-8  -*-
# 作者：zruizhi
# 创建时间： 2020/8/6 2:49 下午
# 描述：用户运行入口，需要用户定义myagent


from .chooseenv import make
from gym import spaces
from functools import reduce

# from seeker.agents.RLTrainer import *
import torch
import gym
import numpy as np


class seek_env(gym.Env):
    def __init__(self, number, episode_limit):
        self.env_type = 'seek_2p'

        # 没有model设为空
        model_name = ""

        # 没有model设为空
        test_or_train = ""

        render_mode = False
        self.number = number # agent number
        self.num_agents = number
        self.episode_limit = episode_limit

        game = make(self.env_type, number, conf=None)

        self.action_dim = game.joint_action_space[0][0]  # The same shape for each agent

        obs_shape = list(game.get_grid_many_obs_space(range(game.n_player))[0])
        self.state_dim = (3 + self.num_agents) * reduce(lambda x, y: x * y, obs_shape)

        self.action_space = spaces.MultiDiscrete([self.action_dim] * self.number)
        self.observation_space = [spaces.Box(-np.inf, np.inf, (self.state_dim,), dtype=np.float64)
                                  for _ in range(self.num_agents)]
        self.share_observation_space = [spaces.Box(-np.inf, np.inf, (self.state_dim*self.num_agents,), dtype=np.float64)
                                  for _ in range(self.num_agents)]
        # self.share_observation_space = self.observation_space

        print("agent number for seeker -------  :", self.num_agents)

        self.game = game

    def reset(self):
        self.game = make(self.env_type, self.number, conf=None)
        init_state = self.game.current_state

        init_state = self.trans_next_state(init_state)

        avail_actions = self.get_avail_actions()
    
        # return init_state, init_state[:0], avail_actions
        return init_state, avail_actions

    def design_final_goal(self):
        final_goal = self.game.get_final_goal()
        return np.array(final_goal).reshape(1,-1)
    
    def trans_next_state(self, next_state):
        obs_state = torch.stack(([torch.Tensor(next_state[i]) for i in range(len(next_state[0]))]),
                                dim=0).reshape([-1, 1]).permute([1, 0])
        obs_state = obs_state.squeeze(0).numpy()
        current_obs_state = np.array([obs_state] * self.game.n_player)
        # current_obs_state = [obs_state] * self.game.n_player
        return current_obs_state


    def step(self, joint_act):
        g = self.game
        next_state, reward, done, info_before, info_after = g.step(joint_act)
        # info = {"info_before", info_before,
        #         "info_after", info_after,
        #         "individual_reward", reward[0]
        #         }
        info = {}
        next_state = self.trans_next_state(next_state)

        avail_actions = self.get_avail_actions()

        reward = np.expand_dims(np.array(reward), axis=1)

        # return (next_state, next_state[:0], reward, np.array([done]*self.number), info, avail_actions)
        info['avail_actions'] = avail_actions
        return next_state, reward, np.array([done]*self.number), info
    
    def get_info(self):
        pass

    def get_obs(self):
        current_state = self.game.current_state
        obs_state = torch.stack(([torch.Tensor(current_state[i]) for i in range(len(current_state[0]))]),
                                        dim=0).reshape([-1, 1]).permute([1, 0])
        obs_state = obs_state.squeeze(0).numpy()
        current_obs_state = np.array([obs_state] * self.game.n_player)
        return current_obs_state

    def get_avail_agent_actions(self, agent_id):
        # for modified env
        each = self.game.get_valid_action(self.game.players[agent_id])
        # each = [1] * 5
        # # idx = np.random.randint(action_space_list[i][j])
        # each = np.array(each)
        return each

    def get_avail_actions(self):
        # for modified env ### all agents get_avail_actions
        each = [self.game.get_valid_action(self.game.players[agent_id]) for agent_id in range(self.number)]
        # each = [1] * 5
        # # idx = np.random.randint(action_space_list[i][j])
        each = np.array(each)
        return each



    def call_action_dim(self):
        return self.action_dim
    def call_state_dim(self):
        return self.state_dim


    def observation(self):
        return self.get_obs()[0]
    def action_space(self):
        return spaces.MultiDiscrete([self.action_dim]*self.num_agents)

    def get_state(self):
        return self.get_obs()[0]

    def get_env_info(self):# {'state_shape': 61, 'obs_shape': 42, 'n_actions': 10, 'n_agents': 3, 'episode_limit': 200}
        info = {}
        info['state_shape'] = self.call_state_dim()
        info['obs_shape'] = self.call_state_dim()
        info['n_actions'] = self.call_action_dim()
        info['n_agents'] = self.num_agents
        info['episode_limit'] = self.episode_limit
        return info

    def close(self):
        pass








