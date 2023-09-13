import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.algorithms.utils.util import check
from collections import Counter
import wandb
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class MPEGraphRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPEGraphRunner, self).__init__(config)
        self.all_args = config['all_args']
        self.cur_group_lst = None
        if self.all_args.cuda:
            device=torch.device("cuda:0")
        else:
            device=torch.device("cpu")
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            self.warmup() ###### 每次开局重置位置
            for step in range(self.episode_length):
                if step == 0: ### design the goal
                    final_obs = np.array([self.policy.cluster.design_final_goal(i) for i in self.buffer.obs[step]])         
                    obs_ = check(torch.from_numpy(self.buffer.obs[step])).to(**self.tpdv)           
                    local_h = self.trainer.policy.bis_agent_local.encoder(obs_)
                    self.cur_group_lst = [self.trainer.policy.cluster.construct_groups(i) for i in local_h]
                     
            
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, intention, group = self.collect(step, final_obs)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, intention, group
                

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            ####### update the graph
            obs_cur = check(np.concatenate(np.concatenate(self.buffer.obs[:self.episode_length]))).to(**self.tpdv) 
            obs_cur_next = check(np.concatenate(np.concatenate(self.buffer.obs[1:]))).to(**self.tpdv) 
            cur_id = self.trainer.policy.cluster.get_cluster_id(self.trainer.policy.bis_agent_local.encoder(obs_cur))
            cur_next_id = self.trainer.policy.cluster.get_cluster_id(self.trainer.policy.bis_agent_local.encoder(obs_cur_next))
            index_cat = torch.cat((cur_id[None], cur_next_id[None]), axis=0).transpose(1, 0).tolist() ## 2,9600
            node_num = self.trainer.policy.cluster.edges_count.shape[0]
            
            for i_ in range(node_num):
                for j_ in range(node_num):
                    self.trainer.policy.cluster.edges_count[i_][j_] += index_cat.count([i_, j_])
                    
            self.trainer.policy.cluster.update_G() #### 更新图
            
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step, final_obs):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic, self.cur_group_lst, intention, group \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            final_obs,
                            self.cur_group_lst
                            )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        intention = np.array(np.split(_t2n(intention), self.n_rollout_threads))
        group = np.array(np.split(_t2n(group), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, intention, group

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, intentions, groups = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, intentions=intentions, groups=groups)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)


    @torch.no_grad()
    def test(self):
        ####### 10000 eps
        eval_obs_lsts, eval_act_lsts, eval_rew_lsts, eval_done_lsts, eval_next_obs_lsts = [], [], [], [], []
        eval_obs_lsts_done, eval_act_lsts_done, eval_rew_lsts_done, eval_done_lsts_done, eval_next_obs_lsts_done = [], [], [], [], []
        eval_global_s, eval_global_next_s, eval_global_a = [], [], []
        # eval_episode_rewards = []
        dones_flag = 0
        
        N = 50001
        slice_ = N // 5
        
        # testing
        # N = 101
        # slice_ = N // 5
        
        for eps_ in range(1, N):
            eval_obs_lst, eval_act_lst, eval_rew_lst, eval_done_lst, eval_next_obs_lst = [], [], [], [], []
            eval_obs = self.eval_envs.reset()
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for eval_step in range(self.episode_length):
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                
                if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[0].shape):
                        eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                        if i == 0:
                            eval_actions_env = eval_uc_actions_env
                        else:
                            eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
                elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                    eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                eval_obs_lst.append(eval_obs)
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                eval_act_lst.append(eval_actions)
                eval_rew_lst.append(eval_rewards)
                eval_done_lst.append(eval_dones[:,:,None])
                eval_next_obs_lst.append(eval_obs)
            
                # eval_episode_rewards.append(np.mean(eval_rewards))

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
        
            if np.sum(np.vstack(eval_obs_lst)) > 0 :
                dones_flag += 1
                eval_obs_lsts_done.append(np.vstack(eval_obs_lst))
                eval_act_lsts_done.append(np.vstack(eval_act_lst))
                eval_rew_lsts_done.append(np.vstack(eval_rew_lst))
                eval_done_lsts_done.append(np.vstack(eval_done_lst))
                eval_next_obs_lsts_done.append(np.vstack(eval_next_obs_lst))
                # np.vstack(eval_obs_lst).reshape(self.episode_length, -1)[:,None].repeat(self.num_agents ,axis=1)
                eval_global_s.append(np.vstack(eval_obs_lst).reshape(self.episode_length, -1)[:,None].repeat(self.num_agents ,axis=1))
                eval_global_next_s.append(np.vstack(eval_next_obs_lst).reshape(self.episode_length, -1)[:,None].repeat(self.num_agents ,axis=1))
                eval_global_a.append(np.vstack(eval_act_lst).reshape(self.episode_length, -1)[:,None].repeat(self.num_agents ,axis=1))
                
                
            eval_obs_lsts.append(np.vstack(eval_obs_lst))
            eval_act_lsts.append(np.vstack(eval_act_lst))
            eval_rew_lsts.append(np.vstack(eval_rew_lst))
            eval_done_lsts.append(np.vstack(eval_done_lst))
            eval_next_obs_lsts.append(np.vstack(eval_next_obs_lst))
            
            if eps_ % slice_ == 0:
                slices = eps_ // slice_
                #### save ## N, 25, 3, 39
                global_s = np.array(eval_obs_lsts).reshape(*(np.array(eval_obs_lsts).shape[:2]), -1)[:,:,None].repeat(self.num_agents ,axis=2)
                global_next_s = np.array(eval_next_obs_lsts).reshape(*(np.array(eval_next_obs_lsts).shape[:2]), -1)[:,:,None].repeat(self.num_agents ,axis=2)
                global_a = np.array(eval_act_lsts).reshape(*(np.array(eval_act_lsts).shape[:2]), -1)[:,:,None].repeat(self.num_agents ,axis=2)
                
                data_all = np.concatenate((np.array(eval_obs_lsts), np.array(eval_act_lsts), np.array(eval_rew_lsts), np.array(eval_next_obs_lsts), np.array(eval_done_lsts), \
                    global_s, global_next_s, global_a), axis=-1)
                num = data_all.shape[0]
                np.save(self.all_args.test_model_path + 'o_a_r_o_prime_done_s_s_prime_u-slices_' + str(slices) + '-num_' + str(num) + '.npy', data_all, allow_pickle=True)
                eval_obs_lsts, eval_act_lsts, eval_rew_lsts, eval_done_lsts, eval_next_obs_lsts = [], [], [], [], []
                print("eval dones number: " + str(dones_flag))
                
        
        data_all_done = np.concatenate((np.array(eval_obs_lsts_done), np.array(eval_act_lsts_done), np.array(eval_rew_lsts_done), np.array(eval_next_obs_lsts_done), np.array(eval_done_lsts_done),\
            np.array(eval_global_s), np.array(eval_global_next_s), np.array(eval_global_a)), axis=-1)
  
        num = data_all_done.shape[0]
        np.save(self.all_args.test_model_path + 'o_a_r_o_prime_done_s_s_prime_u' + '-num_' + str(num)  + '_done.npy', data_all_done, allow_pickle=True)
        # eval_episode_rewards = np.array(eval_episode_rewards)
        # eval_env_infos = {}
        # eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        # eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        # print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        
        # self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)




class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.all_args = config['all_args']
        
        if self.all_args.cuda:
            device=torch.device("cuda:0")
        else:
            device=torch.device("cpu")
        
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                if self.all_args.use_dbc:
                    obs = check(obs).to(**self.tpdv)
                    dbc_feature = self.trainer.policy.bis_model.encoder(obs)
                    obs_ = torch.cat((obs,dbc_feature), dim=-1)
                    obs = obs_.cpu().detach().numpy() 
                

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        if self.all_args.use_dbc:
            obs = check(obs).to(**self.tpdv)
            dbc_feature = self.trainer.policy.bis_model.encoder(obs)
            obs_ = torch.cat((obs,dbc_feature), dim=-1)
            # obs = obs_.cpu().numpy() 
            obs = obs_.cpu().detach().numpy() 
        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)


    @torch.no_grad()
    def test(self):
        ####### 10000 eps
        eval_obs_lsts, eval_act_lsts, eval_rew_lsts, eval_done_lsts, eval_next_obs_lsts = [], [], [], [], []
        eval_obs_lsts_done, eval_act_lsts_done, eval_rew_lsts_done, eval_done_lsts_done, eval_next_obs_lsts_done = [], [], [], [], []
        eval_global_s, eval_global_next_s, eval_global_a = [], [], []
        # eval_episode_rewards = []
        dones_flag = 0
        
        N = 50001
        slice_ = N // 5
        
        # testing
        # N = 101
        # slice_ = N // 5
        
        for eps_ in range(1, N):
            eval_obs_lst, eval_act_lst, eval_rew_lst, eval_done_lst, eval_next_obs_lst = [], [], [], [], []
            eval_obs = self.eval_envs.reset()
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for eval_step in range(self.episode_length):
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                
                if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[0].shape):
                        eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                        if i == 0:
                            eval_actions_env = eval_uc_actions_env
                        else:
                            eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
                elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                    eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                eval_obs_lst.append(eval_obs)
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                eval_act_lst.append(eval_actions)
                eval_rew_lst.append(eval_rewards)
                eval_done_lst.append(eval_dones[:,:,None])
                eval_next_obs_lst.append(eval_obs)
            
                # eval_episode_rewards.append(np.mean(eval_rewards))

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
        
            if np.sum(np.vstack(eval_obs_lst)) > 0 :
                dones_flag += 1
                eval_obs_lsts_done.append(np.vstack(eval_obs_lst))
                eval_act_lsts_done.append(np.vstack(eval_act_lst))
                eval_rew_lsts_done.append(np.vstack(eval_rew_lst))
                eval_done_lsts_done.append(np.vstack(eval_done_lst))
                eval_next_obs_lsts_done.append(np.vstack(eval_next_obs_lst))
                # np.vstack(eval_obs_lst).reshape(self.episode_length, -1)[:,None].repeat(self.num_agents ,axis=1)
                eval_global_s.append(np.vstack(eval_obs_lst).reshape(self.episode_length, -1)[:,None].repeat(self.num_agents ,axis=1))
                eval_global_next_s.append(np.vstack(eval_next_obs_lst).reshape(self.episode_length, -1)[:,None].repeat(self.num_agents ,axis=1))
                eval_global_a.append(np.vstack(eval_act_lst).reshape(self.episode_length, -1)[:,None].repeat(self.num_agents ,axis=1))
                
                
            eval_obs_lsts.append(np.vstack(eval_obs_lst))
            eval_act_lsts.append(np.vstack(eval_act_lst))
            eval_rew_lsts.append(np.vstack(eval_rew_lst))
            eval_done_lsts.append(np.vstack(eval_done_lst))
            eval_next_obs_lsts.append(np.vstack(eval_next_obs_lst))
            
            if eps_ % slice_ == 0:
                slices = eps_ // slice_
                #### save ## N, 25, 3, 39
                global_s = np.array(eval_obs_lsts).reshape(*(np.array(eval_obs_lsts).shape[:2]), -1)[:,:,None].repeat(self.num_agents ,axis=2)
                global_next_s = np.array(eval_next_obs_lsts).reshape(*(np.array(eval_next_obs_lsts).shape[:2]), -1)[:,:,None].repeat(self.num_agents ,axis=2)
                global_a = np.array(eval_act_lsts).reshape(*(np.array(eval_act_lsts).shape[:2]), -1)[:,:,None].repeat(self.num_agents ,axis=2)
                
                data_all = np.concatenate((np.array(eval_obs_lsts), np.array(eval_act_lsts), np.array(eval_rew_lsts), np.array(eval_next_obs_lsts), np.array(eval_done_lsts), \
                    global_s, global_next_s, global_a), axis=-1)
                num = data_all.shape[0]
                np.save(self.all_args.test_model_path + 'o_a_r_o_prime_done_s_s_prime_u-slices_' + str(slices) + '-num_' + str(num) + '.npy', data_all, allow_pickle=True)
                eval_obs_lsts, eval_act_lsts, eval_rew_lsts, eval_done_lsts, eval_next_obs_lsts = [], [], [], [], []
                print("eval dones number: " + str(dones_flag))
                
        
        data_all_done = np.concatenate((np.array(eval_obs_lsts_done), np.array(eval_act_lsts_done), np.array(eval_rew_lsts_done), np.array(eval_next_obs_lsts_done), np.array(eval_done_lsts_done),\
            np.array(eval_global_s), np.array(eval_global_next_s), np.array(eval_global_a)), axis=-1)
  
        num = data_all_done.shape[0]
        np.save(self.all_args.test_model_path + 'o_a_r_o_prime_done_s_s_prime_u' + '-num_' + str(num)  + '_done.npy', data_all_done, allow_pickle=True)
        # eval_episode_rewards = np.array(eval_episode_rewards)
        # eval_env_infos = {}
        # eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        # eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        # print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        
        # self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
