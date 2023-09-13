import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
from torch.nn import functional as F

class R_MAPPO_GRAPH():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None
            
        self.train_steps = 0

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch,\
        intention_batch, group_batch, next_obs_batch, next_share_obs_batch, joint_act_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        available_actions_batch = check(available_actions_batch).to(**self.tpdv)
        
        # intention_batch = check(intention_batch).to(**self.tpdv)
        # group_batch = check(group_batch).to(**self.tpdv)

        if self.args.abl_flag == '_del_graph': # # abl_flag=0 删除 态势图   # abl_flag=1 删除组别  # abl_flag=2 删除obs
            share_obs_batch_cat = np.concatenate((share_obs_batch, group_batch), axis=-1)
            obs_batch_cat = np.concatenate((obs_batch, group_batch), axis=-1)
        elif self.args.abl_flag == '_del_group':
            share_obs_batch_cat = np.concatenate((share_obs_batch, intention_batch), axis=-1)
            obs_batch_cat = np.concatenate((obs_batch, intention_batch), axis=-1)
        elif self.args.abl_flag == '_del_obs':
            share_obs_batch_cat = np.concatenate((intention_batch, group_batch), axis=-1)
            obs_batch_cat = np.concatenate((intention_batch, group_batch), axis=-1)

        # share_obs_batch_cat = np.concatenate((share_obs_batch, intention_batch, group_batch), axis=-1)
        # obs_batch_cat = np.concatenate((obs_batch, intention_batch, group_batch), axis=-1)
        
        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch_cat,
                                                                              obs_batch_cat, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()
        
        
        #### clustering update
        # print('------')
        with torch.no_grad():
            share_obs_batch = check(share_obs_batch).to(**self.tpdv)
            batch_embed = self.policy.bis_agent_global.encoder(share_obs_batch) # 1800.16
            
        cluster_loss = self.policy.cluster.embed_loss(batch_embed)
        
        self.policy.cluster_optimizer.zero_grad()
        (cluster_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            cluster_grad_norm = nn.utils.clip_grad_norm_(self.policy.cluster.parameters(), self.max_grad_norm)
        else:
            cluster_grad_norm = get_gard_norm(self.policy.cluster.parameters())

        self.policy.cluster_optimizer.step()
        
        
        ###### bisimulation update
        if self.args.use_joint_train and self.train_steps % 100 == 0:
            self.bis_update(obs_batch, share_obs_batch, actions_batch, return_batch, next_obs_batch, next_share_obs_batch, joint_act_batch)


        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cluster_loss, cluster_grad_norm

    
    
    def bis_update(self, obs_batch, share_obs_batch, actions_batch, return_batch, next_obs_batch, next_share_obs_batch, joint_act_batch):
        ##### obs_batch actions_batch return_batch
        batch_size, num_actions, transition_model_type = obs_batch.shape[0], 5, 'ensemble'
        obs_batch = check(obs_batch).to(**self.tpdv)
        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv).long()
        action = torch.zeros(batch_size, num_actions).to(**self.tpdv).scatter_(1, actions_batch, 1)
        rew = check(return_batch).to(**self.tpdv)
        next_obs_batch = check(next_obs_batch).to(**self.tpdv)
        next_share_obs_batch = check(next_share_obs_batch).to(**self.tpdv)
        joint_act_batch = check(joint_act_batch).to(**self.tpdv)

        print('---bis traing---') 
        h = self.policy.bis_agent_local.encoder(obs_batch)
        perm = np.random.permutation(batch_size)
        h2 = h[perm]
        with torch.no_grad():
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.policy.bis_agent_local.transition_model(torch.cat([h, action], dim=1))
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
        pred_next_latent_mu, pred_next_latent_sigma = self.policy.bis_agent_local.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
            

        next_h = self.policy.bis_agent_local.encoder(next_obs_batch)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        
        pred_next_latent = self.policy.bis_agent_local.transition_model.sample_prediction(torch.cat([h, action], dim=1))
        pred_next_reward = self.policy.bis_agent_local.reward_decoder(pred_next_latent)
        reward_loss = F.mse_loss(pred_next_reward, rew)
        transition_reward_loss = transition_loss + reward_loss

        ######################### p 去拟合 q，即：local 去拟合 global的
        ######################### 数据分布 归一化之后 近似成概率分布
        ######################### kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum') 
        global_h = self.policy.bis_agent_global.encoder(share_obs_batch)
        kl_loss = F.kl_div(h.softmax(dim=-1).log(), global_h.softmax(dim=-1).detach(), reduction='sum') 

        ##### update optimizer
        total_loss = 0.5 * encoder_loss + transition_reward_loss + kl_loss
        self.policy.bis_agent_local.encoder_optimizer.zero_grad()
        self.policy.bis_agent_local.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.policy.bis_agent_local.encoder_optimizer.step()
        self.policy.bis_agent_local.decoder_optimizer.step()
        
        
        
        """
        ------------------------------------------------------------------
        the upfate of global bis 
        ------------------------------------------------------------------
        """
        # joint_action = torch.cat(([torch.zeros(batch_size, action_shape).scatter_(1, joint_act[:,[i]], 1) for i in range(num_agents)]), 1)
        
        #### update_encoder
        h = self.policy.bis_agent_global.encoder(share_obs_batch)
        # batch_size_now = h.shape[0]
        perm = np.random.permutation(batch_size)
        h2 = h[perm]
        joint_action = torch.cat(([torch.zeros(batch_size, num_actions).cuda().scatter_(1, joint_act_batch[:,[i]].long(), 1) for i in range(self.args.num_agents)]), 1)
        
        with torch.no_grad():
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.policy.bis_agent_global.transition_model(torch.cat([h, joint_action], dim=1))
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
        pred_next_latent_mu, pred_next_latent_sigma = self.policy.bis_agent_global.transition_model(torch.cat([h, joint_action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
            

        next_h = self.policy.bis_agent_global.encoder(next_share_obs_batch)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        
        pred_next_latent = self.policy.bis_agent_global.transition_model.sample_prediction(torch.cat([h, joint_action], dim=1))
        pred_next_reward = self.policy.bis_agent_global.reward_decoder(pred_next_latent)
        reward_loss = F.mse_loss(pred_next_reward, rew)
        transition_reward_loss = transition_loss + reward_loss

        ##### update optimizer
        total_loss = 0.5 * encoder_loss + transition_reward_loss
        self.policy.bis_agent_global.encoder_optimizer.zero_grad()
        self.policy.bis_agent_global.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.policy.bis_agent_global.encoder_optimizer.step()
        self.policy.bis_agent_global.decoder_optimizer.step()
    
    
    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['cluster_loss'] = 0
        train_info['cluster_grad_norm'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cluster_loss, cluster_grad_norm \
                    = self.ppo_update(sample, update_actor)
                self.train_steps += 1

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                
                train_info['cluster_loss'] += cluster_loss.mean()
                train_info['cluster_grad_norm'] += cluster_grad_norm.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()


class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        self.train_steps = 0
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
