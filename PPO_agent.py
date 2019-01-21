# adapted from https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)

import numpy as np 
import torch
import torch.nn as nn

import utils
from network import *

rollout_length = 999
discount = 0.99
use_gae = True
gae_tau = 0.95
optimization_epochs = 10
mini_batch_size = 64
ppo_ratio_clip = 0.2
entropy_weight = 0
gradient_clip = 0.5


class PPOAgent:
    def __init__(self, env, state_size, action_size, num_agents):
        self.env = env
        self.brain_name = self.env.brain_names[0]
        self.num_agents = num_agents
        self.network = GaussianActorCriticNet(state_size, action_size, actor_body=FCBody(state_size, gate=F.tanh),critic_body=FCBody(state_size, gate=F.tanh))
        self.opt = torch.optim.Adam(self.network.parameters(), 3e-4, eps=1e-5)
        self.device = utils.device
        self.online_rewards = np.zeros(self.num_agents)
        self.reward_normalizer = utils.RescaleNormalizer()
        self.episode_rewards = []
        self.total_steps = 0

    def episode(self):
        storage = utils.Storage(rollout_length)
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        obs = env_info.vector_observations
        for _ in range(rollout_length):
            prediction = self.network(torch.from_numpy(obs).float().to(self.device))
            actions = torch.clamp(prediction['a'], -1, 1).cpu().data.numpy()
            env_info = self.env.step(actions)[self.brain_name]
            next_obs = env_info.vector_observations 
            rewards = env_info.rewards 
            terminals = np.array(env_info.local_done).astype(int)
            self.online_rewards += rewards
            rewards = self.reward_normalizer(rewards)

            storage.add(prediction)
            storage.add({'r': torch.from_numpy(rewards).float().to(self.device).unsqueeze(-1),
                         'm': torch.from_numpy(1 - terminals).float().to(self.device).unsqueeze(-1),
                         's': torch.from_numpy(obs).float().to(self.device)})
            obs = next_obs

        self.episode_rewards.append(self.online_rewards.mean())
        self.online_rewards = np.zeros(self.num_agents)
        prediction = self.network(torch.from_numpy(obs).float().to(self.device))
        storage.add(prediction)
        storage.placeholder()

        advantages = torch.from_numpy(np.zeros((len(env_info.agents), 1))).float().to(self.device)
        returns = prediction['v'].detach()
        for i in reversed(range(rollout_length)):
            returns = storage.r[i] + discount * storage.m[i] * returns
            if not use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * gae_tau * discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(optimization_epochs):
            sampler = utils.random_sample(np.arange(states.size(0)), mini_batch_size)
            for batch_indices in sampler:
                batch_indices = torch.from_numpy(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - ppo_ratio_clip,
                                          1.0 + ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), gradient_clip)
                self.opt.step()

        steps = rollout_length * self.num_agents
        self.total_steps += steps

