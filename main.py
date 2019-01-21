import os
import torch
import numpy as np 
import argparse

from unityagents import UnityEnvironment

from PPO_agent import PPOAgent
from DDPG_agent import DDPGAgent


def main(args):
    # initialize the environment and the agent
    if args.algo == 'PPO':
        env = UnityEnvironment(file_name=args.env_PPO)
        agent = PPOAgent(env, state_size=33, action_size=4, num_agents=20)
    else:
        env = UnityEnvironment(file_name=args.env_DDPG)
        agent = DDPGAgent(env, state_size=33, action_size=4, random_seed=10)

    # start training
    i_episode = 0
    while True:
        i_episode += 1
        agent.episode()
        rewards = agent.episode_rewards[-100:]
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(rewards)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(rewards)))
        if np.mean(rewards)>=5.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(rewards)))
            if args.algo == 'PPO':
                torch.save(agent.network.state_dict(), 
                    os.path.join(args.checkpoint_path, 'checkpoint_PPO_network.pth'))
            else:
                torch.save(agent.actor_local.state_dict(), 
                    os.path.join(args.checkpoint_path, 'checkpoint_DDPG_actor.pth'))
                torch.save(agent.critic_local.state_dict(), 
                    os.path.join(args.checkpoint_path, 'checkpoint_DDPG_critic.pth'))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drlnd Navigation Project",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_PPO', default="unity_env/Reacher_v2.app", 
        help='twenty agents version of environment')
    parser.add_argument('--env_DDPG', default="unity_env/Reacher_v1.app", 
        help='one agent version of environment')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/', 
        help='checkpoint path to save')
    parser.add_argument('--algo', choices=['PPO','DDPG'], default='DDPG', help='which algo to train')

    args = parser.parse_args()

    main(args)
