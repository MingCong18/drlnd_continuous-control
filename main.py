from unityagents import UnityEnvironment
import numpy as np 

from network import *
from PPO_agent import PPOAgent


env = UnityEnvironment(file_name='Reacher.app')
state_size = 33
action_size = 4
num_agents = 20

network = GaussianActorCriticNet(state_size, action_size, 
    actor_body=FCBody(state_size, gate=F.tanh),critic_body=FCBody(state_size, gate=F.tanh))

optimizer = torch.optim.Adam(network.parameters(), 3e-4, eps=1e-5)

agent = PPOAgent(env, num_agents, network, optimizer, DEVICE)

i_episode = 0
while True:
    i_episode += 1
    agent.step()
    rewards = agent.episode_rewards[-100:]
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(rewards)), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(rewards)))
    if np.mean(rewards)>=30.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(rewards)))
        torch.save(agent.network.state_dict(), 'checkpoint.pth')
        break


