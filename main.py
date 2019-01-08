from unityagents import UnityEnvironment
import numpy as np 


from network import *
from PPO_agent import PPOAgent


env = UnityEnvironment(file_name='Reacher.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

network = GaussianActorCriticNet(state_size, action_size, 
    actor_body=FCBody(state_size, gate=F.tanh),critic_body=FCBody(state_size, gate=F.tanh))

optimizer = torch.optim.Adam(network.parameters(), 3e-4, eps=1e-5)

agent = PPOAgent(env, states, brain_name, num_agents, network, optimizer, DEVICE)
agent.step()


