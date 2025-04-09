from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import streamlit as st




class ShootingNetwork(nn.Module):
    def __init__(self):
        super(ShootingNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)  # Output: speed and angle
        self.fc4 = nn.Linear(16, 2)  # Output: speed and angle stddev
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        means = F.sigmoid(self.fc3(x)) * torch.tensor([25.0, np.pi / 2.5])
        means[0]+=5
        stddevs = F.sigmoid(self.fc4(x))*torch.tensor([1.0, 0.1])
        return means, stddevs

class REINFORCE():
    def __init__(self, lr=0.005, gamma=0.97):
        self.shooting_policy = ShootingNetwork()
        self.optimizer = optim.Adam(
            list(self.shooting_policy.parameters()), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.n_sample=40  # this is the number of samples drawn from the distribution predicted as a function of the state
        self.rewards = []
        self.loss=torch.tensor(0.0)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        shot_params, shot_stddev = self.shooting_policy(state)
        dist = torch.distributions.Normal(shot_params, shot_stddev+0.005)
        sampled_shot_L=[]
        log_prob_L=[]
        for i in range(self.n_sample):
            sampled_shot = dist.sample()
            sampled_shot_L.append(sampled_shot)
            log_prob = dist.log_prob(sampled_shot_L[-1]).sum()
            log_prob_L.append(log_prob)
        self.log_probs=log_prob_L
        return "shoot", [sampled_shot.detach().numpy() for sampled_shot in sampled_shot_L], state.item()

    def update_policy(self):
        self.loss=torch.tensor(0.0)
        for i,rewards in enumerate(self.rewards):
          self.loss += -self.log_probs[i]*rewards
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.log_probs = []
        self.rewards = []

def simulate_shot(x, v, alpha):
    z = 1.8  # Initial height
    dt = 0.01
    vh = v * np.cos(alpha)
    vv = v * np.sin(alpha)
    # the reward needs to be a continous
    while x<30:
        x += vh * dt
        z += vv * dt
        vv -= 9.8 * dt
    distance=np.abs(z-3.05)
    return -0.5*distance
import tqdm
def train(agent, episodes=2000):
    reward_list=[]
    for itrain in tqdm.tqdm(range(episodes)):
        x_pos = np.random.randn()*5+15  # Initial player position
        state = np.array([(x_pos-15)/5])
        action_type, action_L, x_pos = agent.select_action(state)
        x_pos_in = x_pos*5+15
        last_reward_L=[]
        for action in action_L:
            reward = simulate_shot(x_pos_in, action[0], action[1])
            agent.rewards.append(reward)
            last_reward_L.append(reward)
        reward_list.append(np.array(agent.rewards).mean())

        agent.update_policy()
    return reward_list,last_reward_L,action_L, x_pos_in

if __name__ == "__main__":
    agent = REINFORCE()
    reward_L,last_reward_L,action_L,x_pos_in=train(agent)

import matplotlib.pyplot as plt
plt.plot(reward_L)
plt.ylim(-5,0)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Training Performance")