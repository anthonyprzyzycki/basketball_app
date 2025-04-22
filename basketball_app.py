import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
import tqdm
import uuid 




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

def simulate_shot(x, v, alpha, reward_mode = "distance", eps=1e-4):
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

    if reward_mode == "distance":
        return -0.5*distance
    elif reward_mode == "inverse": 
        return 1 / (eps + distance*2)
    elif reward_mode == "sparse":
        return 1 if distance <= 0.15 else -1
    else:
        return -distance

def train(agent, x_slider_val, reward_mode, episodes=2000):
    reward_list=[]
    for itrain in tqdm.tqdm(range(episodes)):
        x_pos = x_slider_val
        state = np.array([(x_pos-15)/5])
        action_type, action_L, x_pos = agent.select_action(state)
        x_pos_in = x_pos*5+15
        last_reward_L=[]
        
        for action in action_L:
            reward = simulate_shot(x_pos_in, action[0], action[1], reward_mode= reward_mode)
            agent.rewards.append(reward)
            last_reward_L.append(reward)
        reward_list.append(np.array(agent.rewards).mean())
        agent.update_policy()

    return reward_list,last_reward_L,action_L, x_pos_in


def main():
    st.title("Basketball Shot Training Simulator 🏀")

    st.markdown("Gamma (controls how much future rewards are valued compared to immediate rewards).")
    gamma = st.slider("Gamma", 0.8, 0.999, 0.97, step = .001)

    st.markdown("Learning Rate (determines how quickly the model updates based on new experiences).")
    lr = st.slider("Learning Rate", 0.0001, .02, .005, step = .0001)

    st.markdown("Reward Function (defines how the agent is penalized or rewarded for its shot attempt).")
    reward_mode = st.selectbox("Select Reward Function", 
                               options=["distance", "inverse", "sparse"],
                               format_func=lambda x: {
                                   "distance": "-0.5 * distance (hybrid)",
                                   "inverse": "1 / (eps + distance * 2) (hybrid)",
                                   "sparse": "1 for a make, -1 for a miss  (sparse)"
                               }[x])
    x_slider_val = st.slider("Enter the initial X position that the agent will shoot from.", -5, 5, 1, key="x_slider_unique")

    agent = REINFORCE(lr=lr, gamma=gamma)

    if st.button("Start Training"):
        reward_L, last_reward_L, action_L, x_pos_in = train(agent, x_slider_val, reward_mode)
        
        # save to session state
        st.session_state.reward_L = reward_L
        st.session_state.last_reward_L = last_reward_L
        st.session_state.action_L = action_L
        st.session_state.x_pos_in = x_pos_in

    # show results
    if "reward_L" in st.session_state:
        fig, ax = plt.subplots()
        ax.plot(st.session_state.reward_L)
        ax.set_ylim(-5, 1.2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Reward")
        ax.set_title("Training Performance")
        st.pyplot(fig)



if __name__ == "__main__": 
    main()

#plt.plot(reward_L)
#plt.ylim(-5,0)
#plt.xlabel("Episode")
#plt.ylabel("Average Reward")
#plt.title("Training Performance")
