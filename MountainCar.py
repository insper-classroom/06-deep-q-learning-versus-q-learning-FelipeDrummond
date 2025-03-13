import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from DeepQLearning import DeepQLearning
import csv
from collections import deque
import os
from dqn_model import DQNModel
torch.manual_seed(0)
np.random.seed(0)

env = gym.make('MountainCar-v0')
print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

os.makedirs('results', exist_ok=True)
os.makedirs('data', exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQNModel(env.observation_space.shape[0], env.action_space.n)
model = model.to(device)

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_dec = 0.99
episodes = 200
batch_size = 64
memory = deque(maxlen=10000)
max_steps = 500
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

DQN = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, 
                    batch_size, memory, model, optimizer, max_steps, device)
rewards = DQN.train()

plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('# Rewards')
plt.title('# Rewards vs Episodes')
plt.savefig("results/mountaincar_DeepQLearning.jpg")     
plt.close()

with open('results/mountaincar_DeepQLearning_rewards.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    for episode, reward in enumerate(rewards):
        writer.writerow([episode, reward])

torch.save(model.state_dict(), 'data/model_mountaincar.pth')

