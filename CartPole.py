import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from DeepQLearning import DeepQLearning
import csv
from collections import deque
import os

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Create environment
env = gym.make('CartPole-v1')
print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Define the PyTorch model
class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Create model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQNModel(env.observation_space.shape[0], env.action_space.n)
model = model.to(device)

# Define hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_dec = 0.99
episodes = 200
batch_size = 64
memory = deque(maxlen=10000)
max_steps = 500
learning_rate = 0.001

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create DQN agent
DQN = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, 
                    batch_size, memory, model, optimizer, max_steps, device)
rewards = DQN.train()

# Plot results
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('# Rewards')
plt.title('# Rewards vs Episodes')
plt.savefig("results/cartpole_DeepQLearning.jpg")     
plt.close()

# Save results
with open('results/cartpole_DeepQLearning_rewards.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    for episode, reward in enumerate(rewards):
        writer.writerow([episode, reward])

# Save model
torch.save(model.state_dict(), 'data/model_cart_pole.pth')

