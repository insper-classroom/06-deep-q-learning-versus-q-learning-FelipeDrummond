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
import seaborn as sns
import pandas as pd
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
epsilon_dec = 0.995
episodes = 200
batch_size = 32
memory = deque(maxlen=10000)
max_steps = 1000
learning_rate = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

all_rewards = []
n_runs = 5
window_size = 20

for run in range(n_runs):
    print(f"\nStarting Run {run + 1}/{n_runs}")
    
    model = DQNModel(env.observation_space.shape[0], env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    DQN = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, 
                        batch_size, memory, model, optimizer, max_steps, device)
    rewards = DQN.train()
    all_rewards.append(rewards)

df_rewards = pd.DataFrame()
for run in range(n_runs):
    df_run = pd.DataFrame({
        'Episode': range(episodes + 1),
        'Reward': all_rewards[run],
        'Run': f'Run {run + 1}'
    })
    df_rewards = pd.concat([df_rewards, df_run], ignore_index=True)

df_rewards['Rolling_Average'] = df_rewards.groupby('Run')['Reward'].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean()
)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_rewards, x='Episode', y='Rolling_Average', 
            color='blue', label='Average across runs',
            ci='sd') 

plt.xlabel('Episodes')
plt.ylabel('Rolling Average Reward')
plt.title(f'Average Training Progress over {n_runs} Runs (Rolling Window: {window_size} episodes)')
plt.grid(True, alpha=0.3)
plt.show()
 
results_df = pd.DataFrame(all_rewards).T
results_df.columns = [f'Run_{i+1}' for i in range(n_runs)]
results_df.to_csv('results/mountaincar_multiple_runs.csv', index=True)

best_run = np.argmax([np.mean(rewards) for rewards in all_rewards])
torch.save(model.state_dict(), f'data/model_mountaincar_best_run_{best_run+1}.pth')

