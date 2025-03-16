import gymnasium as gym
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dqn_model import DQNModel
from q_learning_functions import QLearning

# Setup environment
env = gym.make('MountainCar-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
n_episodes = 100  # Number of evaluation episodes
max_steps = 1000
inference_data = {
    'Episode': [],
    'Reward': [],
    'Steps': [],
    'Method': []
}

# Load DQN model
dqn_model = DQNModel(env.observation_space.shape[0], env.action_space.n)
dqn_model.load_state_dict(torch.load('data/model_mountaincar_best_run_5.pth'))
dqn_model.to(device)
dqn_model.eval()

# Use the already trained Q-Learning instance
q_learning = QLearning(env)
# Train Q-Learning if not already trained
alpha = 0.1
gamma = 0.99
epsilon = 0.9
epsilon_min = 0.0
epsilon_dec = 0.999
episodes = 5000
q_learning.train(alpha=alpha, gamma=gamma, epsilon=epsilon, 
                epsilon_decay=epsilon_dec, episodios=episodes, 
                epsilon_end=epsilon_min, max_steps=max_steps)

# Evaluate DQN
for episode in range(n_episodes):
    state, _ = env.reset()
    episode_reward = 0
    steps = 0
    done = False
    truncated = False
    
    while not (done or truncated) and steps < max_steps:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            Q_values = dqn_model(state_tensor)
        action = torch.argmax(Q_values).item()
        
        state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        steps += 1
    
    inference_data['Episode'].append(episode)
    inference_data['Reward'].append(episode_reward)
    inference_data['Steps'].append(steps)
    inference_data['Method'].append('Deep Q-Learning')

# Evaluate Q-Learning using robust_run
mean_steps, std_steps, mean_reward, std_reward = q_learning.robust_run()

# Add Q-Learning results
for episode in range(n_episodes):
    inference_data['Episode'].append(episode)
    inference_data['Reward'].append(mean_reward)
    inference_data['Steps'].append(mean_steps)
    inference_data['Method'].append('Q-Learning')

# Create DataFrame and plot
df = pd.DataFrame(inference_data)

# Plot Rewards
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='Method', y='Reward')
plt.title('Reward Distribution During Inference')
plt.grid(True, alpha=0.3)

# Plot Steps
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='Method', y='Steps')
plt.title('Steps Distribution During Inference')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistics
print("\nPerformance Statistics:")
for method in ['Q-Learning', 'Deep Q-Learning']:
    method_data = df[df['Method'] == method]
    print(f"\n{method}:")
    print(f"Average Reward: {method_data['Reward'].mean():.2f} ± {method_data['Reward'].std():.2f}")
    print(f"Average Steps: {method_data['Steps'].mean():.2f} ± {method_data['Steps'].std():.2f}")
    print(f"Success Rate: {(method_data['Reward'] > -200).mean() * 100:.2f}%") 