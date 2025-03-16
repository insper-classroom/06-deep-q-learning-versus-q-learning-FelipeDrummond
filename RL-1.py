import gymnasium as gym
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from q_learning_functions import QLearning

env_name = "MountainCar-v0"
env = gym.make(env_name)

env.reset()

q_learning = QLearning(env)


alpha = 0.1
gamma = 0.99
epsilon = 0.9
epsilon_min = 0.0
epsilon_dec = 0.999
episodes = 5000
n_experiments = 5
max_steps = 1000
q_learning_data = {
    'Episode': [], 
    'Reward': [],
    'Experiment': []  
}

all_rewards_q = []
arraived_goal_q_learning = 0

for i in range(n_experiments):
    q_learning = QLearning(env)
    
    rewards_q = q_learning.train(alpha=alpha, gamma=gamma, epsilon=epsilon, 
                                epsilon_decay=epsilon_dec, episodios=episodes, 
                                epsilon_end=epsilon_min, max_steps=max_steps)
    
    param_str = f'α={alpha}, γ={gamma}'
    for episode, reward in enumerate(rewards_q):
        q_learning_data['Episode'].append(episode)
        q_learning_data['Reward'].append(reward)
        q_learning_data['Experiment'].append(i)
        if reward < 1000:
            arraived_goal_q_learning += 1


q_learning_df = pd.DataFrame(q_learning_data)

# Prepare Q-Learning data
window_size = 50
q_learning_df = q_learning_df.groupby(['Experiment', 'Episode']).mean().reset_index()
q_learning_df['Rolled_Reward'] = q_learning_df.groupby(['Experiment'])['Reward'].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean())
q_learning_df['Method'] = 'Q-Learning'

# Load and prepare DQN data
dqn_df = pd.read_csv('results/dqn_results.csv')
dqn_df = dqn_df.rename(columns={'Run': 'Experiment', 'Rolling_Average': 'Rolled_Reward'})
dqn_df['Method'] = 'Deep Q-Learning'

# Combine and plot
combined_df = pd.concat([q_learning_df[['Episode', 'Rolled_Reward', 'Method']], 
                        dqn_df[['Episode', 'Rolled_Reward', 'Method']]], 
                        ignore_index=True)

plt.figure(figsize=(15, 8))
sns.lineplot(data=combined_df, 
            x='Episode', 
            y='Rolled_Reward',
            hue='Method',
            errorbar=('sd', 1))

plt.title(f'Q-Learning vs Deep Q-Learning: Rolling Average Reward\n({window_size} episodes window)')
plt.grid(True, alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

