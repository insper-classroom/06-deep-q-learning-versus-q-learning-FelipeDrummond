import gymnasium as gym
import torch
import numpy as np
from CartPole import DQNModel  # Only import the model class

# Create environment
env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()

# Create model and load trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQNModel(env.observation_space.shape[0], env.action_space.n)
model.load_state_dict(torch.load('data/model_cart_pole.pth'))
model.to(device)
model.eval()  # Set to evaluation mode

done = False
truncated = False
rewards = 0
steps = 0
max_steps = 500

while (not done) and (not truncated) and (steps < max_steps):
    # Convert state to tensor and get model prediction
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        Q_values = model(state_tensor)
    
    # Select action with highest Q-value
    action = torch.argmax(Q_values).item()
    
    # Take action in environment
    state, reward, done, truncated, info = env.step(action)
    rewards += reward
    env.render()
    steps += 1

print(f'Score = {rewards}')
input('press a key...')