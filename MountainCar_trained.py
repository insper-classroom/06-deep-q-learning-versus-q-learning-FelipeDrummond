import gymnasium as gym
import torch
import numpy as np
from dqn_model import DQNModel  

env = gym.make('MountainCar-v0', render_mode='human')
state, _ = env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQNModel(env.observation_space.shape[0], env.action_space.n)
model.load_state_dict(torch.load('data/model_mountaincar.pth'))
model.to(device)
model.eval()    

done = False
truncated = False
rewards = 0
steps = 0
max_steps = 500

while (not done) and (not truncated) and (steps < max_steps):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        Q_values = model(state_tensor)
    
    action = torch.argmax(Q_values).item()
    
    state, reward, done, truncated, info = env.step(action)
    rewards += reward
    env.render()
    steps += 1

print(f'Score = {rewards}')
input('press a key...')