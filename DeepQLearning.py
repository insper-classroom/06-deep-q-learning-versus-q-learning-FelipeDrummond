import numpy as np
import random
import torch
import torch.nn as nn
import gc
import keras

class DeepQLearning:

    #
    # Implementacao do algoritmo proposto em 
    # Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013
    # https://arxiv.org/abs/1312.5602
    #

    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, 
                 episodes, batch_size, memory, model, optimizer, max_steps, device):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = memory
        self.model = model
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.device = device
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_values = self.model(state)
            return action_values.argmax().item()

    # cria uma memoria longa de experiencias
    def experience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def experience_replay(self):
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            
            # Convert batch to tensors
            states = torch.FloatTensor(np.array([i[0] for i in batch])).squeeze(1).to(self.device)
            actions = torch.LongTensor([i[1] for i in batch]).to(self.device)
            rewards = torch.FloatTensor([i[2] for i in batch]).to(self.device)
            next_states = torch.FloatTensor(np.array([i[3] for i in batch])).squeeze(1).to(self.device)
            terminals = torch.FloatTensor([i[4] for i in batch]).to(self.device)

            # Compute current Q values
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

            # Compute next Q values
            with torch.no_grad():
                next_q_values = self.model(next_states).max(1)[0]
                target_q_values = rewards + (1 - terminals) * self.gamma * next_q_values

            # Compute loss and update weights
            loss = self.criterion(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec

    def train(self):
        rewards = []
        for i in range(self.episodes + 1):
            state, _ = self.env.reset()
            state = np.reshape(state, (1, self.env.observation_space.shape[0]))
            score = 0
            steps = 0
            done = False

            while not done:
                steps += 1
                action = self.select_action(state)
                next_state, reward, terminal, truncated, _ = self.env.step(action)
                
                if terminal or truncated or (steps > self.max_steps):
                    done = True
                
                score += reward
                next_state = np.reshape(next_state, (1, self.env.observation_space.shape[0]))
                self.experience(state, action, reward, next_state, terminal)
                state = next_state
                self.experience_replay()
                
                if done:
                    print(f'Episode: {i+1}/{self.episodes}. Score: {score}')
                    break
            
            rewards.append(score)
            gc.collect()
            keras.backend.clear_session()

        return rewards
