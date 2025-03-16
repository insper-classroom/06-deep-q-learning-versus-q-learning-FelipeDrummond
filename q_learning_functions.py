import random
import numpy as np
import gymnasium as gym
class QLearning:
    def __init__(self, env):
        self.env = env
        self.num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
        self.num_states = np.round(self.num_states, 0).astype(int) + 1
        self.q_table = np.zeros([self.num_states[0], self.num_states[1], env.action_space.n])

    def select_action(self, state_adj, epsilon):
        rv = random.uniform(0, 1)
        if rv < epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state_adj[0], state_adj[1]])

    def update_q_table(self, alpha, state_adj, action, reward, gamma, new_state_adj):
        self.q_table[state_adj[0], state_adj[1], action] = self.q_table[state_adj[0], state_adj[1], action] + alpha*(reward+gamma*max(self.q_table[new_state_adj[0], new_state_adj[1]]) - self.q_table[state_adj[0], state_adj[1], action])


    def train(self, alpha, gamma, epsilon, epsilon_decay, episodios, epsilon_end, max_steps):
        rewards = []
        for episode in range(episodios):
            
            state, info = self.env.reset()
            done = False
            step = 0
            accumulated_reward = 0
            
            while not done and step < max_steps:
                state_adj = self.transform_state(state)
                action = self.select_action(state_adj, epsilon)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state_adj = self.transform_state(next_state)
                accumulated_reward += reward
                self.update_q_table(alpha=alpha, state_adj=state_adj, action=action, reward=reward, gamma=gamma, new_state_adj=next_state_adj)
                state = next_state
                step += 1
                
            rewards.append(accumulated_reward)
            if epsilon > epsilon_end:
                epsilon = epsilon * epsilon_decay
        return rewards

    def run(self):
        self.env = gym.make("MountainCar-v0", render_mode="human")
        state, info = self.env.reset()
        done = False
        actions = 0
        rewards = 0
        while not done:
            state_adj = self.transform_state(state)
            action = self.select_action(state_adj, 0)
            next_state, reward, done, truncated, info = self.env.step(action)
            state = next_state
            actions += 1
            rewards += reward

        return actions, rewards

    def robust_run(self):
        list_rewards = []
        list_actions = []
        for i in range(100):
            state, info = self.env.reset()
            done = False
            rewards = 0
            actions = 0
            while not done:
                state_adj = self.transform_state(state)
                action = self.select_action(state_adj, 0)
                next_state, reward, done, truncated, info = self.env.step(action)
                state = next_state
                rewards += reward
                actions += 1

            list_actions.append(actions)
            list_rewards.append(rewards)

        return np.mean(list_actions), np.std(list_actions), np.mean(list_rewards), np.std(list_rewards)

    def transform_state(self, state):
        state_adj = (state - self.env.observation_space.low)*np.array([10, 100])
        return np.round(state_adj, 0).astype(int)