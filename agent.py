import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from environment import Environment

from collections import deque
import random

MAX_MEMORY = 100
# BATCH_SIZE = 100
LR = 0.001

class Agent:
    def __init__(self, model, data_path):
        self.model = model
        self.memory = deque(maxlen=MAX_MEMORY)
        # self.batch_size = BATCH_SIZE
        raw_data = pd.read_csv(data_path)
        self.env = Environment(raw_data)
        self.lr=LR
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def get_state(self):
        return self.env.get_state()

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        prediction = self.model(state)
        return torch.argmax(prediction).item()
    
    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def train_long_memory(self):
        # if len(self.memory) > BATCH_SIZE:
        #     mini_sample = random.sample(self.memory, BATCH_SIZE)
        # else:
        mini_sample = self.memory
        states, actions, rewards = zip(*mini_sample)
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).reshape(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.long).reshape(-1, 1)
        return self.train_step(states, actions, rewards)

    def train_short_memory(self, state, action, reward):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).reshape(1, -1)
        reward = torch.tensor(reward, dtype=torch.long).reshape(1, -1)
        return self.train_step(state, action, reward)

    def get_target(self, actions, rewards):
        # trick:
        # if action is "sell" and reward is positive, then target is "sell"
        # if action is "sell" and reward is negative, then target is "buy"
        # if action is "sell" and reward is zero, then target is "hold"
        # if action is "buy" and reward is positive, then target is "buy"
        # if action is "buy" and reward is negative, then target is "sell"
        # if action is "buy" and reward is zero, then target is "hold"
        # if action is "hold" and reward is positive, then target is "buy"
        # if action is "hold" and reward is negative, then target is "sell"
        # if action is "hold" and reward is zero, then target is "hold"
        # "buy"->0, "sell"->1, "hold"->2
        target = torch.zeros(len(actions), dtype=torch.long)
        for i, (a, r) in enumerate(zip(actions, rewards)):
            if r==0:
                target[i] = 2
                continue
            if a==1:
                target[i] = int(r>0)
            else:
                target[i] = int(r<0)

        target.reshape(-1, 1)
        return target
    
    def train_step(self, states, actions, rewards):
        target = self.get_target(actions, rewards)
        self.optimizer.zero_grad()
        output = self.model(states)
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":

    esn_input_size = 4
    esn_reservoir_size = 1024
    esn_output_size = 128
    lstm_input_size = 128
    lstm_num_hidden = 512
    lstm_num_layers = 3
    lstm_output_size = 3
    
    from models import ESN, LSTM
    esn = ESN(esn_input_size, esn_reservoir_size, esn_output_size)
    lstm = LSTM(lstm_input_size, lstm_num_hidden, lstm_num_layers, lstm_output_size, esn)
    agent = Agent(lstm, "crypto_data/train_data.csv")
    state = agent.env.reset()

    while True:
        action = agent.get_action(state)
        new_state, reward, done = agent.env.step(action)
        agent.remember(state, action, reward)
        print(f"state: {state}, action: {action}, reward: {reward}")
        agent.train_short_memory(state, action, reward)
        if done:
            loss = agent.train_long_memory()
            print(f"loss: {loss}")
            break