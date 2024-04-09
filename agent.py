import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import environment

from collections import deque
import random

MAX_MEMORY = 2000
BATCH_SIZE = 256
LR = 0.001

class Agent:
    def __init__(self, model, batch_size, data_path):
        self.model = model
        self.memory = deque(maxlen=MAX_MEMORY)
        self.batch_size = batch_size
        raw_data = pd.read_csv(data_path)
        self.env = environment.Environment(raw_data)

    def get_state(self):
        return self.env.get_state()

    def get_action(self, state):
        return torch.argmax(self.model(state))
    
    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards = zip(*mini_sample)
        self.train_step(states, actions, rewards)

    def train_short_memory(self, state, action, reward):
        self.train_step(state, action, reward)

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
        target = torch.zeros(len(actions))
        for i, (a, r) in enumerate(zip(actions, rewards)):
            if r==0:
                target[i] = 2
                continue
            if a==1:
                target[i] = int(r>0)
            elif a==0:
                target[i] = int(r<0)
            else:
                target[i] = int(r<0)

        target.reshape(-1, 1)
        return target
    
    def train_step(self, states, actions, rewards):
        pass


if __name__ == "__main__":
    try: 
        
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
        agent = Agent(lstm, BATCH_SIZE, "crypto_data/train_data.csv")
        state = agent.env.reset()

        while True:
            action = agent.get_action(state)
            new_state, reward, done = agent.env.step(action)
            agent.remember(state, action, reward)
            print(f"state: {state}, action: {action}, reward: {reward}")
            # agent.train_short_memory(state, action, reward)
            # if done:
            #    agent.train_long_memory()
            #    save the model
            

    except Exception as e:
        print(e)