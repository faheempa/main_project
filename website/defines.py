import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9):
        super(ESN, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size

        # Initialize reservoir weights
        self.Win = nn.Parameter(torch.randn(reservoir_size, input_size))
        self.W = nn.Parameter(torch.randn(reservoir_size, reservoir_size))

        # Scaling W to have spectral radius = spectral_radius
        self.W.data *= spectral_radius / torch.max(
            torch.abs(torch.linalg.eigvals(self.W))
        )

        # Output layer
        self.Wout = nn.Linear(reservoir_size, output_size)

    def forward(self, input_data, initial_state=None):
        if initial_state is None:
            state = torch.zeros((input_data.size(1), self.reservoir_size)).to(device)
        else:
            state = initial_state

        state = torch.tanh(
            torch.matmul(input_data, self.Win.t()) + torch.matmul(state, self.W.t())
        )
        state = torch.tanh(self.Wout(state))
        return state


class LSTM(nn.Module):
    def __init__(self, input_size, num_hidden, num_layers, output_size, esn):
        super().__init__()

        # store parameters
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers

        # RNN Layer (notation: LSTM \in RNN)
        self.lstm = nn.LSTM(input_size, num_hidden, num_layers)

        # linear layer for output
        self.out = nn.Linear(num_hidden, output_size)

        # esn
        self.ESN = esn

    def forward(self, x, print_shapes=False):
        print(f"Input: {list(x.shape)}") if print_shapes else None
        # pass the input through the ESN
        x = self.ESN(x)
        print(f"Output-ESN: {list(x.shape)}") if print_shapes else None

        # run through the RNN layer
        y, hidden = self.lstm(x)
        print(f"RNN-cell: {list(hidden[1].shape)}") if print_shapes else None
        print(f"RNN-hidden: {list(hidden[0].shape)}") if print_shapes else None
        print(f"RNN-out: {list(y.shape)}") if print_shapes else None

        # pass the RNN output through the linear output layer
        o = self.out(y)
        o = o[:, -1, :]
        print(f"Output: {list(o.shape)}") if print_shapes else None

        return o


class Environment:
    def __init__(self, data, history_length=10, batch_size=30):
        self.price = data["Close"][0]
        self.dataX, self.dataY = self._make_data(
            self._preprocess_data(data), history_length
        )
        self.history_length = history_length
        self.batch_size = batch_size

    def reset(self):
        # get the date iter
        self.data_iter = self._make_data_iter(
            self.dataX, self.dataY, batch_size=self.batch_size
        )
        # get first month data
        self.current_month_data, _ = next(self.data_iter)
        # set day 0
        self.current_day = 0
        # set done to false
        self.done = False
        # set profit and current order
        self.change = 0
        self.current_order = None
        # return the first day data
        return self.current_month_data[self.current_day]

    def execute_action(self, action):
        # get the close price of the current day
        close_price = self.current_month_data[self.current_day][-1][3]
        # if the current order is not none, then calculate the profit
        if self.current_order == "buy":
            self.change = close_price
        elif self.current_order == "sell":
            self.change = -close_price
        else:
            self.change = 0
        # update the price
        self.price += close_price
        # set the current order to the action
        self.current_order = "buy" if action == 0 else "sell" if action == 1 else None

    def step(self, action):
        # actions are 0-buy, 1-sell, 2-hold
        self.execute_action(action)

        if self.done:
            # month ended, get next month data
            self.current_month_data, _ = next(self.data_iter)
            self.current_day = 0
        else:
            # next day
            self.current_day += 1

        # get the next day close price to calculate reward
        next_day_close_price = self.current_month_data[self.current_day][-1][3]

        # reward 1 means profit, -1 means loss, 0 means no profit no loss
        if action == 1:
            reward = (
                1
                if next_day_close_price < -10
                else -1 if next_day_close_price > 10 else 0
            )
        else:
            reward = (
                1
                if next_day_close_price > 10
                else -1 if next_day_close_price < -10 else 0
            )

        # check if the month ended
        self.done = True if self.current_day == self.batch_size - 1 else False

        #  return the next day data, reward and done
        return self.current_month_data[self.current_day], reward, self.done, self.change

    def _preprocess_data(self, df):
        df = df.dropna()
        df = df.drop(["Volume", "Adj Close"], axis=1)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        return df

    def _make_data(self, df, n_steps):
        X, y = [], []
        for i in range(1, len(df) - n_steps):
            # get x values, which will be the difference between the current day and the previous day
            X.append(
                df.iloc[i : i + n_steps, 1:].values
                - df.iloc[i - 1 : i + n_steps - 1, 1:].values
            )
            # y will be the difference between the close price of the next day and the current day
            t = df.iloc[i + n_steps, 4] - df.iloc[i + n_steps - 1, 4]
            # if the value is greater than 10, then the value will be 0 - buy
            # if the value is less than 10, then the value will be 1 - sell
            # else the value will be 2 - hold
            t = 0 if t > 10 else 1 if t < -10 else 2
            y.append(t)
        x, y = np.array(X), np.array(y)
        y = y.reshape(-1, 1)
        return x, y

    def _make_data_iter(self, x, y, batch_size=1, shuffle=False):
        # convert data to tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        # convert to tensor dataset and then to data loader
        data_set = TensorDataset(x, y)
        data_loader = DataLoader(
            data_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
        )
        return iter(data_loader)


from collections import deque

MAX_MEMORY = 100
LR = 0.001


class Agent:
    def __init__(self, model, data_path):
        self.model = model
        self.memory = deque(maxlen=MAX_MEMORY)
        raw_data = pd.read_csv(data_path)
        self.env = Environment(raw_data)
        self.lr = LR
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def get_state(self):
        return self.env.get_state()

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(
            0
        )  # [10, 4] -> [1, 10, 4]
        prediction = self.model(state)
        return torch.argmax(prediction).item()

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def train_long_memory(self):
        mini_sample = self.memory
        states, actions, rewards = zip(*mini_sample)
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).reshape(
            -1, 1
        )  # [1, 0, 1] -> [[1], [0], [1]]
        rewards = torch.tensor(rewards, dtype=torch.long).reshape(-1, 1)
        return self.train_step(states, actions, rewards)

    def train_short_memory(self, state, action, reward):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).reshape(1, -1)
        reward = torch.tensor(reward, dtype=torch.long).reshape(1, -1)
        return self.train_step(state, action, reward)

    def get_target(self, actions, rewards):
        target = torch.zeros(len(actions), dtype=torch.long)
        for i, (a, r) in enumerate(zip(actions, rewards)):
            if r == 0:
                target[i] = 2
                continue
            if a == 1:
                target[i] = int(r > 0)
            else:
                target[i] = int(r < 0)

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
