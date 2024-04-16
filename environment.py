import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Environment:
    def __init__(self, data, history_length=10, batch_size=30):
        # preprocess and make data in needed format
        self.dataX, self.dataY = self._make_data(self._preprocess_data(data), history_length)
        self.history_length = history_length
        self.batch_size = batch_size

    def reset(self):
        # get the date iter
        self.data_iter = self._make_data_iter(self.dataX, self.dataY, batch_size=self.batch_size)
        # get first month data
        self.current_month_data, _ = next(self.data_iter)
        # set day 0
        self.current_day = 0
        # set done to false
        self.done = False
        # set profit and current order
        self.profit = 0
        self.current_order = None
        # return the first day data
        return self.current_month_data[self.current_day]

    def execute_action(self, action):
        # get the close price of the current day
        close_price = self.current_month_data[self.current_day][-1][3]
        # if the current order is not none, then calculate the profit
        if self.current_order is not None:
            if self.current_order == "buy":
                self.profit += close_price
            else:
                self.profit -= close_price
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
            reward = 1 if next_day_close_price < -10 else -1 if next_day_close_price > 10 else 0
        else:
            reward = 1 if next_day_close_price > 10 else -1 if next_day_close_price < -10 else 0

        # check if the month ended
        self.done = True if self.current_day == self.batch_size - 1 else False

        #  return the next day data, reward and done
        return self.current_month_data[self.current_day], reward, self.done

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
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        return iter(data_loader)


if __name__ == "__main__":
    try:
        raw_data = pd.read_csv("crypto_data/train_data.csv")
        env = Environment(raw_data)
        price = env.reset()
        next_day_price, reward, done = env.step(0) # buy
        print(next_day_price, reward, done)

        for i in range(10000):
            price, reward, done = env.step(0)
            print(i)

    except StopIteration:
        print("Iteration ended")
