import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, LR=0.001):
        super(ESN, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size

        # Initialize reservoir weights
        self.Win = nn.Parameter(torch.randn(reservoir_size, input_size))
        self.W = nn.Parameter(torch.randn(reservoir_size, reservoir_size))

        # Scaling W to have spectral radius = spectral_radius
        self.W.data *= spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.W)))

        # Output layer
        self.Wout = nn.Linear(reservoir_size, output_size)

        # lr, loss and optimizer
        self.lr=LR
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)

    def forward(self, input_data, initial_state=None):
        if initial_state is None:
            state = torch.zeros((input_data.size(0), self.reservoir_size)).to(device)
        else:
            state = initial_state

        state = torch.tanh(torch.matmul(input_data, self.Win.t()) + torch.matmul(state, self.W.t()))
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

    def forward(self, x):
        # pass the input through the ESN
        x = self.ESN(x)

        # run through the RNN layer
        y, hidden = self.lstm(x)

        # pass the RNN output through the linear output layer
        o = self.out(y)

        return o[-1]