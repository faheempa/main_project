import torch
import torch.nn as nn

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
        self.W.data *= spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.W)))

        # Output layer
        self.Wout = nn.Linear(reservoir_size, output_size)


    def forward(self, input_data, initial_state=None):
        if initial_state is None:
            state = torch.zeros((input_data.size(1), self.reservoir_size)).to(device)
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


if __name__=="__main__":
    esn_input_size = 4
    esn_reservoir_size = 1024
    esn_output_size = 128
    lstm_input_size = 128
    lstm_num_hidden = 512
    lstm_num_layers = 3
    lstm_output_size = 3

    # test
    esn = ESN(esn_input_size, esn_reservoir_size, esn_output_size)
    lstm = LSTM(lstm_input_size, lstm_num_hidden, lstm_num_layers, lstm_output_size, esn)
    input_data = torch.randn(5, 10, esn_input_size)
    output = lstm(input_data, print_shapes=True)
    print(output.shape)

    print(f"output: {output}")

    actions = torch.argmax(output, dim=1)
    for action in actions:
        if action == 0:
            print("action: buy")
        elif action == 1:
            print("action: sell")
        else:
            print("action: hold")
