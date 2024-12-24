import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams  

        self.rnn = nn.RNN(
            input_size=self.hyperparams.INPUT_SIZE,
            hidden_size=self.hyperparams.HIDDEN_SIZE,
            num_layers=self.hyperparams.NUM_LAYERS,
            batch_first=True,
            nonlinearity='tanh'
        )

        self.linear = nn.Linear(self.hyperparams.HIDDEN_SIZE, self.hyperparams.OUTPUT_SIZE)

    def forward(self, x):
        h0 = torch.zeros(self.hyperparams.NUM_LAYERS, x.size(0), self.hyperparams.HIDDEN_SIZE).to(x.device)

        x, hn = self.rnn(x, h0)

        x = self.linear(x[:, -1, :])

        return x
