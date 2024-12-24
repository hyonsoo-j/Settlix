import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams 

        self.lstm = nn.LSTM(
            input_size=self.hyperparams.INPUT_SIZE,
            hidden_size=self.hyperparams.HIDDEN_SIZE,
            num_layers=self.hyperparams.NUM_LAYERS,
            batch_first=True
        )
        self.linear = nn.Linear(self.hyperparams.HIDDEN_SIZE, self.hyperparams.OUTPUT_SIZE)

    def forward(self, x):
        h0 = torch.zeros(self.hyperparams.NUM_LAYERS, x.size(0), self.hyperparams.HIDDEN_SIZE).to(x.device)
        c0 = torch.zeros(self.hyperparams.NUM_LAYERS, x.size(0), self.hyperparams.HIDDEN_SIZE).to(x.device)

        x, (hn, cn) = self.lstm(x, (h0, c0))

        x = self.linear(x[:, -1, :])
        return x
