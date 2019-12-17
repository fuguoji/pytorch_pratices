import torch
import torch.nn as nn

class RNNModel(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 num_classes):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        output, _ = self.lstm(x, (h0, c0))
        output = self.fc(output[:, -1, :])

        return output
        