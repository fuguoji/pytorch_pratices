import torch
import torch.nn as nn


class RNNLMModel(torch.nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embed_size, 
                 hidden_size, 
                 num_layers,
                 batch_first=True):
        super(RNNLMModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=self.batch_first)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        output, (h, c) = self.lstm(x, h)
        output = output.reshape(output.size(0)*output.size(1), output.size(2))
        output = self.linear(output)

        return output, (h, c)

    