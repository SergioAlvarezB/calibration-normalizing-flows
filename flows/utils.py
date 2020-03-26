import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim, hidden_size=[], activation=F.relu):
        super(MLP, self).__init__()
        self.activation = activation
        units = [dim] + hidden_size + [dim]
        self.layers = nn.ModuleList([nn.Linear(units[i], units[i+1])
                                     for i in range(len(units)-1)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        y = self.layers[-1](x)

        return y


class TempScaler(nn.Module):
    def __init__(self):
        super(TempScaler, self).__init__()

        self.T = nn.Parameter(torch.ones(1))

    def forward(self, x):
        z = x/torch.abs(self.T)

        return z

    def backward(self, z):
        x = z*torch.abs(self.T)

        return x
