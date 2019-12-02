import numpy as np

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


class NvpCouplingLayer(nn.Module):
    def __init__(self, s, t, mask):
        super(NvpCouplingLayer, self).__init__()
        self.s = s
        self.t = t
        self.mask = nn.Parameter(
                torch.as_tensor(mask.copy(), dtype=torch.float),
                requires_grad=False)

    def forward(self, x):
        x_b = self.mask*x
        b_1 = 1 - self.mask

        y = x_b + b_1 * (x * torch.exp(self.s(x_b)) + self.t(x_b))
        return y

    def backward(self, x):
        x_b = self.mask*x
        b_1 = 1 - self.mask

        y = x_b + b_1*((x - self.t(x_b)) / torch.exp(self.s(x_b)))
        return y


class RealNvpFlow(nn.Module):
    def __init__(self, dim, **kwargs):
        super(RealNvpFlow, self).__init__()

        # Get keyword arguments
        layers = kwargs.get('layers', 4)
        hidden_size = kwargs.get('hidden_size', [dim])

        # Create layers
        mask = np.zeros((1, dim))
        mask[:, dim//2:] = 1.

        flow_layers = []
        for l in range(layers):
            flow_layers.append(NvpCouplingLayer(
                    MLP(dim, hidden_size, torch.tanh),
                    MLP(dim, hidden_size, F.relu),
                    mask
            ))
            mask = np.flip(mask)
        self.layers = nn.ModuleList(flow_layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x

    def backward(self, x):
        for l in self.layers[::-1]:
            x = l.backward(x)

        return x
