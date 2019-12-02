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


class AdditiveCouplingLayer(nn.Module):
    def __init__(self, coupling_func, mask):
        super(AdditiveCouplingLayer, self).__init__()
        self.coupling_func = coupling_func
        self.mask = nn.Parameter(
                torch.as_tensor(mask.copy(), dtype=torch.float),
                requires_grad=False)

    def forward(self, x):
        y = self.mask*x + (1-self.mask)*(x + self.coupling_func(x))
        return y

    def backward(self, x):
        y = self.mask*x + (1-self.mask)*(x - self.coupling_func(x))
        return y


class NiceFlow(nn.Module):
    def __init__(self, dim, **kwargs):
        super(NiceFlow, self).__init__()

        layers = kwargs.get('layers', 5)
        hidden_size = kwargs.get('hidden_size', [dim])

        mask = np.zeros((1, dim))
        mask[:, dim//2:] = 1.

        mlps = [MLP(dim, hidden_size) for _ in range(layers)]
        layers = []

        for mlp in mlps:
            layers.append(AdditiveCouplingLayer(mlp, mask))
            mask = np.flip(mask)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, x):
        for l in self.layers[::-1]:
            x = l.backward(x)

        return x
