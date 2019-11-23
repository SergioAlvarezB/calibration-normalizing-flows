import numpy as np

import torch
from torch import nn


class PlanarLayer(nn.Module):
    def __init__(self, dim):
        super(PlanarLayer, self).__init__()

        # Initialize parameters.
        self.w = nn.Parameter(torch.rand(dim))
        self.u = nn.Parameter(torch.rand(dim))
        self.b = nn.Parameter(torch.rand(1))

    def forward(self, x):
        # Ensure w*u >= 1 to mantain invertibility.
        wtu = torch.dot(self.w, self.u)
        m = -1 + torch.log1p(torch.exp(wtu))

        u_hat = self.u + (m - wtu)*self.w/torch.norm(self.w)

        # Forward pass
        h = torch.tanh(torch.matmul(x, self.w) + self.b)
        y = x + torch.matmul(h.view(-1, 1), u_hat.view(1, -1))

        return y


class PlanarFlow(nn.Module):
    def __init__(self, dim, layers=5):
        super(PlanarFlow, self).__init__()

        self.layers = nn.ModuleList([PlanarLayer(dim) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
