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


class RadialLayer(nn.Module):
    def __init__(self, dim):
        super(RadialLayer, self).__init__()

        # Initialize parameters
        self.z0 = nn.Parameter(torch.rand(dim))
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

    def forward(self, x):
        # Parametrize b to ensure invertibility.
        b_hat = -self.a + torch.log1p(torch.exp(self.b))

        x_z0 = x - self.z0
        h = 1./(self.a + torch.norm(x_z0, dim=1, keepdim=True))

        y = x + b_hat*h.expand_as(x_z0)*x_z0

        return y


class PlanarFlow(nn.Module):
    def __init__(self, dim, layers=5):
        super(PlanarFlow, self).__init__()

        self.layers = nn.ModuleList([PlanarLayer(dim) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RadialFlow(nn.Module):
    def __init__(self, dim, layers=5):
        super(RadialFlow, self).__init__()

        self.layers = nn.ModuleList([RadialLayer(dim) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
