import torch
from torch import nn


class AffineConstantLayer(nn.Module):

    def __init__(self, dim, scale=True, shift=True):
        super(AffineConstantLayer, self).__init__()

        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) \
            if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) \
            if shift else None

    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


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

        # Log-determinant
        hp = 1 - h**2
        phi = torch.matmul(hp.view(-1, 1), self.w.view(1, -1))
        det = torch.abs(1 + torch.matmul(phi, u_hat.view(-1, 1)))
        log_det = torch.log(det.squeeze())

        return y, log_det


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

        # Log-determinant
        det = log_det = torch.tensor(1.0)
        log_det = torch.log(det)

        return y, log_det


class AffineConstantFlow(nn.Module):
    def __init__(self, dim, **kwargs):
        super(AffineConstantFlow, self).__init__()

        layers = kwargs.get('layers', 5)
        self.layers = nn.ModuleList([AffineConstantLayer(dim)
                                     for _ in range(layers)])

    def forward(self, x):
        cum_log_det = torch.zeros(x.shape[0])
        for layer in self.layers:
            x, log_det = layer(x)
            cum_log_det += log_det

        return x, cum_log_det


class PlanarFlow(nn.Module):
    def __init__(self, dim, **kwargs):
        super(PlanarFlow, self).__init__()

        layers = kwargs.get('layers', 5)
        self.layers = nn.ModuleList([PlanarLayer(dim) for _ in range(layers)])

    def forward(self, x):
        cum_log_det = torch.zeros(x.shape[0])
        for layer in self.layers:
            x, log_det = layer(x)
            cum_log_det += log_det

        return x, cum_log_det


class RadialFlow(nn.Module):
    def __init__(self, dim, **kwargs):
        super(RadialFlow, self).__init__()

        layers = kwargs.get('layers', 5)
        self.layers = nn.ModuleList([RadialLayer(dim) for _ in range(layers)])

    def forward(self, x):
        cum_log_det = torch.zeros(x.shape[0])
        for layer in self.layers:
            x, log_det = layer(x)
            cum_log_det += log_det

        return x, cum_log_det
