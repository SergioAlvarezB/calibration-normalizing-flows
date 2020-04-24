import torch
from torch import nn

from divergences import gauss_KLD, gauss_revKLD, gauss_jeffrey, gauss_fisher
from divergences import gauss_a, gauss_ar, gauss_beta, gauss_gamma

from utils.metrics import expected_calibration_error


class Linear_LR(nn.Module):

    def __init__(self, input_dim, output_dim, pmean, plog_var):
        super(Linear_LR, self).__init__()

        # Prior distribution
        self.plog_var = nn.Parameter(torch.Tensor([plog_var]),
                                     requires_grad=False)
        self.pmean = nn.Parameter(torch.Tensor([pmean]),
                                  requires_grad=False)

        # Initialize parameters
        self.w_mean = nn.Parameter(torch.randn(input_dim, output_dim))
        self.w_log_var = nn.Parameter(torch.randn(input_dim, output_dim))

        self.b_mean = nn.Parameter(torch.randn(output_dim,))
        self.b_log_var = nn.Parameter(torch.randn(output_dim,))

    def forward(self, x):

        # Local Reparametrization
        Z_mu = (torch.mm(x, self.w_mean)) + self.b_mean

        Z_sigma = torch.sqrt(torch.mm(x**2, torch.exp(self.w_log_var))
                             + torch.exp(self.b_log_var))

        # Sample
        Z = Z_mu + torch.randn_like(Z_mu)*Z_sigma

        return Z

    def MAP(self, x):
        Z_mu = (torch.mm(x, self.w_mean)) + self.b_mean

        return Z_mu

    def get_total_params(self):

        # Return total number of parameters
        return self.w_mean.numel()*2 + self.b_mean.numel()*2

    def get_collapsed_posterior(self):
        # Check If the parameters has collapsed to the prior
        w = ((self.w_mean <= self.pmean + 0.01)
             & (self.w_mean >= self.pmean - 0.01)
             & (self.w_log_var <= self.plog_var + 0.01)
             & (self.w_log_var >= self.plog_var - 0.01)).float().sum()

        b = ((self.b_mean <= self.pmean + 0.01)
             & (self.b_mean >= self.pmean - 0.01)
             & (self.b_log_var <= self.plog_var + 0.01)
             & (self.b_log_var >= self.plog_var - 0.01)).float().sum()

        return w+b


class BNN_GVILR(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 pmean,
                 plog_var,
                 divergence='KL',
                 d_param=None,
                 hidden_size=[]):
        super(BNN_GVILR, self).__init__()

        # Get general divergence
        self.divergence = self._get_divergence(d_param, divergence)

        # DNN architecture
        hs = [input_dim] + hidden_size + [output_dim]

        # Loss function
        self.CE = torch.nn.functional.cross_entropy

        # Initialize layers
        self.layers = nn.ModuleList([Linear_LR(inp, out, pmean, plog_var)
                                     for inp, out in zip(hs[:-1], hs[1:])])

    def forward(self, x):
        z = x
        for i, layer in enumerate(self.layers):
            if i > 0:
                z = torch.relu(z)
            z = layer(z)

        return z

    def compute_D(self):
        D = 0
        for l in self.layers:
            D += self.divergence(l.w_mean, l.w_log_var, l.pmean, l.plog_var)\
                + self.divergence(l.b_mean, l.b_log_var, l.pmean, l.plog_var)

        return D

    def ELBO(self, x, y, beta=1., n_samples=10, warm_up=False):

        NLL = 0
        for _ in range(n_samples):
            NLL += self.CE(self.forward(x), y, reduction='sum')

        NLL /= n_samples
        D = self.compute_D()
        ELBO = NLL + beta*D*(not warm_up)

        return ELBO, NLL, D

    def predictive(self, x, n_samples):
        assert not self.training, "model must be in eval() mode"
        preds = 0.0
        for i in range(n_samples):
            preds += nn.functional.softmax(self.forward(x), dim=1).detach()

        return preds/n_samples

    def MAP(self, x):
        Z = 0
        z = x
        for i, layer in enumerate(self.layers):
            if i > 0:
                z = torch.relu(z)

            z = layer.MAP(z)
        Z = z

        return Z

    def evaluate(self, X, y, n_samples=1000):
        preds = self.predictive(X, n_samples=n_samples)
        _, _preds = torch.max(preds, dim=1)
        acc = torch.mean((_preds == y).float()).item()
        ece = expected_calibration_error(preds.cpu().numpy(), y.cpu().numpy())

        return acc, ece

    def _get_collapsed_posterior(self):
        # Percentage of collapsed parameters
        with torch.no_grad():
            collaps, total_params = [0.0]*2
            for l in self.layers:
                collaps += l.get_collapsed_posterior()
                total_params += l.get_total_params()

            return 100. * collaps/float(total_params)

    def _get_divergence(self, d_param, divergence):

        divergences = ['KL', 'rKL', 'F', 'J', 'A', 'AR', 'B', 'G']

        assert divergence in divergences, ("divergence not recognized, "
                                           "must be one of: " + divergences)

        if divergence in ['A', 'AR', 'B', 'G']:
            assert d_param is not None, ("You need to provide d_param"
                                         " for this divergence!")

        if divergence == 'KL':
            return gauss_KLD

        if divergence == 'rKL':
            return gauss_revKLD

        if divergence == 'J':
            return gauss_jeffrey

        if divergence == 'F':
            return gauss_fisher

        if divergence == 'A':
            return lambda a, b, c, d: gauss_a(d_param, a, b, c, d)

        if divergence == 'AR':
            return lambda a, b, c, d: gauss_ar(d_param, a, b, c, d)

        if divergence == 'B':
            return lambda a, b, c, d: gauss_beta(d_param, a, b, c, d)

        if divergence == 'G':
            return lambda a, b, c, d: gauss_gamma(d_param, a, b, c, d)
