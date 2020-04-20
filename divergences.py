import math
import torch
import numpy as np


def gauss_KLD(qmu, qlog_var, pmu, plog_var):
    return 0.5 * torch.sum(torch.exp(qlog_var - plog_var)
                           + (qmu - pmu)**2/torch.exp(plog_var) - 1
                           + (plog_var - qlog_var))


def gauss_revKLD(qmu, qlog_var, pmu, plog_var):
    return gauss_KLD(pmu, plog_var, qmu, qlog_var)


def gauss_jeffrey(qmu, qlog_var, pmu, plog_var):
    Dj = gauss_KLD(qmu, qlog_var, pmu, plog_var) \
        + gauss_revKLD(qmu, qlog_var, pmu, plog_var)
    return Dj


def gauss_fisher(qmu, qlog_var, pmu, plog_var):
    qvar, pvar = torch.exp(qlog_var), torch.exp(plog_var)

    C1 = (qmu/qvar - pmu/pvar)
    C2 = (1/qvar - 1/pvar)

    Df = torch.sum(C1**2 + 2*C1*C2*qmu + C2**2 * (qvar + qmu**2))

    return Df

# Family of alpha-beta-gamma divergences


def log_Z_gauss(mu, var):
    return 0.5*torch.sum(torch.log(var) + mu**2 / var)


def B_gauss_beta(beta, log_var):
    B = torch.exp(0.5*((1-beta)*log_var - np.log(beta)))
    return B


def C_gauss(qmu, qlog_var, pmu, plog_var, d1, d2):
    qvar, pvar = torch.exp(qlog_var), torch.exp(plog_var)

    new_var = 1/(d1/qvar + d2/pvar)
    new_mu = new_var*(d1*qmu/qvar + d2*pmu/pvar)

    num = torch.log(new_var) + new_mu**2/new_var
    den = d1*(qlog_var + (qmu**2)/qvar) + d2*(plog_var + (pmu**2)/pvar)

    return torch.exp(0.5*(num - den))


def gauss_a(alpha, qmu, qlog_var, pmu, plog_var):
    if alpha == 0:
        # KLD(pi|q)
        return gauss_KLD(pmu, plog_var, qmu, qlog_var)
    elif alpha == 1:
        # KLD(q|pi)
        return gauss_KLD(qmu, qlog_var, pmu, plog_var)

    Da = 1 - C_gauss(qmu, qlog_var, pmu, plog_var, alpha, 1-alpha)

    return 1/(alpha*(1-alpha)) * torch.sum(Da)


def gauss_ar(alpha, qmu, qlog_var, pmu, plog_var):
    if alpha == 0:
        # KLD(pi|q)
        return gauss_KLD(pmu, plog_var, qmu, qlog_var)
    elif alpha == 1:
        # KLD(q|pi)
        return gauss_KLD(qmu, qlog_var, pmu, plog_var)

    qvar, pvar = torch.exp(qlog_var), torch.exp(plog_var)

    log_Z_prior = log_Z_gauss(pmu, pvar)
    log_Z_q = log_Z_gauss(qmu, qvar)

    new_var = 1/(alpha/qvar + (1 - alpha)/pvar)
    new_mu = new_var*(alpha*qmu/qvar + (1-alpha)*pmu/pvar)

    log_Z_new = log_Z_gauss(new_mu, new_var)

    Dar = (log_Z_new - alpha*log_Z_q - (1-alpha)*log_Z_prior)

    return 1/(alpha*(alpha-1)) * Dar


def gauss_beta(beta, qmu, qlog_var, pmu, plog_var):
    if beta == 0:
        # KLD(pi|q)
        return gauss_KLD(pmu, plog_var, qmu, qlog_var)

    elif beta == 1:
        # KLD(q|pi)
        return gauss_KLD(qmu, qlog_var, pmu, plog_var)

    Db = torch.sum(B_gauss_beta(beta, qlog_var)/(beta*(beta-1))
                   + B_gauss_beta(beta, plog_var)/beta
                   - C_gauss(qmu, qlog_var, pmu, plog_var, 1, beta-1)/(beta-1))

    return (2*math.pi)**(-.5*(beta - 1)) * Db


def gauss_gamma(gamma, qmu, qlog_var, pmu, plog_var):
    if gamma == 0:
        # KLD(pi|q)
        return gauss_KLD(pmu, plog_var, qmu, qlog_var)
    elif gamma == 1:
        # KLD(q|pi)
        return gauss_KLD(qmu, qlog_var, pmu, plog_var)

    qvar, pvar = torch.exp(qlog_var), torch.exp(plog_var)
    new_var = 1/(1/qvar + (gamma - 1)/pvar)
    new_mu = new_var*(qmu/qvar + (gamma - 1)*pmu/pvar)

    log_Z_prior = log_Z_gauss(pmu, pvar)
    log_Z_q = log_Z_gauss(qmu, qvar)

    log_Z_gprior = log_Z_gauss(pmu, pvar/gamma)
    log_Z_gq = log_Z_gauss(qmu, qvar/gamma)

    log_Z_new = log_Z_gauss(new_mu, new_var)

    Ef = -0.5*(gamma - 1)*np.log(2*math.pi)

    Dg = 1/(gamma*(gamma-1)) * (log_Z_gq - gamma*log_Z_q + Ef) \
        + 1/gamma * (log_Z_gprior - gamma*log_Z_prior + Ef) \
        - 1/(gamma - 1) * (log_Z_new - log_Z_q - (gamma-1)*log_Z_prior + Ef)

    return Dg
