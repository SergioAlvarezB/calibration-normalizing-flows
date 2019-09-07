import numpy as np

from .ops import onehot_encode


def neg_log_likelihood(probs, target):
    """Computes the cross_entropy between pred and target."""

    if probs.ndim < 2 or probs.shape[1] == 1:
        probs = np.stack(1. - probs.ravel(), probs.ravel())

    if target.shape != probs.shape:
        target = onehot_encode(target)

    return np.mean(-np.sum(target*np.log(probs+1e-7), axis=1))


def empirical_cross_entropy(like_ratios, target, prior):
    """Computes empirical cross entropy. See:
    Daniel Ramos, Javier Franco-Pedroso, Alicia Lozano-Diez
    and Joaquin Gonzalez-Rodriguez. Deconstructing Cross-Entropy
    for Probabilistic Binary Classiﬁers. Entropy 2018, 20, 208.
    """
    prior_odds = prior/(1. - prior)
    post_ratios = like_ratios*prior_odds
    emp_prior = np.sum(target)

    ECE = prior/emp_prior * np.sum(target * np.log2(1 + 1./post_ratios))\
        + ((1 - prior)/(target.size - emp_prior)
           * np.sum((1 - target)*np.log2(1 + post_ratios)))

    return ECE


def expected_calibration_error(probs, target, bins=15):
    """Computes Expected Calibration Error (ECE) as defined in:
    Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger.
    On Calibration of Modern Nerual Networks. arXiv preprint
    arXiv:1706.04599, 2017.

    Implemetation details taken from authors oficial repository:
    https://github.com/gpleiss/temperature_scaling.
    """

    if probs.ndim > 1:
        preds = np.argmax(probs, axis=1)
        if target.shape == probs.shape:
            target = np.argmax(target, axis=1)
        probs = probs[np.arange(probs.shape[0]), preds]
    else:
        preds = np.around(probs)
        probs = np.abs((1-preds) - probs)

    accs = np.equal(preds, target, dtype=np.float32)

    # Compute expected calibration error
    width = 1./bins
    EcalE = 0
    empiric_probs = np.zeros(bins)
    for i in range(bins):
        low, high = i*width, (i+1)*width

        idx = np.where((low < probs) & (probs <= high))
        curr_probs = probs[idx]

        if curr_probs.size > 0:
            acc = np.mean(accs[idx])
            conf = np.mean(curr_probs)
            EcalE += abs(acc - conf)*curr_probs.size

    EcalE /= target.size

    return EcalE


def accuracy(probs, target):
    if target.shape != probs.shape:
        target = onehot_encode(target)

    return np.mean(np.argmax(probs, axis=1) == np.argmax(target, axis=1))
