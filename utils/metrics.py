import numpy as np

from .ops import onehot_encode


def neg_log_likelihood(probs, target):
    """Computes the cross_entropy between pred and target."""
    # TODO support binary classification
    if target.shape != probs.shape:
        target = onehot_encode(target)

    return np.mean(-np.sum(target*np.log(probs+1e-7), axis=1))


def empirical_cross_entropy(like_ratios, target, prior):
    prior_odds = prior/(1. - prior)
    post_ratios = like_ratios*prior_odds
    emp_prior = np.sum(target)

    ECE = prior/emp_prior * np.sum(target * np.log2(1 + 1./post_ratios))\
        + ((1 - prior)/(target.size - emp_prior)
           * np.sum((1 - target)*np.log2(1 + post_ratios)))

    return ECE


def expected_calibration_error(probs, target, bins=20):

    if probs.ndim > 1 and target.shape == probs.shape:
        target = np.argmax(target, axis=1)

    probs = probs[np.arange(probs.shape[0]), target]
    preds = np.around(probs)

    # Compute expected calibration error
    width = 1./bins
    EcalE = 0
    empiric_probs = np.zeros(bins)
    for i in range(bins):
        low, high = i*width, (i+1)*width

        idx = np.where((low < probs) & (probs <= high))
        curr_probs = probs[idx]
        curr_preds = preds[idx]

        if curr_probs.size > 0:
            acc = np.mean(curr_preds)
            conf = np.mean(curr_probs)
            EcalE += abs(acc - conf)*curr_probs.size

    EcalE /= target.size

    return EcalE


def accuracy(probs, target):
    if target.shape != probs.shape:
        target = onehot_encode(target)

    return np.mean(np.argmax(probs, axis=1) == np.argmax(target, axis=1))
