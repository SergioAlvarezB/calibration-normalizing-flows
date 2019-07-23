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


def accuracy(probs, target):
    if target.shape != probs.shape:
        target = onehot_encode(target)

    return np.mean(np.argmax(probs, axis=1) == np.argmax(target, axis=1))
