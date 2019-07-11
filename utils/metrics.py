import numpy as np

from .ops import onehot_encode


def neg_log_likelihood(probs, target):
    """Computes the cross_entropy between pred and target."""
    if target.shape != probs.shape:
        target = onehot_encode(target)

    return np.mean(-np.sum(target*np.log(probs+1e-7), axis=1))


def accuracy(probs, target):
    if target.shape != probs.shape:
        target = onehot_encode(target)

    return np.mean(np.argmax(probs, axis=1) ==  np.argmax(target, axis=1))
