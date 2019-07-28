import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize


def project_point(point):
    """Projects a n-dimensional point in a triplex on to the n-1 dimensional
    space so coordinates became independent.
    """

    return point[1:] - point[0]


def project_sequence(points):
    """Applies the transformation `project_point` to each row of `points`.
    """

    return np.apply_along_axis(project_point, 1, points)


def project_point_onsimplex(point):
    """Projects a n-dimensional point on to the n-standar triplex,
    and returns its coordinates on the n+1 dimensional space.
    """
    n = point.size
    x0 = (1 - np.sum(point)) / (n + 1)
    new_point = np.zeros(n+1)
    new_point[0] = x0
    new_point[1:] = point + x0

    return new_point


def project_sequence_onsimplex(points):
    """Applies the transformation `project_point_ontriplex`
    to each row of `points`.
    """

    return np.apply_along_axis(project_point_onsimplex, 1, points)


def onehot_encode(target):
    """One-hot encodes the label vector `target`."""

    n_samples, n_labels = len(target), np.max(target)+1

    one_hot = np.zeros((n_samples, n_labels))
    one_hot[np.arange(n_samples), target] = 1.

    return one_hot


def detection_log_likelihood_ratios(logits, priors):
    log_LR = np.zeros(logits.shape)

    for cls in range(log_LR.shape[1]):
        for i in range(log_LR.shape[1]):
            if i == cls:
                continue
            log_LR[:, cls] += priors[i]/(1 - priors[cls]) \
                * np.exp(logits[:, i] - logits[:, cls])

    log_LR = -np.log(log_LR)

    return log_LR


def optim_temperature(logits,
                      target,
                      method='newton',
                      min_diff=1e-6,
                      step=0.01):
    """Optimizes neg log-likelihood w.r.t. `T` for the
    temperature scaling calibration method.
    """
    T = 1.0
    if target.shape != logits.shape:
        target = onehot_encode(target)

    def jacobian(x, logits, target):
        probs = softmax(logits/x, axis=1)
        grad = -np.mean(np.sum((probs-target)*logits/x**2, axis=1))
        return grad

    if method.lower() == 'newton':

        def target_func(x, logits, target):
            probs = softmax(logits/x, axis=1)
            return np.mean(-np.sum(target*np.log(probs+1e-7), axis=1))

        optim_T = minimize(
                target_func,
                x0=T,
                args=(logits, target),
                method='Newton-CG',
                jac=jacobian)
        T = optim_T.x[0]

    else:  # SGD
        probs = softmax(logits/T, axis=1)
        while True:
            # Compute gradient w.r.t. T
            grad = -np.mean(np.sum((probs-target)*logits/T**2, axis=1))

            # Update step
            probs_new = softmax(logits/(T-grad*step), axis=1)

            if abs(grad) < min_diff:
                break

            T -= grad*step
            probs = probs_new

    return T
