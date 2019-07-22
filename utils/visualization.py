import ternary
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .ops import project_sequence, project_point, onehot_encode
from .metrics import empirical_cross_entropy


def plot_prob_triplex(probs,
                      target=None,
                      ax=None,
                      scale=50,
                      title='Probabilities',
                      fontsize=12,
                      labels=[0, 1, 2]):

    if ax is None:
        fig, ax = plt.subplots()

    ax.axis('off')
    fig, tax = ternary.figure(ax=ax)
    if target is None:
        tax.scatter(probs)
    else:
        colors = ['red', 'green', 'blue']
        for i, label in enumerate(labels):
            points = probs[target == i, :]
            if len(points) >= 1:
                tax.scatter(
                        points,
                        color=colors[i],
                        label=label)

        tax.legend()
    tax.boundary(linewidth=2.0)
    tax.set_title(title+'\n\n', fontsize=fontsize)
    tax.right_corner_label("P($\\theta$={})".format(labels[0]),
                           fontsize=fontsize)
    tax.top_corner_label("P($\\theta$={})".format(labels[1]),
                         fontsize=fontsize)
    tax.left_corner_label("P($\\theta$={})".format(labels[2]),
                          fontsize=fontsize)

    return tax


def plot_pdf_triplex(probs,
                     ax=None,
                     scale=50,
                     title='Estimated PDF',
                     fontsize=12,
                     labels=[0, 1, 2]):
    """Makes heatmap ternary plot of the estimated probability density."""

    cart_probs = project_sequence(probs)
    kernel = stats.gaussian_kde(cart_probs.T)

    def estimated_pdf(p):
        p = project_point(np.array(p))
        return kernel(p)[0]

    if ax is None:
        fig, ax = plt.subplots()

    ax.axis('off')
    figure, tax = ternary.figure(ax=ax, scale=scale)
    tax.heatmapf(
            estimated_pdf,
            boundary=True,
            style="hexagonal",
            cmap=plt.get_cmap('gnuplot'),
            colorbar=False)

    tax.boundary(linewidth=2.0)
    tax.set_title(title+'\n\n', fontsize=fontsize)
    tax.right_corner_label("P($\\theta$={})".format(labels[0]),
                           fontsize=fontsize)
    tax.top_corner_label("P($\\theta$={})".format(labels[1]),
                         fontsize=fontsize)
    tax.left_corner_label("P($\\theta$={})".format(labels[2]),
                          fontsize=fontsize)

    return tax


def reliability_plot(probs,
                     target,
                     ax=None,
                     bins=20,
                     fontsize=12,
                     labels=None,
                     optimum=True,
                     title='Reliability Plot'):

    if not isinstance(probs, list):
        probs = [probs]

    if target.shape != probs[0].shape:
        target = onehot_encode(target)

    # Evaluate the probability of each sample independently
    probs, target = [prob.ravel() for prob in probs], target.ravel()

    # Generate intervals
    limits = np.linspace(0, 1, num=bins+1)

    # Compute empiric probabilities
    empiric_probs = np.zeros((len(probs), bins))
    ref_probs = np.zeros(bins)
    for i in range(bins):
        low, high = limits[i:i+2]
        ref_probs[i] = (low+high)/2.
        for j in range(len(probs)):
            idx = np.where((low < probs[j]) & (probs[j] <= high))
            curr_targets = target[idx]
            if curr_targets.size > 0:
                empiric_probs[j, i] = np.sum(curr_targets)/curr_targets.size

    # Build plot
    if ax is None:
        fig, ax = plt.subplots()

    markers = ['--bo', '--rx', '--g^', '--ys', '--m*', '--cd']

    if optimum:
        ax.plot(limits, limits, '--k')
        if labels is not None:
            labels = ['Optimum calibration'] + labels

    for j in range(len(probs)):
        ax.plot(ref_probs, empiric_probs[j, :], markers[j % len(markers)])

    if labels is not None:
        ax.legend(labels, loc='upper left')
    ax.set_title(title+'\n\n', fontsize=fontsize)
    ax.set_xlabel('Predicted probability', fontsize=fontsize)
    ax.set_ylabel('Empiric probability', fontsize=fontsize)

    return ax


def ECE_plot(like_ratios,
             target,
             cal_ratios=None,
             ax=None,
             bins=100,
             fontsize=12,
             title='ECE plot',
             range=[-2.5, 2.5],
             ref=True):

    logprior_axis = np.linspace(range[0], range[1], num=bins)
    base_ECE = np.zeros(logprior_axis.shape)
    if ref:
        ref_ECE = np.zeros(logprior_axis.shape)
        ref_LR = np.ones(like_ratios.shape, dtype=np.float32)
    if cal_ratios is not None:
        cal_ECE = np.zeros(logprior_axis.shape)

    # Compute ECE values
    for i, logprior in enumerate(logprior_axis):
        odds = 10**(logprior)
        prior = odds/(1. + odds)
        base_ECE[i] = empirical_cross_entropy(like_ratios, target, prior)

        if ref:
            ref_ECE[i] = empirical_cross_entropy(ref_LR, target, prior)

        if cal_ratios is not None:
            cal_ECE[i] = empirical_cross_entropy(cal_ratios, target, prior)

    # Build plot
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(logprior_axis, base_ECE, 'r')

    if ref:
        ax.plot(logprior_axis, ref_ECE, ':k')

    if cal_ratios is not None:
        ax.plot(logprior_axis, cal_ECE, '--b')

    ax.plot([0, 0], [0, 1], '--k')

    labels = ['LR values']

    if ref:
        labels += ['LR=1']

    if cal_ratios is not None:
        labels += ['After Calibration']

    ax.set_title(title+'\n', fontsize=fontsize)
    ax.set_xlabel('Prior log$_{10}$(odds)', fontsize=fontsize)
    ax.set_ylabel('Empirical cross-entropy', fontsize=fontsize)

    ax.set_ylim(0, 1)

    ax.legend(labels, loc='upper right')

    return ax
