import ternary
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .ops import project_sequence, project_point


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
            points = probs[target==i, :]
            if len(points)>=1:
                tax.scatter(points,
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
    tax.heatmapf(estimated_pdf,
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
