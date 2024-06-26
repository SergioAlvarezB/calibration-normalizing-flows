import ternary
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
from scipy.special import softmax
from ternary.helpers import normalize, simplex_iterator

from .ops import project_sequence, project_point, onehot_encode
from .metrics import empirical_cross_entropy

# Regions ListedColormap
sequ = np.linspace(0.33, 1, 1024)
new_colors = np.zeros(((1024)*3, 3))
for i in range(3):
    new_colors[(1024)*i:(1024)*(i+1), i] = sequ

REGS_CMAP = ListedColormap(new_colors)


# ########## Simplex plots ##########


def plot_prob_simplex(probs,
                      target=None,
                      ax=None,
                      scale=50,
                      title='Probabilities',
                      temp=None,
                      fontsize=12,
                      labels=[0, 1, 2]):
    """Makes scatter plot on the 3-simplex. ´probs´ is a (n, 3) np.array
    where each row is expected to sum to 1.
    """

    # Apply temp scaling if passed
    if temp is not None:
        logits = np.log(probs)
        probs = softmax(logits/temp, axis=1)

    if ax is None:
        fig, ax = plt.subplots()

    ax.axis('off')
    fig, tax = ternary.figure(ax=ax)
    if target is None:
        tax.scatter(probs)
    else:
        if target.ndim == 2:
            target = np.argmax(target, axis=1)
        colors = ['red', 'green', 'blue']
        for i, label in enumerate(labels):
            points = probs[target == i, :]
            if len(points) >= 1:
                tax.scatter(
                        points,
                        marker='.',
                        color=colors[i],
                        alpha=0.5,
                        edgecolors=colors[i],
                        label=label)

        tax.legend()
    tax.boundary(linewidth=2.0)
    tax.set_title(title+'\n\n', fontsize=fontsize+2)
    tax.right_corner_label("P($\\theta$={})".format(labels[0]),
                           fontsize=fontsize)
    tax.top_corner_label("P($\\theta$={})".format(labels[1]),
                         fontsize=fontsize)
    tax.left_corner_label("P($\\theta$={})".format(labels[2]),
                          fontsize=fontsize)

    return tax


def plot_pdf_simplex(probs,
                     target=None,
                     ax=None,
                     bw=None,
                     scale=50,
                     title='Estimated PDF',
                     temp=None,
                     fontsize=12,
                     labels=[0, 1, 2]):
    """Makes heatmap ternary plot of the estimated probability density.
    Rows of `probs` are expected to sum up to 1."""

    # Create plot
    if ax is None:
        fig, ax = plt.subplots()

    ax.axis('off')
    figure, tax = ternary.figure(ax=ax, scale=scale)

    # Apply temp scaling if passed
    if temp is not None:
        logits = np.log(probs)
        probs = softmax(logits/temp, axis=1)

    cart_probs = project_sequence(probs)

    if target is not None:
        if target.ndim == 2:
            target = np.argmax(target, axis=1)
        kernels = [gaussian_kde(cart_probs[target == i, :].T, bw_method=bw)
                   for i in range(3)]

        def estimated_pdf(p):
            p = project_point(np.array(p))
            pred = softmax(np.array([kernels[i](p)[0] for i in range(3)]))
            t = np.argmax(pred)
            return (pred[t] + t)/3.

        tax.heatmapf(
                estimated_pdf,
                boundary=True,
                style="hexagonal",
                cmap=REGS_CMAP,
                colorbar=False)

        # Create legend
        tax.legend(handles=[
                mpatches.Patch(color='red', label=labels[0]),
                mpatches.Patch(color='green', label=labels[1]),
                mpatches.Patch(color='blue', label=labels[2])
                ])

    else:
        kernel = gaussian_kde(cart_probs.T, bw_method=bw)

        def estimated_pdf(p):
            p = project_point(np.array(p))
            return kernel(p)[0]

        tax.heatmapf(
                estimated_pdf,
                boundary=True,
                style="hexagonal",
                cmap=plt.get_cmap('gnuplot'),
                colorbar=False)

    tax.boundary(linewidth=2.0)
    tax.set_title(title+'\n\n', fontsize=fontsize+2)
    tax.right_corner_label("P($\\theta$={})".format(labels[0]),
                           fontsize=fontsize)
    tax.top_corner_label("P($\\theta$={})".format(labels[1]),
                         fontsize=fontsize)
    tax.left_corner_label("P($\\theta$={})".format(labels[2]),
                          fontsize=fontsize)

    return tax


def plot_cal_regions_ternary(calibrator,
                             ax=None,
                             scale=50,
                             title='Calibrated output',
                             fontsize=12,
                             labels=[0, 1, 2]):
    # Create plot
    if ax is None:
        fig, ax = plt.subplots()

    ax.axis('off')
    figure, tax = ternary.figure(ax=ax, scale=scale)

    dataix = {}
    probs = []
    ix = 0
    for i, j, k in simplex_iterator(scale=scale):
        probs.append(normalize([i, j, k]))
        dataix[(i, j, k)] = ix
        ix += 1

    probs = np.array(probs)
    logits = np.log(probs + 1e-7)

    # Equivalent to log(softmax())
    logits -= np.log(np.sum(np.exp(logits), axis=1, keepdims=True))

    pred = calibrator(logits)
    t = np.argmax(pred, axis=1)

    colors = (np.max(pred, axis=1)-0.0001 + t)/3.

    data = {}
    for i, j, k in simplex_iterator(scale=scale):
        data[(i, j)] = colors[dataix[(i, j, k)]]

    tax.heatmap(
            data,
            scale,
            style="hexagonal",
            cmap=REGS_CMAP,
            colorbar=False,
            vmin=0.,
            vmax=1.)

    # Create legend
    tax.legend(handles=[
            mpatches.Patch(color='red', label=labels[0]),
            mpatches.Patch(color='green', label=labels[1]),
            mpatches.Patch(color='blue', label=labels[2])
            ])

    tax.boundary(linewidth=2.0)
    tax.set_title(title+'\n\n', fontsize=fontsize+2)
    tax.right_corner_label("P($\\theta$={})".format(labels[0]),
                           fontsize=fontsize)
    tax.top_corner_label("P($\\theta$={})".format(labels[1]),
                         fontsize=fontsize)
    tax.left_corner_label("P($\\theta$={})".format(labels[2]),
                          fontsize=fontsize)

    return tax

# ########## Other plots ##########


def plot_nll_curve(logits,
                   target,
                   calibrators,
                   ax=None,
                   labels=None,
                   log=True,
                   norm=False,
                   top=None,
                   test_logits=None,
                   test_target=None,
                   title='NLL Curve'):

    test = (test_logits is not None)

    if not isinstance(calibrators, list):
        calibrators = [calibrators]

    if labels is None:
        labels = ['Cal_{}'.format(i+1) for i in range(len(calibrators))]

    if ax is None:
        fig, ax = plt.subplots()

    for cal, label in zip(calibrators, labels):
        indv_nll = -np.sum(target*np.log(cal(logits)+1e-7), axis=1)
        tot_nll = np.sum(indv_nll)
        indv_nll = np.sort(indv_nll)[::-1]
        if top is not None:
            indv_nll = indv_nll[:top]

        if norm:
            indv_nll /= tot_nll

        ax.plot(indv_nll, label=label)

        if test:
            indv_nll = -np.sum(test_target*np.log(cal(test_logits)+1e-7),
                               axis=1)
            tot_nll = np.sum(indv_nll)
            indv_nll = np.sort(indv_nll)[::-1]
            if top is not None:
                indv_nll = indv_nll[:top]

            if norm:
                indv_nll /= tot_nll

            ax.plot(indv_nll, label=label+'_test')

    if log:
        ax.set_yscale('log')
    ax.set_title(title)
    if norm:
        ax.set_ylabel('Normalized NLL')
    else:
        ax.set_ylabel('NLL')
    ax.set_xlabel('Sample')
    if len(calibrators) > 1 or test:
        ax.legend()

    return ax


def reliability_diagram(probs,
                        target,
                        ax=None,
                        bins=15,
                        fontsize=12,
                        label='Output',
                        optimum=True,
                        title='Reliability Diagram'):
    """Plots reliability diagram. See:
    Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger.
    On Calibration of Modern Nerual Networks. arXiv preprint
    arXiv:1706.04599, 2017.
    """

    if probs.ndim > 1:
        preds = np.argmax(probs, axis=1)
        if target.shape == probs.shape:
            target = np.argmax(target, axis=1)
        probs = probs[np.arange(probs.shape[0]), preds]
    else:
        preds = np.around(probs)
        probs = np.abs((1-preds) - probs)

    accs = np.equal(preds, target, dtype=np.int32)

    # Generate intervals
    limits = np.linspace(0, 1, num=bins+1)
    width = 1./bins

    # Compute empiric probabilities
    empiric_probs = np.zeros(bins)
    ref_probs = np.zeros(bins)
    confs = np.zeros(bins)
    for i in range(bins):
        low, high = limits[i:i+2]
        ref_probs[i] = (low+high)/2.

        idx = np.where((low < probs) & (probs <= high))
        curr_accs = accs[idx]
        curr_probs = probs[idx]

        if curr_accs.size > 0:
            empiric_probs[i] = np.mean(curr_accs)
            confs[i] = np.mean(curr_probs)

    # Build plot
    if ax is None:
        fig, ax = plt.subplots()

    ax.bar(ref_probs,
           empiric_probs,
           width,
           label=label,
           color='b',
           edgecolor='#000099')

    if optimum:
        stacked = np.stack((confs, empiric_probs))
        bottom = np.min(stacked, axis=0)
        height = np.max(stacked, axis=0) - bottom
        ax.plot(limits, limits, '--k')
        ax.bar(ref_probs,
               height,
               width,
               bottom=bottom,
               label='Gap',
               alpha=0.3,
               color='r',
               edgecolor='r',
               hatch='/')

    ax.legend(loc='upper left')
    ax.set_title(title, fontsize=fontsize+2)
    ax.set_xlabel('Predicted probability', fontsize=fontsize)
    ax.set_ylabel('Empiric probability', fontsize=fontsize)
    ax.set_xlim(0, 1)

    return ax


def reliability_plot(probs,
                     target,
                     ax=None,
                     bins=20,
                     fontsize=12,
                     labels=None,
                     optimum=True,
                     title='Reliability Plot'):
    """Extension of reliability diagrams to multiple classifiers.
    Line plots are used instead of bars and the center probability of each bin
    is used instead of average confidence.
    """

    if not isinstance(probs, list):
        probs = [probs]

    if probs[0].ndim > 1:
        preds = [np.argmax(prob, axis=1) for prob in probs]
        if target.shape == probs[0].shape:
            target = np.argmax(target, axis=1)
        probs = [prob[np.arange(prob.shape[0]), pred]
                 for prob, pred in zip(probs, preds)]

    else:
        preds = [np.around(prob) for prob in probs]
        probs = [np.abs((1-pred) - prob)
                 for prob, pred in zip(probs, preds)]

    accs = [np.equal(pred, target) for pred in preds]

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
            curr_accs = accs[j][idx]
            if curr_accs.size > 0:
                empiric_probs[j, i] = np.mean(curr_accs)

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
    ax.set_title(title+'\n\n', fontsize=fontsize+2)
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
    """Makes ECE plot. See:
    Daniel Ramos, Javier Franco-Pedroso, Alicia Lozano-Diez
    and Joaquin Gonzalez-Rodriguez. Deconstructing Cross-Entropy
    for Probabilistic Binary Classiﬁers. Entropy 2018, 20, 208.
    """

    if like_ratios.ndim == 2:

        if target.shape != like_ratios.shape:
            target = onehot_encode(target)

        # Flatten ratios to evaluate them altogether
        like_ratios = like_ratios.ravel()
        if cal_ratios is not None:
            cal_ratios = cal_ratios.ravel()
        target = target.ravel()

    neutral_ratio = 1

    logprior_axis = np.linspace(range[0], range[1], num=bins)
    base_ECE = np.zeros(logprior_axis.shape)
    if ref:
        ref_ECE = np.zeros(logprior_axis.shape)
        ref_LR = np.ones(like_ratios.shape) * neutral_ratio
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

    ax.plot([np.log10(neutral_ratio), np.log10(neutral_ratio)], [0, 1], '--k')

    labels = ['LR values']

    if ref:
        labels += ['LR={:.3f}'.format(neutral_ratio)]

    if cal_ratios is not None:
        labels += ['After Calibration']

    ax.set_title(title, fontsize=fontsize+2)
    ax.set_xlabel('Prior log$_{10}$(odds)', fontsize=fontsize)
    ax.set_ylabel('Empirical cross-entropy', fontsize=fontsize)

    ax.set_ylim(0, 1)
    ax.set_xlim(range[0], range[1])

    ax.legend(labels, loc='upper right')

    return ax


def ECE_plot_multi(like_ratios,
                   target,
                   ax=None,
                   bins=100,
                   fontsize=12,
                   labels=None,
                   title='ECE plot',
                   lims=[-2.5, 2.5],
                   ref=True):
    """Makes ECE plot. See:
    Daniel Ramos, Javier Franco-Pedroso, Alicia Lozano-Diez
    and Joaquin Gonzalez-Rodriguez. Deconstructing Cross-Entropy
    for Probabilistic Binary Classiﬁers. Entropy 2018, 20, 208.
    """

    neutral_ratio = 1

    logprior_axis = np.linspace(lims[0], lims[1], num=bins)
    comp_ECE = np.zeros((logprior_axis.size, len(like_ratios)))
    if ref:
        ref_ECE = np.zeros(logprior_axis.shape)
        ref_LR = np.ones(like_ratios[0].shape) * neutral_ratio

    # Compute ECE values
    for i, logprior in enumerate(logprior_axis):
        odds = 10**(logprior)
        prior = odds/(1. + odds)
        for j, like_ratio in enumerate(like_ratios):
            comp_ECE[i, j] = empirical_cross_entropy(like_ratio, target, prior)

        if ref:
            ref_ECE[i] = empirical_cross_entropy(ref_LR, target, prior)

    # Build plot
    if ax is None:
        fig, ax = plt.subplots()

    if labels is None:
        labels = ['LR + {:d}'.format(i) for i in range(len(like_ratios))]

    for i in range(len(like_ratios)):
        ax.plot(logprior_axis, comp_ECE[:, i], label=labels[i])

    if ref:
        ax.plot(logprior_axis, ref_ECE, ':k',
                label='LR={:.1f}'.format(neutral_ratio))

    ax.plot([np.log10(neutral_ratio), np.log10(neutral_ratio)], [0, 1], '--k')

    ax.set_title(title, fontsize=fontsize+2)
    ax.set_xlabel('Prior log$_{10}$(odds)', fontsize=fontsize)
    ax.set_ylabel('Empirical cross-entropy', fontsize=fontsize)

    ax.set_ylim(0, min(1., comp_ECE.max()*1.3))
    ax.set_xlim(lims[0], lims[1])

    ax.legend(loc='upper right')

    return ax
