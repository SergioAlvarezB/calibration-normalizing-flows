import os
import argparse

import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from ternary.helpers import project_sequence

from utils.data import str2bool, load_toy_dataset
from utils.visualization import plot_prob_simplex
from utils.metrics import neg_log_likelihood as nll

plt.ioff()


def create_animation_logits(intermediate_results, target):

    preds = intermediate_results[0]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(20, 45)

    sc = ax.scatter(preds[:, 0], preds[:, 1], preds[:, 2], c=target)

    # manually relim:
    xmin, xmax = 1e7, 1e-7
    ymin, ymax = 1e7, 1e-7
    zmin, zmax = 1e7, 1e-7
    for preds in intermediate_results:
        xmin, xmax = min(preds[:, 0].min(), xmin), max(preds[:, 0].max(), xmax)
        ymin, ymax = min(preds[:, 1].min(), ymin), max(preds[:, 1].max(), ymax)
        zmin, zmax = min(preds[:, 2].min(), zmin), max(preds[:, 2].max(), zmax)

    xmin, xmax = max(xmin, -20.), min(xmax, 20.)
    ymin, ymax = max(xmin, -20.), min(ymax, 20.)
    zmin, zmax = max(xmin, -20.), min(zmax, 20.)

    ax.set_xlim(xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin))
    ax.set_ylim(ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin))
    ax.set_zlim(zmin-0.1*(zmax-zmin), zmax+0.1*(zmax-zmin))

    def update_scat(i):
        preds = intermediate_results[i]
        sc._offsets3d = (preds[:, 0], preds[:, 1], preds[:, 2])

    ani = animation.FuncAnimation(fig,
                                  update_scat,
                                  frames=len(intermediate_results),
                                  interval=30)

    return ani


def create_animation_simplex(intermediate_results, target):

    fig, ax = plt.subplots()

    probs = softmax(intermediate_results[0], 1)

    ax = plot_prob_simplex(probs, target, ax=ax, title='Output probabilities')

    def update_scat(i):
        probs = softmax(intermediate_results[i], 1)
        for j, collection in enumerate(ax.ax.collections):
            offsets2d = np.array(project_sequence(probs[target == j])).T
            collection.set_offsets(offsets2d)
        ax._redraw_labels()

    ani = animation.FuncAnimation(fig,
                                  update_scat,
                                  frames=len(intermediate_results),
                                  interval=30)

    return ani


parser = argparse.ArgumentParser()
parser.add_argument("path", help='path to the experiment to visualize')
parser.add_argument('-s', '--save', help='wether to save plots',
                    type=str2bool, default=True)
parser.add_argument('-v', '--show', help='wether to display results',
                    type=str2bool, default=True)
conf = parser.parse_args()

# Load data
h = np.load(os.path.join(conf.path, 'history.npy'), allow_pickle=True)[()]

logits, target = load_toy_dataset('data/toys/', h['dataset'])
uncal_nll = nll(softmax(logits, axis=1), target)

nll = (np.array(h['loss']) + np.array(h['log_det'])) \
    if h['model'] == 'flow' else h['loss']

fig1, ax = plt.subplots(figsize=(12, 8))
ax.plot(nll)
ax.set_title('NNL')
ax.set_ylim(0, min(2*uncal_nll, max(nll)))
ax.axhline(uncal_nll, c='r', label='Uncalibrated ' + h['dataset'])
ax.set_ylabel(r'$-log(P(z))$')
ax.set_xlabel('Epoch')
ax.legend()

if h['model'] == 'flow':
    fig2, ax = plt.subplots(figsize=(12, 8))
    ax.plot(h['log_det'])
    ax.set_title('Determinant')
    ax.set_ylabel(r'$log(|J|)$')
    ax.set_xlabel('Epoch')

ani_simplex = create_animation_simplex(h['intermediate_results'], target)
ani_logits = create_animation_logits(h['intermediate_results'], target)

if conf.save:
    ani_simplex.save(os.path.join(conf.path, 'ani-simplex.mp4'), codec='h264')
    ani_logits.save(os.path.join(conf.path, 'ani-logits.mp4'), codec='h264')
    fig1.savefig(os.path.join(conf.path, 'nll.png'), dpi=300)
    if h['model'] == 'flow':
        fig2.savefig(os.path.join(conf.path, 'log_det.png'), dpi=300)

if conf.show:
    plt.show()
