import os
import sys
import time
import argparse

import numpy as np
import torch
from torch import nn

from flows.flows import Flow, NvpCouplingLayer
from flows.utils import MLP, TempScaler
from utils.data import load_toy_dataset, str2bool
from gen_graphs import main as plots


MODELS = [
    'flow',
    'dnn',
    'tscal',
]

DATASETS = [
    'bayes',
    'twisted',
    'non',
]

SAVE_PATH = r'C:\Users\sergi\Google Drive\calibration-ml\experiments'


parser = argparse.ArgumentParser()
# Experiment meta-conf
parser.add_argument('--model', help='model used to calibrate',
                    choices=MODELS, type=str.lower)
parser.add_argument('--dataset', help='dataset to calibrate',
                    choices=DATASETS, type=str.lower)
parser.add_argument('--save', help='wether to save final model',
                    type=str2bool, default=True)
parser.add_argument('--hist', help='wether to save training_history',
                    type=str2bool, default=True)
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('-n', '--name', type=str.lower, default='experiment',
                    help='name to save the model')
parser.add_argument('-p', '--plots', help='wether to genertate plots',
                    type=str2bool, default=True)

# General training hyperparameters
parser.add_argument("--lr", help='learning rate', type=float, default=1e-4)
parser.add_argument("-e", "--epochs", help='epochs to train',
                    type=int, default=30000)
parser.add_argument('--weight_decay', help='L2 regularization factor',
                    type=float, default=0)
parser.add_argument("--cuda", help="Whether to use gpu", type=str2bool,
                    nargs='?', const=True, default=True)
parser.add_argument('--step', help='step frequency with which to print info',
                    type=int, default=10)

# Model-specific
parser.add_argument("-k", "--steps", help='Number of flow steps',
                    type=int, default=10)
parser.add_argument("-t", "--shift", help="Whether to use translation in flow",
                    type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-s", "--scale", help="Whether to use scaling in flow",
                    type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-d", "--det", help="Whether to use det in cost function",
                    type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--hidden_size", help='hidden layers size',
                    default=[5, 5], nargs='+', type=int)

conf = parser.parse_args()
dev = torch.device('cuda:0') if conf.cuda else torch.device('cpu')

# Build exp name
if conf.name == 'experiment':
    conf.name += '_' + conf.model + '_'
    conf.name += conf.dataset + '_'
    conf.name += 'lr{:.0e}_'.format(conf.lr)
    conf.name += 'e{:d}_'.format(conf.epochs)
    conf.name += 'wd{:.0e}_'.format(conf.weight_decay)
    if conf.model == 'flow':
        conf.name += 'k{:d}_'.format(conf.steps)
        if not conf.det:
            conf.name += 'nodet_'
        if not conf.shift:
            conf.name += 'noshift_'
        if not conf.scale:
            conf.name += 'noscale_'
    if conf.model in ['flow', 'dnn']:
        conf.name += \
            '[' + '-'.join(['{:d}'.format(h) for h in conf.hidden_size]) + ']'


save_dir = os.path.join(SAVE_PATH, conf.name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    answer = str2bool(input("name already exists, overwrite? [y/N]? ").lower())
    if not answer:
        print('Aborting experiment')
        sys.exit()


# Load data
logits, target = load_toy_dataset('data/toys/', conf.dataset)

n_samples, dim = logits.shape

torch_logits = torch.as_tensor(logits, dtype=torch.float).to(dev)
torch_target = torch.as_tensor(target, dtype=torch.long).to(dev)

# Load model
if conf.model == 'flow':
    model = Flow([NvpCouplingLayer(dim,
                                   hidden_size=conf.hidden_size,
                                   scale=conf.scale,
                                   shift=conf.shift)
                  for _ in range(conf.steps)]).to(dev)
elif conf.model == 'dnn':
    model = MLP(dim, hidden_size=conf.hidden_size).to(dev)

else:
    model = TempScaler().to(dev)


# optimizer
opt = torch.optim.Adam(model.parameters(),
                       lr=conf.lr,
                       weight_decay=conf.weight_decay)
CE = nn.CrossEntropyLoss()


# Initialize training history

h = {
    'dataset': conf.dataset,
    'model': conf.model,
    'lr': conf.lr,
    'weight_decay': conf.weight_decay,
    'loss': [],
    'intermediate_results': [],
}

if conf.model == 'flow':
    h['log_det'] = []
    h['model_dict'] = {
        'steps': conf.steps,
        'scale': conf.scale,
        'shift': conf.shift,
        'hidden_size': conf.hidden_size
    }
if conf.model == 'dnn':
    h['model_dict'] = {
        'hidden_size': conf.hidden_size
    }

t0 = time.time()
for e in range(conf.epochs):

    # Compute Predictions
    if conf.model == 'flow':
        zs, _logdet = model(torch_logits)
        _logdet = torch.mean(_logdet)
        preds = zs[-1]
        _loss = CE(preds, torch_target) - (conf.det)*_logdet
        h['log_det'].append(_logdet.item())

    else:
        preds = model(torch_logits)
        _loss = CE(preds, torch_target)

    h['loss'].append(_loss.item())
    if (preds != preds).any():
        print('Aborting training due to nan values')
        break

    opt.zero_grad()
    _loss.backward()
    opt.step()

    if e % conf.step == (conf.step-1):
        h['intermediate_results'].append(preds.detach().cpu().numpy())
        if conf.verbose:
            print("Finished epoch: {:d} at time {:.2f}, loss: {:.3e}".format(
                  e, time.time()-t0, h['loss'][-1]))

if conf.hist:
    np.save(os.path.join(save_dir, 'history'), h)

    if conf.plots:
        plots(save_dir, conf.hist, False)

print("Experiment succesful!, results saved at: '{}'".format(save_dir))
