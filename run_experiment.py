import os
import argparse

import numpy as np
import torch

from flows.flows import Flow, NvpCouplingLayer
from flows.utils import MLP, TempScaler
from utils.data import load_toy_dataset


MODELS = [
    'flow',
    'dnn',
    'tscal',
]

DATASETS = [
    'bayes',
    'twistted',
    'non',
]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('model', help='model used to calibrate',
                    choices=MODELS, type=str.lower)
parser.add_argument('dataset', help='dataset to calibrate',
                    choices=DATASETS, type=str.lower)

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

# General training hyperparameters
parser.add_argument("--lr", help='learning rate', type=float, default=1e-4)
parser.add_argument("--e", "-epochs", help='epochs to train',
                    type=int, default=30000)
parser.add_argument('--weight_decay', help='L2 regularization factor',
                    type=float, default=0)
parser.add_argument("--cuda", help="Whether to use gpu", type=str2bool,
                    nargs='?', const=True, default=True)

# Model-specific
parser.add_argument("-k", "--steps", help='Number of flow steps',
                    type=int, default=10)
parser.add_argument("-t", "--shift", help="Whether to use translation in flow",
                    type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-s", "--scale", help="Whether to use scaling in flow",
                    type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-h", "--hidden_size", help='hidden layers size',
                    default=[5, 5], nargs='+', type=int)

conf = parser.parse_args()
dev = torch.device('cuda:0') if conf.cuda else torch.device('cpu')

# Load data
logits, target = load_toy_dataset('data/toys/', conf.dataset)

n_samples, dim = logits.shape

torch_logits = torch.as_tensor(logits, dtype=torch.float).to(dev)
torch_target = torch.as_tensor(target, dtype=torch.float).to(dev)

# Load model
if conf.model == 'flow':
    model = Flow([NvpCouplingLayer(dim,
                                   hidden_size=conf.hidden_size,
                                   scale=conf.scale,
                                   shift=conf.shift)
                  for _ in range(conf.k)]).to(dev)
elif conf.model == 'dnn':
    model = MLP(dim, hidden_size=conf.hidden_size).to(dev)

else:
    model = TempScaler().to(dev)


# optimizer
