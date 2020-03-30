import os
import time

import numpy as np
import torch
from torch import nn

from flows.flows import Flow, NvpCouplingLayer
from flows.utils import MLP, TempScaler
from utils.data import load_toy_dataset, parse_conf
from gen_graphs import main as plots


conf = parse_conf()
dev = conf.dev
save_dir = conf.save_dir

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
if conf.optim == 'adam':
    opt = torch.optim.Adam(model.parameters(),
                           lr=conf.lr,
                           weight_decay=conf.weight_decay)
else:
    opt = torch.optim.SGD(model.parameters(),
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
        'det': conf.det,
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
