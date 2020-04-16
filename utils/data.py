import os
import sys
import pickle
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt


DEFAULT_CIFAR3 = ['airplane', 'automobile', 'bird']
SAVE_PATH = r'C:\Users\sergi\Google Drive\calibration-ml\experiments'
CONF = None
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

OPTIMIZERS = [
    'adam',
    'sgd',
]

# Explicitely declare the mapping to use as reference.
ix2label = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'}
label2ix = {v: k for k, v in ix2label.items()}

colors = ['b', 'g', 'r', 'k']
colors_test = ['cyan', 'lightgreen', 'orange', 'gray']


def parse_conf():
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
    parser.add_argument('-n', '--name', type=str.lower, default='exp',
                        help='name to save the model')
    parser.add_argument('-p', '--plots', help='wether to genertate plots',
                        type=str2bool, default=True)

    # General training hyperparameters
    parser.add_argument('--optim', help='which optimizer to use',
                        default='adam', choices=OPTIMIZERS)
    parser.add_argument("--lr", help='learning rate', type=float, default=1e-4)
    parser.add_argument("-e", "--epochs", help='epochs to train',
                        type=int, default=30000)
    parser.add_argument('--weight_decay', help='L2 regularization factor',
                        type=float, default=0)
    parser.add_argument("--cuda", help="Whether to use gpu", type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--step', help='step frequency to print info',
                        type=int, default=10)

    # Model-specific
    parser.add_argument("--inv", help="whether to invert logits",
                        action="store_true")
    parser.add_argument("-k", "--steps", help='Number of flow steps',
                        type=int, default=10)
    parser.add_argument("-t", "--shift", help="Whether to use shift in flow",
                        type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-s", "--scale", help="Whether to use scaling in flow",
                        type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-d", "--det", help="Whether to use det in cost",
                        type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--hidden_size", help='hidden layers size',
                        default=[5, 5], nargs='+', type=int)

    global CONF
    CONF = parser.parse_args()
    CONF.dev = torch.device('cuda:0') if CONF.cuda else torch.device('cpu')

    # Build experiment name
    if CONF.name == 'exp':
        CONF.name += '_' + CONF.model + '_'
        CONF.name += CONF.dataset + '_'
        CONF.name += CONF.optim + '_'
        CONF.name += 'lr{:.0e}_'.format(CONF.lr)
        CONF.name += 'e{:d}_'.format(CONF.epochs)
        CONF.name += 'wd{:.0e}_'.format(CONF.weight_decay)
        if CONF.inv:
            CONF.name += 'inv_'
        if CONF.model == 'flow':
            CONF.name += 'k{:d}_'.format(CONF.steps)
            if not CONF.det:
                CONF.name += 'nodet_'
            if not CONF.shift:
                CONF.name += 'noshift_'
            if not CONF.scale:
                CONF.name += 'noscale_'
        if CONF.model in ['flow', 'dnn']:
            CONF.name += \
                '[' + '-'.join(['{:d}'.format(h)
                                for h in CONF.hidden_size]) + ']'

    CONF.save_dir = os.path.join(SAVE_PATH, CONF.name)
    if not os.path.exists(CONF.save_dir):
        os.makedirs(CONF.save_dir)
    else:
        answer = str2bool(input("name already exists, overwrite? [y/N]? "))
        if not answer:
            print('Aborting experiment')
            sys.exit()

    return CONF


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', ''):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_toy_dataset(data_path, dataset):
    dataset = dataset + '_separable'

    logits = np.load(os.path.join(data_path, dataset + '_logits.npy'))
    target = np.load(os.path.join(data_path, dataset + '_target.npy'))

    return logits, target


def reset_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# JUAN  2D toy dataset
def toy_dataset(N: int) -> list:
    ''' Toy Dataset
        Args:
            N  (int) :->: Number of Samples. If it is not divisible by 4,
            it will be rounded to be
    '''
    reset_seed(1)

    ''' test set '''
    x1 = torch.randn(20, 2)*1 + 0.5
    x2 = torch.randn(20, 2)*0.5 + torch.from_numpy(np.array([0.5, -2])).float()
    x3 = torch.randn(20, 2)*0.3 - 0.5
    x4 = torch.randn(20, 2)*0.8 + torch.from_numpy(np.array([-1., -2])).float()

    t1 = torch.zeros(20,)
    t2 = torch.ones(20,)
    t3 = torch.ones(20,) + 1
    t4 = torch.ones(20,) + 2

    idx = np.random.permutation(80)
    x_test = torch.cat((x1, x2, x3, x4))[idx].float()
    t_test = torch.cat((t1, t2, t3, t4))[idx].long()

    ''' train set '''
    per_class = int(N/4.)

    # sample samples per class
    x1 = torch.randn(per_class, 2)*1.0 + 0.5
    x2 = torch.randn(per_class, 2)*0.5 \
        + torch.from_numpy(np.array([0.5, -2])).float()
    x3 = torch.randn(per_class, 2)*0.3-0.5
    x4 = torch.randn(per_class, 2)*0.8 \
        + torch.from_numpy(np.array([-1., -2])).float()

    t1 = torch.zeros(per_class,)
    t2 = torch.ones(per_class,)
    t3 = torch.ones(per_class,)+1
    t4 = torch.ones(per_class,)+2

    idx = np.random.permutation(per_class*4)
    x = torch.cat((x1, x2, x3, x4))[idx].float()
    t = torch.cat((t1, t2, t3, t4))[idx].long()

    return [x, t], [x_test, t_test]


def plot_toy_dataset(X_tr: list, X_te: list) -> None:

    X, T = X_tr
    X_te, T_te = X_te

    X = X.detach().numpy()
    T = T.detach().numpy()
    X_te = X_te.detach().numpy()
    T_te = T_te.detach().numpy()

    N_labels = np.unique(T)
    assert len(N_labels) == len(np.unique(T_te)), ("Getting different number"
                                                   "of classes")

    for l in N_labels:
        idx_tr = T == l
        idx_te = T_te == l
        plt.plot(X[idx_tr, 0], X[idx_tr, 1], '*' + colors[l])
        plt.plot(X_te[idx_te, 0], X_te[idx_te, 1], 'o', color=colors_test[l])

    plt.show(block=True)


def plot_toy_regions(X_tr, X_te, model, M=300, predictive_samples=1000):
    X_tr, T_tr = X_tr
    X_te, T_te = X_te

    X_tr = X_tr.detach().cpu().numpy()
    T_tr = T_tr.detach().cpu().numpy()
    X_te = X_te.detach().cpu().numpy()
    T_te = T_te.detach().cpu().numpy()

    # Define the grid where we plot
    vx = np.linspace(-5, 4, M)
    vy = np.linspace(-5, 4, M)
    data_feat = np.zeros((M**2, 2), np.float32)

    # this can be done much more efficient for sure
    for x, px in enumerate(vx):
        for y, py in enumerate(vy):
            data_feat[x*M+y] = np.array([px, py])

    # forward through the model
    data_feat = torch.from_numpy(data_feat)
    with torch.no_grad():
        logits = model.predictive(data_feat, predictive_samples)
        max_conf, max_target = torch.max(logits, dim=1)

    conf = np.zeros((M**2), np.float32)
    labl = np.zeros((M**2), np.float32)
    data_feat = 0
    conf[:] = np.nan
    labl[:] = np.nan
    max_conf, max_target = max_conf.detach(), max_target.detach()
    for x, px in enumerate(vx):
        for y, py in enumerate(vy):
            conf[x*M+y] = max_conf[x*M+y]
            labl[x*M+y] = max_target[x*M+y]

    X, Y = np.meshgrid(vx, vy)

    cmap = [
            plt.cm.get_cmap("Reds"),
            plt.cm.get_cmap("Greens"),
            plt.cm.get_cmap("Blues"),
            plt.cm.get_cmap("Greys")
        ]

    color_list_tr = ['*r', '*g', '*b', '*k']
    color_list_te = ['orange', 'lightgreen', 'cyan', 'gray']
    markers = ['d', '*', 'P', 'v']
    fig, ax = plt.subplots(figsize=(18, 12))

    for ctr, cte, i, c, marker in zip(color_list_tr,
                                      color_list_te,
                                      range(4),
                                      cmap,
                                      markers):

        idx_tr = T_tr == i
        idx_te = T_te == i
        xtr = X_tr[idx_tr, :]
        xte = X_te[idx_te, :]

        aux = np.zeros((M**2), np.float32)
        x1, x2 = xtr[:, 0], xtr[:, 1]

        plt.plot(x1, x2, marker, color=cte, markersize=10, alpha=0.5)

        index = np.where(labl == i)[0]
        aux[index] = conf[index]
        index_neg = np.where(labl != i)[0]
        aux[index_neg] = np.nan
        aux = aux.reshape(M, M, order='F')
        dib = ax.contourf(X, Y, aux, cmap=c, alpha=0.5, levels=[
                0,
                0.25,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.92,
                0.94,
                0.96,
                0.98,
                1.0
            ])

        if i == 3:
            plt.xlabel('x1', fontsize=22)
            plt.ylabel('x2', fontsize=22)
            plt.xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], fontsize=18)
            plt.yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], fontsize=18)

    plt.show()


def load_logits(model_dir):
    with open(os.path.join(model_dir, 'train_logits.pkl'), 'rb') as f:
        train_logits = pickle.load(f)

    with open(os.path.join(model_dir, 'test_logits.pkl'), 'rb') as f:
        test_logits = pickle.load(f)

    return train_logits, test_logits


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def reshape(ima_row):
    ima = ima_row.reshape([32, 32, 3], order='F')
    ima = np.transpose(ima, (1, 0, 2))
    return ima


def process_batch(data_batch, target_ixlabels):
    new_data_batch = {
        'images': [],
        'labels': []}

    label_map = {ix: i for i, ix in enumerate(target_ixlabels)}

    for i, true_label in enumerate(data_batch[b'labels']):
        if true_label not in target_ixlabels:
            continue

        new_data_batch['images'].append(reshape(data_batch[b"data"][i]))
        new_data_batch['labels'].append(label_map[true_label])

    return new_data_batch


def match_priors(images, labels, prior):
    new_images, new_labels = [], []
    # Normalize to more frequent label
    prior = np.array(prior)/np.max(prior)

    for label in range(len(prior)):
        idx = np.where(labels == label)[0]
        n = len(idx)
        idx = idx[np.random.permutation(n)[:int(n*prior[label])]]

        new_images += images[idx].tolist()
        new_labels += labels[idx].tolist()

    # Convert to numpy array and shuffle
    perm = np.random.permutation(len(new_images))

    images = np.array(new_images)[perm]
    labels = np.array(new_labels)[perm]

    return images, labels


def get_cifar3(data_path,
               target_labels=DEFAULT_CIFAR3[:],
               test=False,
               prior=None,
               test_prior=None):
    cifar3 = {
        'images': [],
        'labels': []}

    train_batches = ["data_batch_{}".format(i+1) for i in range(5)]
    target_ixlabels = [label2ix[label] for label in target_labels]
    cifar3_ix2label = {i: ix2label[ix] for i, ix in enumerate(target_ixlabels)}

    for batch in train_batches:
        data_batch = unpickle(os.path.join(data_path, batch))
        curr_batch = process_batch(data_batch, target_ixlabels)

        cifar3['images'] += curr_batch['images']
        cifar3['labels'] += curr_batch['labels']

    # Convert to numpy array
    cifar3['images'] = np.array(cifar3['images'])
    cifar3['labels'] = np.array(cifar3['labels'])

    # Adjust dataset to meet the specified priors
    if prior is not None:
        imas, labels = match_priors(cifar3['images'], cifar3['labels'], prior)
        cifar3['images'], cifar3['labels'] = imas, labels

    if test:
        test_batch = unpickle(os.path.join(data_path, 'test_batch'))
        test_batch = process_batch(test_batch, target_ixlabels)

        cifar3['test_images'] = test_batch['images']
        cifar3['test_labels'] = test_batch['labels']

        # Convert to numpy array
        cifar3['test_images'] = np.array(cifar3['test_images'])
        cifar3['test_labels'] = np.array(cifar3['test_labels'])

        # Adjust test set to meet the specified priors
        if test_prior is not None:
            imas, labels = match_priors(
                    cifar3['test_images'],
                    cifar3['test_labels'],
                    test_prior)
            cifar3['test_images'], cifar3['test_labels'] = imas, labels

    return cifar3, cifar3_ix2label


def get_cifarn(data_path,
               n=None,
               target_labels=None,
               test=False,
               prior=None,
               test_prior=None):

    if n is None and target_labels is None:
        raise ValueError("Either `n` or `target_labels` must be specified")

    if n is None:
        assert all([label in ix2label.values() for label in target_labels])
        n = len(target_labels)

    if target_labels is None:
        target_labels = [ix2label[k] for k in range(n)]

    cifarn = {
        'images': [],
        'labels': []}

    train_batches = ["data_batch_{}".format(i+1) for i in range(5)]
    target_ixlabels = [label2ix[label] for label in target_labels]
    cifarn_ix2label = {i: ix2label[ix] for i, ix in enumerate(target_ixlabels)}

    for batch in train_batches:
        data_batch = unpickle(os.path.join(data_path, batch))
        curr_batch = process_batch(data_batch, target_ixlabels)

        cifarn['images'] += curr_batch['images']
        cifarn['labels'] += curr_batch['labels']

    # Convert to numpy array
    cifarn['images'] = np.array(cifarn['images'])
    cifarn['labels'] = np.array(cifarn['labels'])

    # Adjust dataset to meet the specified priors
    if prior is not None:
        imas, labels = match_priors(cifarn['images'], cifarn['labels'], prior)
        cifarn['images'], cifarn['labels'] = imas, labels

    if test:
        test_batch = unpickle(os.path.join(data_path, 'test_batch'))
        test_batch = process_batch(test_batch, target_ixlabels)

        cifarn['test_images'] = test_batch['images']
        cifarn['test_labels'] = test_batch['labels']

        # Convert to numpy array
        cifarn['test_images'] = np.array(cifarn['test_images'])
        cifarn['test_labels'] = np.array(cifarn['test_labels'])

        # Adjust test set to meet the specified priors
        if test_prior is not None:
            imas, labels = match_priors(
                    cifarn['test_images'],
                    cifarn['test_labels'],
                    test_prior)
            cifarn['test_images'], cifarn['test_labels'] = imas, labels

    return cifarn, cifarn_ix2label


def get_cifar10(data_path, test=False, prior=None, test_prior=None):
    cifar10 = {
        'images': [],
        'labels': []}

    train_batches = ["data_batch_{}".format(i+1) for i in range(5)]
    target_ixlabels = list(range(10))

    for batch in train_batches:
        data_batch = unpickle(os.path.join(data_path, batch))
        curr_batch = process_batch(data_batch, target_ixlabels)

        cifar10['images'] += curr_batch['images']
        cifar10['labels'] += curr_batch['labels']

    # Convert to numpy array
    cifar10['images'] = np.array(cifar10['images'])
    cifar10['labels'] = np.array(cifar10['labels'])

    # Adjust dataset to meet the specified priors
    if prior is not None:
        imas, labels = match_priors(
                cifar10['images'],
                cifar10['labels'],
                prior)
        cifar10['images'], cifar10['labels'] = imas, labels

    if test:
        test_batch = unpickle(os.path.join(data_path, 'test_batch'))
        test_batch = process_batch(test_batch, target_ixlabels)

        cifar10['test_images'] = test_batch['images']
        cifar10['test_labels'] = test_batch['labels']

        # Convert to numpy array
        cifar10['test_images'] = np.array(cifar10['test_images'])
        cifar10['test_labels'] = np.array(cifar10['test_labels'])

        # Adjust test set to meet the specified priors
        if test_prior is not None:
            imas, labels = match_priors(
                    cifar10['test_images'],
                    cifar10['test_labels'],
                    test_prior)
            cifar10['test_images'], cifar10['test_labels'] = imas, labels

    return cifar10, ix2label
