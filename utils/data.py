import os
import sys
import pickle
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_CIFAR3 = ['airplane', 'automobile', 'bird']
SAVE_PATH = r'..\experiments'
CONF = None
MODELS = [
    'flow',
    'dnn',
    'tscal',
    'bnn',
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

divergences = [
        'KL',
        'rKL',
        'F',
        'J',
        'A',
        'AR',
        'B',
        'G'
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
    parser.add_argument("--divergence", help='What divergence to use for gvi',
                        default='KL', choices=divergences)
    parser.add_argument("--d_param", help='divergence hyperparameter',
                        type=float, default=1)
    parser.add_argument("--p_mean", help='prior mean',
                        type=float, default=0)
    parser.add_argument("--p_var", help='prior var',
                        type=float, default=1)

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
        if CONF.model == 'bnn':
            CONF.name += '{}D_'.format(CONF.divergence)
            CONF.name += '{:.2f}_'.format(CONF.d_param)
            CONF.name += '{:.0e}_'.format(CONF.p_mean)
            CONF.name += '{:.0e}_'.format(CONF.p_var)
        if CONF.model in ['flow', 'dnn', 'bnn']:
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


def load_logits(dataset, model, data_path='../data'):

    # Build path.
    name = '_'.join([model, dataset])
    path = os.path.join(data_path, name)

    # Load logits and labels
    name = '_'.join([dataset, model])
    logits_train = np.load(os.path.join(path, name
                                        + '_logit_prediction_train.npy'))
    logits_val = np.load(os.path.join(path, name
                                      + '_logit_prediction_valid.npy'))
    logits_test = np.load(os.path.join(path, name
                                       + '_logit_prediction_test.npy'))

    true_train = np.load(os.path.join(path, name + '_true_train.npy'))
    true_val = np.load(os.path.join(path, name + '_true_valid.npy'))
    true_test = np.load(os.path.join(path, name + '_true_test.npy'))

    train = (torch.as_tensor(logits_train, dtype=torch.float32),
             torch.as_tensor(true_train, dtype=torch.int64))
    validation = (torch.as_tensor(logits_val, dtype=torch.float32),
                  torch.as_tensor(true_val, dtype=torch.int64))
    test = (torch.as_tensor(logits_test, dtype=torch.float32),
            torch.as_tensor(true_test, dtype=torch.int64))

    return train, validation, test


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
