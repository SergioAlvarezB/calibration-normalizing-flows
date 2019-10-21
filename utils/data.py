import os
import pickle

import numpy as np


DEFAULT_CIFAR3 = ['airplane', 'automobile', 'bird']

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
