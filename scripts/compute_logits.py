import os
import sys
import pickle
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from utils.data import get_cifar3, get_cifar10
from utils.ops import onehot_encode

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str,
                    help='Path to the directory of the model.')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--data_path', type=str, default='cifar-10')
parser.add_argument('--name', type=str, default='model.h5',
                    help='Filename inside model directory')
parser.add_argument('-3', '--cifar3', action="store_true",
                    help='Whether the model was trained using cifar3')
parser.add_argument('--flip_sets', action="store_true",
                    help='whether the model was trained on test set')

config = parser.parse_args()

if config.cifar3:
    cifar, ix2label = get_cifar3(config.data_path, test=True)
    n_classes = 3
else:
    cifar, ix2label = get_cifar10(config.data_path, test=True)
    n_classes = 10

# Pre-process data
y_train = onehot_encode(cifar['labels'])
y_test = onehot_encode(cifar['test_labels'])

x_train = cifar['images'].astype('float32')
x_test = cifar['test_images'].astype('float32')

if config.flip_sets:
    x_train, x_test = x_test, x_train
    y_train, y_test = y_test, y_train

x_train /= 255.
train_mean = np.mean(x_train, axis=0)
x_train -= train_mean
x_test = x_test/255. - train_mean

try:
    model = tf.keras.models.load_model(
            os.path.join(config.model_dir, config.name),
            custom_objects={'tf': tf})
    print("Model loaded succesfully.")

except Exception as e:
    print(("Unable to load model at ''{}''. Raised exception: \n "
           + "{}").format(config.model_dir, e))
    sys.exit()

# Remove softmax layer.
logits = model.layers[-2].output
logit_model = Model(inputs=model.input, outputs=logits)

# Save logits.
train_logits = logit_model.predict(x_train, batch_size=config.batch_size)
test_logits = logit_model.predict(x_test, batch_size=config.batch_size)


with open(os.path.join(config.model_dir, 'train_logits.pkl'), 'wb') as f:
    pickle.dump(train_logits, f)

with open(os.path.join(config.model_dir, 'test_logits.pkl'), 'wb') as f:
    pickle.dump(test_logits, f)
