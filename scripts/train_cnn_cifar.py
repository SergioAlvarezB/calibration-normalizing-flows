import os
import json
import pickle
import argparse

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from utils.data import get_cifar3, get_cifar10
from utils.ops import onehot_encode

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_path', type=str, default='cifar-10')
parser.add_argument('--epochs', type=int, default=160)
parser.add_argument('--name', type=str, default='cnn_cifar')
parser.add_argument('--save_dir', type=str, default='pretrained-models')
parser.add_argument('-3', '--cifar3', action="store_true",
                    help='If set, it will train the net on cifar3 alone')
parser.add_argument('--flip_sets', action="store_true",
                    help='If set, test set is used for training and viceversa')

config = parser.parse_args()

batch_size = config.batch_size
data_path = config.data_path
epochs = config.epochs
model_name = config.name
save_dir = config.save_dir

if config.cifar3:
    cifar, ix2label = get_cifar3(data_path, test=True)
    n_classes = 3
else:
    cifar, ix2label = get_cifar10(data_path, test=True)
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

# Build model
inp = Input(shape=(32, 32, 3))

x = Conv2D(16, (3, 3), padding='same')(inp)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), padding='same')(inp)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(256)(x)
x = Activation('relu')(x)
x = Dense(n_classes)(x)
y = Activation('softmax')(x)

model = Model(inputs=inp, outputs=y)

logit_model = Model(inputs=inp, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train
H = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

# Save model and weights.
save_dir = os.path.join(save_dir, model_name)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name+'.h5')
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
probs = model.predict(x_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Save model details.
H = H.history
H["batch_size"] = batch_size
H["n_classes"] = n_classes
H["trainable_parameters"] = model.count_params()
H['Test loss'] = scores[0]
H['Test Accuracy'] = scores[1]

try:
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(H, f)
except Exception as e:
    print('######Impossible to save history: \n {}'.format(e))

# Save logits.
train_logits = logit_model.predict(x_train, batch_size=batch_size)
test_logits = logit_model.predict(x_test, batch_size=batch_size)

with open(os.path.join(save_dir, 'train_logits.pkl'), 'wb') as f:
    pickle.dump(train_logits, f)

with open(os.path.join(save_dir, 'test_logits.pkl'), 'wb') as f:
    pickle.dump(test_logits, f)
