import os
import pickle

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from utils.data import get_cifar3
from utils.ops import onehot_encode


batch_size = 64
epochs = 2
data_path = 'cifar-10'
save_dir = 'pretrained-models'
model_name = 'cnn_cifar'
save_logits = True


cifar3, ix2label = get_cifar3(data_path, test=True)

# Use only 10% of the data to favour overfitting
n_samples = cifar3['images'].shape[0]
idx = np.random.permutation(n_samples)[:int(n_samples//10)]
cifar3['images'] = cifar3['images'][idx]
cifar3['labels'] = cifar3['labels'][idx]

# Pre-process data
y_train = onehot_encode(cifar3['labels'])
y_test = onehot_encode(cifar3['test_labels'])

x_train = cifar3['images'].astype('float32')
x_test = cifar3['test_images'].astype('float32')
x_train /= 255.
x_test /= 255.


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
x = Dense(3)(x)
y = Activation('softmax')(x)

model = Model(inputs=inp, outputs=y)

if save_logits:
    logit_model = Model(inputs=inp, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train
h = model.fit(x_train, y_train,
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

# Save logits.
train_logits = logit_model.predict(x_train, batch_size=batch_size)
test_logits = logit_model.predict(x_test, batch_size=batch_size)

with open(os.path.join(save_dir, 'train_logits.pkl'), 'wb') as f:
    pickle.dump(train_logits, f)

with open(os.path.join(save_dir, 'test_logits.pkl'), 'wb') as f:
    pickle.dump(test_logits, f)
