import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Input
import tensorflow.keras.backend as K


def MLP(dim, hidden_size=[], activation='relu'):
    inp = Input(shape=(dim,))
    x = inp

    for h in hidden_size:
        x = Dense(h, activation=activation)(x)
    y = Dense(dim)(x)

    return Model(inputs=inp, outputs=y)


class NvpCoupling(Layer):
    def __init__(self, s, t, mask, backward=False, **kwargs):
        self.s = s
        self.t = t
        self.mask = mask
        self.backward = backward

        super(NvpCoupling, self).__init__(**kwargs)

    def call(self, inputs):

        x_b = K.constant(self.mask) * inputs
        b_1 = K.constant(1. - self.mask)

        if self.backward:
            y = x_b \
                + b_1 * ((inputs - self.t(x_b)) / K.exp(self.s(x_b)))
        else:
            y = x_b + b_1 * (inputs * K.exp(self.s(x_b)) + self.t(x_b))

        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class RealNvpFlow:

    def __init__(self, dim, layers=4, hidden_size=None, activation='relu'):
        self.dim = dim
        self.layers = layers

        if hidden_size is None:
            hidden_size = [dim]

        self.coup_funcs = self._get_coup_funcs(hidden_size, activation)
        self.forward_model = self._flow_forward()
        self.backward_model = self._flow_backward()

    def _get_coup_funcs(self, hidden_size, activation):
        coup_funcs = []
        for l in range(self.layers):
            coup_funcs.append((MLP(self.dim, hidden_size, 'tanh'),
                               MLP(self.dim, hidden_size, activation)))

        return coup_funcs

    def _flow_forward(self):
        b = np.zeros((1, self.dim))
        b[:, self.dim//2:] = 1.

        inp = Input(shape=(self.dim,))
        x = inp
        for l, (s, t) in enumerate(self.coup_funcs):
            x = NvpCoupling(s, t, b)(x)
            b = np.flip(b)

        return Model(inputs=inp, outputs=x)

    def _flow_backward(self):
        b = np.zeros((1, self.dim))
        b[:, self.dim//2:] = 1.

        if self.layers % 2 == 0:
            b = np.flip(b)

        inp = Input(shape=(self.dim,))
        x = inp
        for l in range(self.layers-1, -1, -1):
            s, t = self.coup_funcs[l]
            x = NvpCoupling(s, t, b, backward=True)(x)
            b = np.flip(b)

        return Model(inputs=inp, outputs=x)
