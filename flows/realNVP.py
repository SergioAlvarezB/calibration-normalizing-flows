import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.backend as K


def MLP(dim, hidden_size=[], activation='relu'):
    inp = Input(shape=(dim,))
    x = inp

    for h in hidden_size:
        x = Dense(h, activation=activation)(x)
    y = Dense(dim)(x)

    return Model(inputs=inp, outputs=y)


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
            coup_funcs.append(MLP(self.dim, hidden_size, activation),
                              MLP(self.dim, hidden_size, activation))

        return coup_funcs

    def _flow_forward(self):
        b = np.zeros(self.dim)
        b[self.dim//2:] = 1.

        inp = Input(shape=(self.dim,))
        x = inp
        for l, (s, t) in enumerate(self.coup_funcs):
            x_b = K.constant(b) @ x
            x = x_b + K.constant(1. - b) @ (K.exp(s(x_b)) + t(x_b))

            b = np.flip(b)

        return Model(inputs=inp, outputs=x)

    def _flow_backward(self):
        b = np.zeros(self.dim)
        b[self.dim//2:] = 1.

        if self.layers % 2 == 0:
            b = np.flip(b)

        inp = Input(shape=(self.dim,))
        x = inp
        for l in range(self.layers-1, -1, -1):
            s, t = self.coup_funcs[l]
            x_b = K.constant(b) @ x
            x = x_b + K.constant(1. - b) @ (K.exp(s(x_b)) + t(x_b))

            b = np.flip(b)

        return Model(inputs=inp, outputs=x)
