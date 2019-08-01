import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Concatenate, Dense, Input
import tensorflow.keras.backend as K


def MLP(input_dim, output_dim, hidden_size=[], activation='relu'):
    inp = Input(shape=(input_dim,))
    x = inp

    for h in hidden_size:
        x = Dense(h, activation=activation)(x)
    y = Dense(output_dim)(x)

    return Model(inputs=inp, outputs=y)


class Split(Layer):
    def __init__(self, **kwargs):
        super(Split, self).__init__(**kwargs)

    def call(self, inputs):
        dim = K.int_shape(inputs)[-1]
        idxs = list(range(dim))
        x = K.transpose(inputs)

        x1 = K.gather(x, idxs[:dim//2])
        x2 = K.gather(x, idxs[dim//2:])

        return K.transpose(x1), K.transpose(x2)

    def compute_output_shape(self, input_shape):
        dim = input_shape[-1]
        return [(None, dim//2), (None, dim - dim//2)]


class ReIndex(Layer):
    def __init__(self, idx=None, **kwargs):
        self.idx = idx
        super(ReIndex, self).__init__(**kwargs)

    def call(self, inputs):
        if self.idx is None:
            dim = K.int_shape(inputs)[-1]
            self.idx = list(range(dim-1, -1, -1))
        x = K.transpose(inputs)
        x = K.gather(x, self.idx)
        return K.transpose(x)

    def compute_output_shape(self, input_shape):
        return input_shape


class AddCouplingLayer(Layer):
    def __init__(self,
                 coupling_function,
                 mode='odd',
                 inverse=False,
                 **kwargs):
        self.coupling_func = coupling_function
        self.mode = mode
        self.inverse = inverse

        self.split = Split()
        self.concatenate = Concatenate()
        super(AddCouplingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x1, x2 = self.split(inputs)
        if self.mode == 'odd':
            x2 = x2 + self.coupling_func(x1)*(-1 if self.inverse else 1)
        else:
            x1 = x1 + self.coupling_func(x2)*(-1 if self.inverse else 1)

        return self.concatenate([x1, x2])

    def compute_output_shape(self, input_shape):
        return input_shape


class NiceFlow:
    def __init__(self,
                 input_dim,
                 layers=4,
                 hidden_size=None,
                 activation='relu'):
        self.input_dim = input_dim
        self.layers = layers

        if hidden_size is None:
            hidden_size = [input_dim]

        self.coup_funcs = self._get_coup_funcs(hidden_size, activation)
        self.forward_model = self._flow_forward()
        self.inverse_model = self._flow_inverse()

    def _flow_forward(self):
        inp = Input(shape=(self.input_dim,))
        x = inp
        for l in range(self.layers):
            x = AddCouplingLayer(
                    coupling_function=self.coup_funcs[l],
                    mode=('even' if l % 2 else 'odd'))(x)

        return Model(inputs=inp, outputs=x)

    def _flow_inverse(self):
        inp = Input(shape=(self.input_dim,))
        x = inp
        for l in range(self.layers-1, -1, -1):
            x = AddCouplingLayer(
                    coupling_function=self.coup_funcs[l],
                    mode=('even' if l % 2 else 'odd'),
                    inverse=True)(x)

        return Model(inputs=inp, outputs=x)

    def _get_coup_funcs(self, hidden_size, activation):
        coup_funcs = []
        for l in range(self.layers):
            if l % 2:
                inp_dim = self.input_dim - self.input_dim//2
                out_dim = self.input_dim//2
            else:
                inp_dim = self.input_dim//2
                out_dim = self.input_dim - self.input_dim//2
            coup_funcs.append(
                MLP(inp_dim,
                    output_dim=out_dim,
                    activation=activation,
                    hidden_size=hidden_size)
                )
        return coup_funcs


class NiceFlow_v2:
    def __init__(self,
                 input_dim,
                 layers=4,
                 hidden_size=None,
                 activation='relu'):
        self.input_dim = input_dim
        self.layers = layers

        if hidden_size is None:
            hidden_size = [input_dim]

        self.coup_funcs = self._get_coup_funcs(hidden_size, activation)
        self.forward_model = self._flow_forward()
        self.inverse_model = self._flow_inverse()

    def _flow_forward(self):
        inp = Input(shape=(self.input_dim,))
        x = inp
        for l in range(self.layers):
            x = AddCouplingLayer(
                    coupling_function=self.coup_funcs[l],
                    mode='even')(x)
            x = ReIndex()(x)
        if self.layers%2 == 1:
            x = ReIndex()(x)

        return Model(inputs=inp, outputs=x)

    def _flow_inverse(self):
        inp = Input(shape=(self.input_dim,))
        x = inp
        if self.layers%2 == 1:
            x = ReIndex()(x)
        for l in range(self.layers-1, -1, -1):
            x = ReIndex()(x)
            x = AddCouplingLayer(
                    coupling_function=self.coup_funcs[l],
                    mode='even',
                    inverse=True)(x)

        return Model(inputs=inp, outputs=x)

    def _get_coup_funcs(self, hidden_size, activation):
        coup_funcs = []
        inp_dim = self.input_dim - self.input_dim//2
        out_dim = self.input_dim//2
        for l in range(self.layers):
            coup_funcs.append(
                MLP(inp_dim,
                    output_dim=out_dim,
                    activation=activation,
                    hidden_size=hidden_size)
                )
        return coup_funcs
