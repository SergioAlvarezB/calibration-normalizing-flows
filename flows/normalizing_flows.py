import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Activation
import tensorflow.keras.backend as K


class PlanarLayer(Layer):
    def __init__(self, **kwargs):
        super(PlanarLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize parameters.
        self.w = self.add_weight(name='w',
                                 shape=(int(input_shape[1]), 1),
                                 initializer='uniform',
                                 trainable=True)

        self.u = self.add_weight(name='u',
                                 shape=(1, int(input_shape[1])),
                                 initializer='uniform',
                                 trainable=True)

        self.b = self.add_weight(name='b',
                                 shape=(1,),
                                 initializer='uniform',
                                 trainable=True)

        # Call parent method.
        super(PlanarLayer, self).build(input_shape)

    def call(self, x):
        y = x + K.dot(Activation('tanh')(K.dot(x, self.w) + self.b), self.u)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class PlanarFlow:
    def __init__(self, input_shape, layers=5):
        self.layers = layers
        self.forward_model = self._flow_forward(input_shape)

    def _flow_forward(self, input_shape):
        inp = Input(shape=(input_shape,))
        x = inp

        for l in range(self.layers):
            x = PlanarLayer()(x)

        return Model(inputs=inp, outputs=x)
