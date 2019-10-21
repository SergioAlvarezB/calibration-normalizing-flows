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
        # Ensure w*u >= -1 to mantain invertibility.
        wtu = K.dot(self.u, self.w)
        m = -1 + K.log(1 + K.exp(wtu))

        u_hat = self.u + (m - wtu)*K.transpose(K.l2_normalize(self.w, axis=0))

        y = x + K.dot(Activation('tanh')(K.dot(x, self.w) + self.b), u_hat)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class RadialLayer(Layer):

    def __init__(self, **kwargs):
        super(RadialLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize parameters.
        self.z0 = self.add_weight(name='z0',
                                  shape=(1, int(input_shape[1])),
                                  initializer='uniform',
                                  trainable=True)

        self.a = self.add_weight(name='a',
                                 shape=(1,),
                                 initializer='uniform',
                                 trainable=True)

        self.b = self.add_weight(name='b',
                                 shape=(1,),
                                 initializer='uniform',
                                 trainable=True)

        # Call parent method.
        super(RadialLayer, self).build(input_shape)

    def call(self, x):
        # Ensure b >= -a to mantain invertibility.
        b_hat = -self.a + K.log(1 + K.exp(self.b))

        x_z0 = x - self.z0
        r = K.sqrt(K.sum(K.square(x_z0), axis=1))
        h = 1./(self.a + r)
        y = x + b_hat * K.expand_dims(h, axis=-1) * x_z0

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


class RadialFlow:
    def __init__(self, input_shape, layers=5):
        self.layers = layers
        self.forward_model = self._flow_forward(input_shape)

    def _flow_forward(self, input_shape):
        inp = Input(shape=(input_shape,))
        x = inp

        for l in range(self.layers):
            x = RadialLayer()(x)

        return Model(inputs=inp, outputs=x)
