from scipy.special import softmax

from tensorflow.keras import Model, Activation

from flows.nice import NiceFlow, NiceFlow_v2, NiceFlow_v3
from flows.realNVP import RealNvpFlow
from flows.normalizing_flows import PlanarFlow, RadialFlow
from ..calibrators import Calibrator


class NiceCalibrator(Calibrator):

    def __init__(self, logits, target, **kwargs):
        super().__init__(logits, target)

        self._build_flow(kwargs)

        self.history = self.fit(
                self.logits,
                self.target,
                epochs=kwargs.get('epochs', 1000),
                batch_size=kwargs.get('batch_size', 100))

    def _build_flow(self, kwargs):
        flow_args = {k: v for k, v in kwargs.items()
                     if k in ['layers', 'hidden_size', 'activation']}
        if kwargs.get('version', 3) == 1:
            self.flow = NiceFlow(input_dim=self.n_classes, **flow_args)
        elif kwargs.get('version', 3) == 2:
            self.flow = NiceFlow_v2(input_dim=self.n_classes, **flow_args)
        else:
            self.flow = NiceFlow_v3(input_dim=self.n_classes, **flow_args)

        self.train_model = self._train_model()
        self.train_model.compile(optimizer=kwargs.get('optimizer', 'adam'),
                                 loss='categorical_crossentropy')

    def _train_model(self):
        # Softmax output layer
        y = Activation('softmax')(self.flow.forward_model.output)
        return Model(inputs=self.flow.forward_model.input, outputs=y)

    def fit(self, logits, target, epochs=1000, batch_size=100):

        h = self.train_model.fit(
                logits,
                target,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)
        return h

    def predict_logits(self, logits):
        return self.flow.forward_model.predict(logits, batch_size=100)

    def predict_post(self, logits):
        logits = self.predict_logits(logits)
        probs = softmax(logits, axis=1)
        return probs


class PlanarFlowCalibrator(Calibrator):

    def __init__(self, logits, target, **kwargs):
        super().__init__(logits, target)

        self.flow = PlanarFlow(self.n_classes, kwargs.get('layers', 5))

        self.train_model = Model(
            inputs=self.flow.forward_model.input,
            outputs=Activation('softmax')(self.flow.forward_model.output)
        )
        self.train_model.compile(optimizer=kwargs.get('optimizer', 'adam'),
                                 loss='categorical_crossentropy')

        self.history = self.fit(self.logits,
                                self.target,
                                epochs=kwargs.get('epochs', 1000),
                                batch_size=kwargs.get('batch_size', 128))

    def fit(self, logits, target, epochs, batch_size):

        h = self.train_model.fit(
                logits,
                target,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)
        return h

    def predict_logits(self, logits):
        return self.flow.forward_model.predict(logits, batch_size=128)

    def predict_post(self, logits):
        logits = self.predict_logits(logits)
        probs = softmax(logits, axis=1)
        return probs


class RadialFlowCalibrator(Calibrator):

    def __init__(self, logits, target, **kwargs):
        super().__init__(logits, target)

        self.flow = RadialFlow(self.n_classes, kwargs.get('layers', 5))

        self.train_model = Model(
            inputs=self.flow.forward_model.input,
            outputs=Activation('softmax')(self.flow.forward_model.output)
        )
        self.train_model.compile(optimizer=kwargs.get('optimizer', 'adam'),
                                 loss='categorical_crossentropy')

        self.history = self.fit(self.logits,
                                self.target,
                                epochs=kwargs.get('epochs', 1000),
                                batch_size=kwargs.get('batch_size', 128))

    def fit(self, logits, target, epochs, batch_size):

        h = self.train_model.fit(
                logits,
                target,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)
        return h

    def predict_logits(self, logits):
        return self.flow.forward_model.predict(logits, batch_size=128)

    def predict_post(self, logits):
        logits = self.predict_logits(logits)
        probs = softmax(logits, axis=1)
        return probs


class RealNvpCalibrator(Calibrator):

    def __init__(self, logits, target, **kwargs):
        super().__init__(logits, target)

        self.flow = RealNvpFlow(self.n_classes,
                                kwargs.get('layers', 4),
                                kwargs.get('hidden_size', None),
                                kwargs.get('activation', 'relu'))

        self.train_model = Model(
            inputs=self.flow.forward_model.input,
            outputs=Activation('softmax')(self.flow.forward_model.output)
        )
        self.train_model.compile(optimizer=kwargs.get('optimizer', 'adam'),
                                 loss='categorical_crossentropy')

        self.history = self.fit(self.logits,
                                self.target,
                                epochs=kwargs.get('epochs', 1000),
                                batch_size=kwargs.get('batch_size', 128))

    def fit(self, logits, target, epochs, batch_size):

        h = self.train_model.fit(
                logits,
                target,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)
        return h

    def predict_logits(self, logits):
        return self.flow.forward_model.predict(logits, batch_size=128)

    def predict_post(self, logits):
        logits = self.predict_logits(logits)
        probs = softmax(logits, axis=1)
        return probs
