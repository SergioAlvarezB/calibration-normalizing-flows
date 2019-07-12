import numpy as np
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation

from utils.ops import onehot_encode, optim_temperature
from flows.nice import NiceFlow


class Calibrator:
    def __init__(self, logits, target):
        """Implementes base abstract class for calibrators."""
        self.logits = logits

        if target.shape != logits.shape:
            target = onehot_encode(target)
        self.target = target

        (_, self.n_classes) = target.shape

        self.log_priors = self._get_log_priors(target)

    def _get_log_priors(self, target):
        priors = np.sum(target, axis=0)
        priors = priors/np.sum(priors)

        return np.log(priors)

    def predict(self, logits):
        raise NotImplementedError

    def predict_RL(self, logits):
        probs = self.predict(logits)
        return np.log(probs) - self.log_priors


class TempScalingCalibrator(Calibrator):

    def __init__(self, logits, target):
        super().__init__(logits, target)

        self.fit(logits, target)

    def fit(self, logits, target):
        self.T = optim_temperature(logits, target)

    def predict(self, logits):
        return softmax(logits/self.T, axis=1)


class PAVCalibrator(Calibrator):

    def __init__(self, logits, target):
        super().__init__(logits, target)

        # We use 1 vs rest approach, perform a
        # isotonic regression for each class
        self.models = [IsotonicRegression(y_min=0, y_max=1)
                       for _ in range(self.n_classes)]

        self.fit(logits, target)

    def fit(self, logits, target):

        probs = softmax(logits, axis=1)
        for cls in range(self.n_classes):
            model = self.models[cls]

            x, y = probs[:, cls], target[:, cls]
            model.fit(x, y)

    def predict(self, logits):
        probs = softmax(logits, axis=1)
        cal_probs = np.zeros(logits.shape)
        for cls in range(self.n_classes):
            model = self.models[cls]
            cal_probs[:, cls] = model.predict(probs[:, cls])

        cal_probs /= np.sum(cal_probs, axis=1, keepdims=True)

        return cal_probs


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
        self.flow = NiceFlow(input_dim=self.n_classes, **flow_args)

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

    def predict(self, logits):
        logits = self.predict_logits(logits)
        probs = softmax(logits, axis=1)
        return probs
