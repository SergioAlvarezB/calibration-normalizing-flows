import numpy as np
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation

from utils.ops import onehot_encode
from flows.nice import NiceFlow


class PAVCalibrator:

    def __init__(self, logits, target):
        self.logits = logits
        if target.shape != logits.shape:
            target = onehot_encode(target)
        self.target = target
        (_, self.n_classes) = target.shape

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


class NiceCalibrator:

    def __init__(self, logits, target, **kwargs):
        self.logits = logits

        if target.shape != logits.shape:
            target = onehot_encode(target)
        self.target = target

        (_, self.n_classes) = target.shape

        self._build_flow(kwargs)

        self.fit(self.logits,
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
        self.train_model.fit(logits,
                target,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)

    def predict_logits(self, logits):
        return self.flow.forward_model.predict(logits, batch_size=100)

    def predict(self, logits):
        logits = self.predict_logits(logits)
        probs = softmax(logits, axis=1)
        return probs
