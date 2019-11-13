import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation

from utils.ops import onehot_encode, optim_temperature
from flows.nice import NiceFlow, NiceFlow_v2, NiceFlow_v3
from flows.realNVP import RealNvpFlow
from flows.normalizing_flows import PlanarFlow, RadialFlow


class Calibrator:
    def __init__(self, logits, target):
        """Implementes base abstract class for calibrators."""
        self.logits = logits

        if target.shape != logits.shape:
            target = onehot_encode(target)
        self.target = target

        (_, self.n_classes) = target.shape

        self.log_priors = self._get_log_priors(target)

    def __call__(self, logits):
        return self.predict(logits)

    def _get_log_priors(self, target):
        priors = np.sum(target, axis=0)
        priors = priors/np.sum(priors)

        return np.log(priors)

    def predict_post(self, logits):
        raise NotImplementedError

    def predict(self, logits):
        probs = self.predict_post(logits)
        return softmax(np.log(probs) - self.log_priors, axis=1)


class DummyCalibrator(Calibrator):
    """Implements uncalibrated model."""

    def __init__(self, logits, target):
        super().__init__(logits, target)

    def predict_post(self, logits):
        return softmax(logits, axis=1)


class TempScalingCalibrator(Calibrator):

    def __init__(self, logits, target):
        super().__init__(logits, target)

        self.fit(self.logits, self.target)

    def fit(self, logits, target):
        self.T = optim_temperature(logits, target)

    def predict_post(self, logits):

        logits = logits - np.mean(logits, axis=1, keepdims=True)

        return softmax(logits/self.T, axis=1)


class MatrixScalingCalibrator(Calibrator):

    def __init__(self, logits, target):
        super().__init__(logits, target)
        self.n = logits.shape[1]  # Number of input dimensions.
        self.W = np.zeros([self.n, self.n])
        self.b = np.zeros(self.n)

        self.fit(self.logits, self.target)

    def fit(self, logits, target):

        def target_func(x, logits, target):
            tlogits = logits @ x[self.n:].reshape([self.n, self.n]) \
                + x[:self.n]
            probs = softmax(tlogits, axis=1)
            return np.mean(-np.sum(target*np.log(probs+1e-7), axis=1))

        def grads(x, logits, target):
            grad = np.zeros(x.shape)
            tlogits = logits @ x[self.n:].reshape([self.n, self.n]) \
                + x[:self.n]
            probs = softmax(tlogits, axis=1)
            dW = np.mean((probs-target).reshape([-1, self.n, 1])
                         @ logits.reshape([-1, 1, self.n]), axis=0).T
            db = np.mean((probs-target), axis=0)
            grad[:self.n], grad[self.n:] = db, dW.ravel()

            return grad

        x0 = np.concatenate((self.b, self.W.ravel()))

        self.optim = minimize(
                target_func,
                x0=x0,
                args=(logits, target),
                method='CG',
                jac=grads)

        self.b = self.optim.x[:self.n]
        self.W = self.optim.x[self.n:].reshape([self.n, self.n])

    def predict_post(self, logits):
        logits = logits - np.mean(logits, axis=1, keepdims=True)

        tlogits = logits @ self.W + self.b
        probs = softmax(tlogits, axis=1)
        return probs


class VectorScalingCalibrator(Calibrator):

    def __init__(self, logits, target):
        super().__init__(logits, target)
        self.n = logits.shape[1]  # Number of input dimensions.
        self.W = np.zeros(self.n)
        self.b = np.zeros(self.n)

        self.fit(self.logits, self.target)

    def fit(self, logits, target):

        def target_func(x, logits, target):
            tlogits = x[self.n:] * logits + x[:self.n]
            probs = softmax(tlogits, axis=1)
            return np.mean(-np.sum(target*np.log(probs+1e-7), axis=1))

        def grads(x, logits, target):
            grad = np.zeros(x.shape)
            tlogits = x[self.n:] * logits + x[:self.n]
            probs = softmax(tlogits, axis=1)
            dW = np.mean((probs-target) * logits, axis=0)
            db = np.mean((probs-target))
            grad[:self.n], grad[self.n:] = db, dW

            return grad

        x0 = np.concatenate((self.b, self.W))

        self.optim = minimize(
                target_func,
                x0=x0,
                args=(logits, target),
                method='CG',
                jac=grads)

        self.b = self.optim.x[:self.n]
        self.W = self.optim.x[self.n:]

    def predict_post(self, logits):
        logits = logits - np.mean(logits, axis=1, keepdims=True)

        tlogits = self.W * logits + self.b
        probs = softmax(tlogits, axis=1)
        return probs


class MLRCalibrator(Calibrator):

    def __init__(self, logits, target):
        super().__init__(logits, target)
        self.alpha = 1.
        self.gamma = np.zeros(self.n_classes)

        self.fit(self.logits, self.target)

    def fit(self, logits, target):

        def target_func(x, logits, target):
            tlogits = x[0]*logits + x[1:]
            probs = softmax(tlogits, axis=1)
            return np.mean(-np.sum(target*np.log(probs+1e-7), axis=1))

        def grads(x, logits, target):
            grad = np.zeros(x.shape)
            tlogits = x[0]*logits + x[1:]
            probs = softmax(tlogits, axis=1)
            dalpha = np.mean(np.sum((probs-target)*logits, axis=1))
            dgamma = np.mean((probs-target), axis=0)
            grad[0], grad[1:] = dalpha, dgamma

            return grad

        x0 = np.concatenate(([self.alpha], self.gamma))

        self.optim = minimize(
                target_func,
                x0=x0,
                args=(logits, target),
                method='CG',
                jac=grads)

        self.alpha = self.optim.x[0]
        self.gamma = self.optim.x[1:]

    def predict_post(self, logits):
        logits = logits - np.mean(logits, axis=1, keepdims=True)

        tlogits = self.alpha*logits + self.gamma
        probs = softmax(tlogits, axis=1)
        return probs


class PAVCalibrator(Calibrator):

    def __init__(self, logits, target):
        super().__init__(logits, target)

        # We use 1 vs rest approach, perform a
        # isotonic regression for each class
        self.models = [IsotonicRegression(y_min=0,
                                          y_max=1,
                                          out_of_bounds='clip')
                       for _ in range(self.n_classes)]

        self.fit(self.logits, self.target)

    def fit(self, logits, target):
        probs = softmax(logits, axis=1).astype(np.float)
        for cls, model in enumerate(self.models):
            x, y = probs[:, cls], target[:, cls]
            model.fit(x, y)

    def predict_post(self, logits):
        probs = softmax(logits, axis=1).astype(np.float)
        cal_probs = np.zeros(logits.shape)
        for cls, model in enumerate(self.models):
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
        # Normalize input to net.
        logits = logits - np.mean(logits, axis=1, keepdims=True)

        h = self.train_model.fit(
                logits,
                target,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)
        return h

    def predict_logits(self, logits):
        # Normalize input to net.
        logits = logits - np.mean(logits, axis=1, keepdims=True)

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
        # Normalize input to net.
        logits = logits - np.mean(logits, axis=1, keepdims=True)

        h = self.train_model.fit(
                logits,
                target,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)
        return h

    def predict_logits(self, logits):
        # Normalize input to net.
        logits = logits - np.mean(logits, axis=1, keepdims=True)

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
        # Normalize input to net.
        logits = logits - np.mean(logits, axis=1, keepdims=True)

        h = self.train_model.fit(
                logits,
                target,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)
        return h

    def predict_logits(self, logits):
        # Normalize input to net.
        logits = logits - np.mean(logits, axis=1, keepdims=True)

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
        # Normalize input to net.
        logits = logits - np.mean(logits, axis=1, keepdims=True)

        h = self.train_model.fit(
                logits,
                target,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0)
        return h

    def predict_logits(self, logits):
        # Normalize input to net.
        logits = logits - np.mean(logits, axis=1, keepdims=True)

        return self.flow.forward_model.predict(logits, batch_size=128)

    def predict_post(self, logits):
        logits = self.predict_logits(logits)
        probs = softmax(logits, axis=1)
        return probs
