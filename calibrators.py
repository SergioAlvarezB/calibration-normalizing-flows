import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.ops import onehot_encode, optim_temperature


class Calibrator:
    def __init__(self, logits, target):
        """Implementes base abstract class for calibrators."""
        # Normalize logits.
        logits = logits - np.mean(logits, axis=1, keepdims=True)
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
        # Normalize logits.
        logits = logits - np.mean(logits, axis=1, keepdims=True)
        probs = self.predict_post(logits)
        return softmax(np.log(probs + 1e-7) - self.log_priors, axis=1)


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
            tlogits = logits @ (x[self.n:].reshape([self.n, self.n])
                                + x[:self.n])
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


class TorchFlowCalibrator(Calibrator):

    def __init__(self, Flow, logits, target, **kwargs):
        super().__init__(logits, target)

        # Torch CE expects int labels.
        self.target = np.argmax(self.target, axis=1)

        # Convert data to torch tensors.
        self.logits = torch.as_tensor(self.logits, dtype=torch.float)
        self.target = torch.as_tensor(self.target, dtype=torch.long)

        self.flow = Flow(self.n_classes, **kwargs)

        self.dev = (kwargs.get('dev',
                               torch.device("cuda")
                               if torch.cuda.is_available()
                               else torch.device("cpu")))

        self.CE = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.flow.parameters())

        self.history = self.fit(self.logits,
                                self.target,
                                epochs=kwargs.get('epochs', 1000),
                                batch_size=kwargs.get('batch_size', 128))

    def fit(self, logits, target, epochs, batch_size):
        # Send data to device.
        logits = logits.to(self.dev)
        target = target.to(self.dev)
        self.flow.to(self.dev)

        # Data loader.
        train_ds = TensorDataset(logits, target)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        history = {}
        history['loss'] = []

        for epoch in range(epochs):
            self.flow.train()
            for xb, yb in train_dl:
                pred = self.flow(xb)
                loss = self.CE(pred, yb)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.flow.eval()
            _loss = 0
            num = 0
            with torch.no_grad():
                for xb, yb in train_dl:
                    pred = self.flow(xb)
                    _loss += self.CE(pred, yb).item() * len(xb)
                    num += len(xb)
                history['loss'].append(_loss/num)

        # Return data to cpu
        self.flow.cpu()

        # Delete reference to unused variables.
        del logits, target, train_ds, train_dl, pred, loss

        # Clean GPU memory.
        torch.cuda.empty_cache()

        return history

    def predict_logits(self, logits):
        # Create logits tensor.
        logits = torch.as_tensor(logits, dtype=torch.float)

        # Send data to decive.
        self.flow.to(self.dev)
        logits = logits.to(self.dev)

        # Predictions.
        preds = self.flow(logits)

        # Return data to cpu
        self.flow.cpu()
        preds = preds.cpu().detach().numpy()

        # Clean GPU memory.
        torch.cuda.empty_cache()

        return preds

    def predict_post(self, logits):
        logits = self.predict_logits(logits)
        probs = softmax(logits, axis=1)
        return probs
