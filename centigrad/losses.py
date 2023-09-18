import numpy as np

from centigrad.nn import Module


class _Loss(Module):
    def forward(self, predictions: np.ndarray, targets: np.ndarray):
        raise NotImplementedError

    def __call__(self, predictions: np.ndarray, targets: np.ndarray):
        return self.forward(predictions, targets)


class MSELoss(_Loss):
    def forward(self, predictions: np.ndarray, targets: np.ndarray):
        assert predictions.shape == targets.shape, f"shapes {predictions.shape} and {targets.shape} don't align"
        assert predictions.ndim == targets.ndim == 1, "inputs are not one dimensional"
        num_samples = predictions.shape[0]
        return np.sum(((predictions - targets) ** 2)) / num_samples


class CrossEntropyLoss(_Loss):
    def forward(self, predictions: np.ndarray, targets: np.ndarray, epsilon=1e-12):
        assert predictions.shape == targets.shape, f"shapes {predictions.shape} and {targets.shape} don't align"
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        losses = -np.log(np.sum(predictions * targets, axis=1))
        return np.sum(losses) / predictions.shape[0]


class HingeLoss(_Loss):
    def forward(self, predictions: np.ndarray, targets: np.ndarray):
        assert predictions.shape == targets.shape, f"shapes {predictions.shape} and {targets.shape} don't align"
        losses = np.maximum(0, (1 + -targets * predictions))
        return np.sum(losses) * (1.0 / losses.shape[0])
