import numpy as np

from core.new import Module


class _Loss(Module):
    def forward(self, predictions: np.ndarray, targets: np.ndarray):
        raise NotImplementedError

    def __call__(self, predictions: np.ndarray, targets: np.ndarray):
        return self.forward(predictions, targets)


class MSELoss(_Loss):
    def forward(self, predictions: np.ndarray, targets: np.ndarray):
        assert predictions.shape == targets.shape, "shapes don't align"
        assert predictions.ndim == targets.ndim == 1, "inputs are not one dimensional"
        num_samples = predictions.shape[0]
        return np.sum(((predictions - targets) ** 2)) / num_samples


class CrossEntropyLoss(_Loss):
    def forward(self, predictions: np.ndarray, targets: np.ndarray, epsilon=1e-12):
        assert predictions.shape[0] == targets.shape[0], "shapes don't align"
        assert len(targets.shape) == 1, "targets must be 1D array"
        size = predictions.shape[1]
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        if targets.dtype == 'O':
            targets = np.eye(size)[[target.data for target in targets]]
        else:
            targets = np.eye(size)[targets]
        # targets are not values anymore
        return np.sum(-np.log(np.sum(predictions * targets, axis=1)))


class CategoricalCrossEntropyLoss(_Loss):
    def forward(self, predictions: np.ndarray, targets: np.ndarray, epsilon=1e-12):
        assert predictions.shape == targets.shape, "shapes don't align"
        assert predictions.ndim == targets.ndim == 1, "inputs are not two dimensional"
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        return -(np.sum(predictions * targets, axis=1)).log()

