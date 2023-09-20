import numpy as np


class _Loss:
    """A private base class for loss functions."""

    def __call__(self, predictions: np.ndarray, targets: np.ndarray):
        """Computes the loss between predictions and targets."""
        raise NotImplementedError


class MSELoss(_Loss):
    """Mean Squared Error (MSE) loss function."""

    def __call__(self, predictions: np.ndarray, targets: np.ndarray):
        """Computes the MSE loss between predictions and targets."""
        assert predictions.shape == targets.shape, f"shapes {predictions.shape} and {targets.shape} don't align"
        return np.mean((predictions - targets) ** 2)


class CrossEntropyLoss(_Loss):
    """Cross Entropy loss function."""

    def __call__(self, predictions: np.ndarray, targets: np.ndarray):
        """Computes the cross entropy loss between predictions and targets"""
        assert predictions.shape == targets.shape, f"shapes {predictions.shape} and {targets.shape} don't align"
        predictions -= np.max(predictions, axis=1, keepdims=True)
        probs = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
        return -np.mean(np.sum(targets * np.log(probs), axis=1))


class HingeLoss(_Loss):
    """Hinge loss function."""

    def __call__(self, predictions: np.ndarray, targets: np.ndarray):
        """Computes the hinge loss between predictions and targets."""
        assert predictions.shape == targets.shape, f"shapes {predictions.shape} and {targets.shape} don't align"
        return np.mean(np.maximum(0, (1 - targets * predictions)))
