import numpy as np
from .loss import Loss


class MSE(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: np.ndarray, targets: np.ndarray):
        assert predictions.shape == targets.shape
        n = len(predictions)
        return np.sum((predictions - targets) ** 2) / n
