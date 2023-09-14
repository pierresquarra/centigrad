import numpy as np
from .loss import Loss


class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: np.ndarray, targets: np.ndarray):
        samples = len(predictions)
        y_pred_clipped = np.clip(predictions, 1e-7, 1-1e-7) # prevent -log(0) = inf

        # check if scalar values or one hot encoded values have been passed
        # scalar: [0, 1]
        # one-hot encoded: [[1,0],[0,1]]
        if len(targets.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), targets]
        else:
            correct_confidences = np.sum(y_pred_clipped*targets, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
