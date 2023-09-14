import numpy as np

class Loss:
    def __init__(self):
        pass

    def forward(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Computes the loss given the predictions and the ground truth targets.
        Subclasses must implement this method.

        Args:
        predictions (ndarray): The predictions from the model.
        targets (ndarray): The ground truth target values.

        Returns:
        float: The computed loss value.
        """
        raise NotImplementedError

    def backward(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Computes the gradient of the loss with respect to the predictions.
        Subclasses must implement this method.

        Args:
        predictions (ndarray): The predictions from the model.
        targets (ndarray): The ground truth target values.

        Returns:
        ndarray: The gradient of the loss with respect to the predictions.
        """
        pass
