import numpy as np

class Activation:
    def __init__(self):
        pass

    def forward(self, inputs: np.ndarray):
        """
        Computes the output of the activation function given some inputs.
        Subclasses must implement this method.
        
        Args:
        inputs (ndarray): The input data.
        
        Returns:
        ndarray: The output data.
        """
        raise NotImplementedError

    def backward(self, inputs: np.ndarray, grad: np.ndarray):
        """
        Computes the gradient of the activation function with respect to its inputs.
        Subclasses must implement this method.
        
        Args:
        inputs (ndarray): The input data.
        grad (ndarray): The gradient of the loss with respect to the activation's outputs.
        
        Returns:
        ndarray: The gradient of the loss with respect to the activation's inputs.
        """
        pass
