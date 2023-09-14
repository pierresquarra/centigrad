import numpy as np

class Layer:
    def __init__(self):
        self.trainable = True
    
    def forward(self, inputs: np.ndarray):
        """
        Computes the output of the layer given some inputs.
        Subclasses must implement this method.
        
        Args:
        inputs (ndarray): The input data.
        
        Returns:
        ndarray: The output data.
        """
        raise NotImplementedError
    
    def backward(self, grad: np.ndarray):
        """
        Computes the gradient of the layer with respect to its parameters and inputs.
        Subclasses must implement this method.
        
        Args:
        grad (ndarray): The gradient of the loss with respect to the layer's outputs.
        
        Returns:
        ndarray: The gradient of the loss with respect to the layer's inputs.
        """
        pass
    
    def update_params(self, optimizer):
        """
        Updates the layer's parameters using the optimizer.
        Subclasses must implement this method if the layer has parameters.
        
        Args:
        optimizer (Optimizer): The optimizer to use for the parameter updates.
        """
        pass
