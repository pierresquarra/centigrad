import numpy as np

from .activation import Activation

class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilites