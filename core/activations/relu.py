import numpy as np

from .activation import Activation

class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.ndarray):
        return np.maximum(0, inputs)
