import numpy as np
from .layer import Layer


class Dense(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.random.randn(input_dim, output_dim)
        self.biases = np.zeros((output_dim))

    def forward(self, inputs: np.ndarray):
        return np.dot(inputs, self.weights) + self.biases