import numpy as np

from centigrad.engine import Value


class Module:
    def forward(self, inputs: np.ndarray):
        raise NotImplementedError

    def __call__(self, inputs: np.ndarray):
        return self.forward(inputs)

    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, num_inputs, num_outputs, low=0.0, high=1.0):
        self.weights = np.vectorize(Value)(np.random.uniform(low=low, high=high, size=(num_outputs, num_inputs)), label='w')
        self.bias = np.vectorize(Value)(np.random.uniform(low=low, high=high, size=num_outputs), label='b')

    def __repr__(self):
        return str(self.parameters())

    def forward(self, inputs: np.ndarray):
        assert np.ndim(inputs) == 1, "input array is not 1D"
        assert self.weights.shape[1] == inputs.shape[0], "shapes of layer and inputs don't align"
        return np.dot(self.weights, inputs) + self.bias

    def parameters(self):
        return list(np.concatenate((self.weights.flatten(), self.bias)))


class Tanh(Module):
    def __repr__(self):
        return "Tanh layer"

    def forward(self, inputs: np.ndarray):
        assert np.ndim(inputs) == 1, "input array is not 1D"
        assert inputs.dtype == 'O', "inputs are not value objects"
        return np.array([value.tanh() for value in inputs])


class ReLU(Module):
    def __repr__(self):
        return "ReLu layer"

    def forward(self, inputs: np.ndarray):
        assert np.ndim(inputs) == 1, "input array is not 1D"
        assert inputs.dtype == 'O', "inputs are not value objects"
        return np.array([value.relu() for value in inputs])


class Sigmoid(Module):
    def __repr__(self):
        return "Sigmoid layer"

    def forward(self, inputs: np.ndarray):
        assert np.ndim(inputs) == 1, "input array is not 1D"
        assert inputs.dtype == 'O', "inputs are not value objects"
        return np.array([value.sigmoid() for value in inputs])


class Softmax(Module):
    def __repr__(self):
        return "Softmax layer"

    def forward(self, inputs: np.ndarray):
        assert np.ndim(inputs) == 1, "input array is not 1D"
        assert inputs.dtype == 'O', "inputs are not value objects"
        exp_values = np.array([np.exp(value) for value in inputs])
        return np.array([exp_value / np.sum(exp_values) for exp_value in exp_values])
