import numpy as np

from centigrad.engine import Value


class Module:
    def __call__(self, inputs: np.ndarray):
        return self.forward(inputs)

    def forward(self, inputs: np.ndarray):
        raise NotImplementedError

    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        k = np.sqrt(1 / in_features)
        self.weights = np.vectorize(Value)(np.random.uniform(low=-k, high=k, size=(out_features, in_features)))
        self.bias = np.vectorize(Value)(np.random.uniform(low=-k, high=k, size=(out_features,))) if bias else None

    def __repr__(self):
        return str(self.parameters())

    def __call__(self, inputs: np.ndarray):
        assert np.ndim(inputs) == 1, "input array is not 1D"
        assert self.weights.shape[1] == inputs.shape[
            0], f"shapes of weights {self.weights.shape[1]} and inputs {inputs.shape[0]} don't align"
        self.out = np.dot(self.weights, inputs)
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def forward(self, inputs: np.ndarray):
        return self(inputs)

    def parameters(self):
        return self.weights.flatten().tolist() + (self.bias.tolist() if self.bias is not None else [])


class Tanh(Module):
    def __call__(self, inputs: np.ndarray):
        assert np.ndim(inputs) == 1, "input array is not 1D"
        assert inputs.dtype == 'O', "inputs are not value objects"
        self.out = np.tanh(inputs)
        return self.out

    def forward(self, inputs: np.ndarray):
        return self(inputs)


class ReLU(Module):
    def __call__(self, inputs: np.ndarray):
        assert np.ndim(inputs) == 1, f"input array is not 1D ({inputs.shape})"
        assert inputs.dtype == 'O', f"inputs are not value objects ({inputs.dtype})"
        self.out = np.array([value.relu() for value in inputs])
        return self.out

    def forward(self, inputs: np.ndarray):
        return self(inputs)


class Sigmoid(Module):
    def __call__(self, inputs: np.ndarray):
        assert np.ndim(inputs) == 1, f"input array is not 1D ({inputs.shape})"
        assert inputs.dtype == 'O', f"inputs are not value objects ({inputs.dtype})"
        self.out = np.array([value.sigmoid() for value in inputs])
        return self.out

    def forward(self, inputs: np.ndarray):
        return self(inputs)


class Softmax(Module):
    def __call__(self, inputs: np.ndarray):
        assert np.ndim(inputs) == 1, f"input array is not 1D ({inputs.shape})"
        assert inputs.dtype == 'O', f"inputs are not value objects ({inputs.dtype})"
        self.out = np.array([np.exp(v) / np.sum(np.exp(inputs)) for v in inputs])
        return self.out

    def forward(self, inputs: np.ndarray):
        return self(inputs)


class LogSoftMax(Module):
    def __call__(self, inputs: np.ndarray):
        assert np.ndim(inputs) == 1, f"input array is not 1D ({inputs.shape})"
        assert inputs.dtype == 'O', f"inputs are not value objects ({inputs.dtype})"
        self.out = np.log([np.exp(v) / np.sum(np.exp(inputs)) for v in inputs])
        return self.out

    def forward(self, inputs: np.ndarray):
        return self(inputs)
