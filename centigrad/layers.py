import numpy as np

from centigrad.engine import Value
from centigrad.nn import Module


class _Layer(Module):
    """Private base class for neural network layers"""

    def __call__(self, inputs: np.ndarray):
        """Applies the layer transformation to the input."""
        raise NotImplementedError

    def forward(self, inputs: np.ndarray):
        """Alias for the __call__ method, applying the layer transformation to the input."""
        return self(inputs)

    def parameters(self):
        """Gets the parameters of the layer."""
        return super().parameters()


class Linear(_Layer):
    """Simple linear (dense) layer which applies a linear transformation to the input data."""

    def __init__(self, in_features, out_features, bias=True):
        k = np.sqrt(1 / in_features)
        self.weight = np.vectorize(Value)(np.random.uniform(low=-k, high=k, size=(out_features, in_features)))
        self.bias = np.vectorize(Value)(np.random.uniform(low=-k, high=k, size=(out_features,))) if bias else None

    def __call__(self, inputs: np.ndarray):
        self.out = np.dot(inputs, self.weight.T) + (self.bias if self.bias is not None else 0)
        return self.out

    def parameters(self):
        return self.weight.flatten().tolist() + (self.bias.tolist() if self.bias is not None else [])


class Tanh(_Layer):
    """Layer that applies the hyperbolic tangent activation function element-wise to the input data."""

    def __call__(self, inputs: np.ndarray):
        self.out = np.tanh(inputs)
        return self.out


class ReLU(_Layer):
    """Layer that applies the Rectified Linear Unit (ReLU) activation function element-wise to the input data."""

    def __call__(self, inputs: np.ndarray):
        self.out = np.vectorize(Value.relu)(inputs)
        return self.out


class Sigmoid(_Layer):
    """Layer that applies the sigmoid activation function element-wise to the input data."""

    def __call__(self, inputs: np.ndarray):
        self.out = np.vectorize(Value.sigmoid)(inputs)
        return self.out


class Softmax(_Layer):
    """Layer that applies the softmax activation function to the input data"""

    def __call__(self, inputs: np.ndarray):
        self.out = np.exp(inputs) / np.sum(np.exp(inputs), axis=1, keepdims=True)
        return self.out


class LogSoftmax(_Layer):
    """A layer that applies the log-softmax activation function to the input data"""

    def __call__(self, inputs: np.ndarray):
        self.out = np.log(np.exp(inputs) / np.sum(np.exp(inputs), axis=1, keepdims=True))
        return self.out


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def __call__(self, inputs: np.ndarray):
        if self.training:
            xmean = np.mean(inputs, axis=0, keepdims=True)
            xvar = np.var(inputs, axis=0, keepdims=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (inputs - xmean) / np.sqrt(xvar + self.eps)  # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with np.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
