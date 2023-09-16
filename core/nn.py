from typing import List
import numpy as np
from core.engine import Value


class Neuron:
    def __init__(self, num_inputs: int):
        self.weights = [Value(np.random.normal()) for _ in range(num_inputs)]
        self.bias = Value(np.random.normal())
        self.parameters = self.weights + [self.bias]

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        out = act.tanh()
        return out


class Layer:
    def __init__(self, num_inputs: int, num_outputs: int):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]
        self.parameters = [p for neuron in self.neurons for p in neuron.parameters]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def __repr__(self):
        return f"Layer with {len(self.neurons)} neurons"


class NN:
    def __init__(self, num_inputs: int, num_outputs: List[int]):
        self.layer_sizes = [num_inputs] + num_outputs
        self.layers = [Layer(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(num_outputs))]
        self.parameters = [p for layer in self.layers for p in layer.parameters]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return str(self.layers)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0.0
