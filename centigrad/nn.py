import numpy as np


class Module:
    def __call__(self, inputs: np.ndarray):
        return self.forward(inputs)

    def forward(self, inputs: np.ndarray):
        raise NotImplementedError

    def parameters(self):
        return []
