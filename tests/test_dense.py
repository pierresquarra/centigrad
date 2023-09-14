import numpy as np
import unittest
from core.layers.dense import Dense

class TestDense(unittest.TestCase):
    def setUp(self):
        self.input_dim = np.random.randint(10)
        self.output_dim = np.random.randint(10)
        self.layer = Dense(self.input_dim, self.output_dim)

    def test_init(self):
        self.assertEqual(self.layer.weights.shape, (self.input_dim, self.output_dim))
        self.assertEqual(self.layer.biases.shape, (self.output_dim,))

    def test_forward(self):
        self.inputs = np.random.rand(self.input_dim)
        self.outputs = self.layer.forward(self.inputs)
        self.assertEqual(self.outputs.shape, (self.output_dim,))

if __name__ == "__main__":
    unittest.main()