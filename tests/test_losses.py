import numpy as np
import unittest
from core.losses.mse import MSE

class TestMSE(unittest.TestCase):
    def setUp(self):
        self.predictions = np.random.rand(10)
        self.targets = np.random.rand(10)
        self.loss_function = MSE()

    def test_forward(self):
        self.assertEqual(self.loss_function.forward(self.predictions, self.predictions), 0.0)

if __name__ == "__main__":
    unittest.main()