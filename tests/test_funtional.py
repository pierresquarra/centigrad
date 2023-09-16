from unittest import TestCase

from core.nn import NN
from core.functional import softmax


class TestFunctions(TestCase):
    def test_softmax(self):
        n = NN(3, [10])
        x = [2.0, 3.0, -1.0]
        res = n(x)
        sm = softmax(res)
        sm_sum = sum([value.data for value in sm])
        self.assertAlmostEqual(1, sm_sum)
