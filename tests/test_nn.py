from unittest import TestCase

from core.nn import NN


class TestNN(TestCase):
    def test_nn(self):
        x = [2.0, 3.0, -1.0]
        n = NN(3, [4, 4, 1])
        print(n.layer_sizes)
        print(len(n.layers))
        print(n.layers)
        print(n(x))

        xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
        ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

        for k in range(100):

            # forward pass
            ypred = [n(x) for x in xs]
            loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

            # backward pass
            n.zero_grad()
            loss.backward()

            # update
            for p in n.parameters:
                p.data += -0.1 * p.grad

            if k % 10 == 0:
                print(k, loss.data)
        print(ypred)


class TestNeuron(TestCase):
    pass
