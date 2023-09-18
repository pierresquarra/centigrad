import torch
from unittest import TestCase
from centigrad.engine import Value


class TestValue(TestCase):
    def test_value(self):
        def calculate(x):
            z = 2 * x + 2 + x
            q = z.relu() + z * x
            h = (z * z).relu()
            y = h + q + q * x
            y.backward()
            return x, y

        x = Value(-4.0)
        my_x, my_y = calculate(x)

        x = torch.Tensor([-4.0]).double()
        x.requires_grad = True
        pt_x, pt_y = calculate(x)

        # forward pass went well
        assert my_y.data == pt_y.data.item()
        # backward pass went well
        assert my_x.grad == pt_x.grad.item()

    def test_more_ops(self):
        a = Value(-4.0)
        b = Value(2.0)
        c = a + b
        d = a * b + b ** 3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e ** 2
        g = f / 2.0
        g += 10.0 / f
        g.backward()
        amg, bmg, gmg = a, b, g

        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([2.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = a + b
        d = a * b + b ** 3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e ** 2
        g = f / 2.0
        g = g + 10.0 / f
        g.backward()
        apt, bpt, gpt = a, b, g

        tol = 1e-6
        # forward pass went well
        assert abs(gmg.data - gpt.data.item()) < tol
        # backward pass went well
        assert abs(amg.grad - apt.grad.item()) < tol
        assert abs(bmg.grad - bpt.grad.item()) < tol
