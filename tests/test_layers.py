from unittest import TestCase

import numpy as np
import torch
from torch import nn

from centigrad.engine import Value
from centigrad.layers import Linear, Tanh, ReLU, Softmax, Sigmoid, LogSoftmax, BatchNorm1d


class TestLinear(TestCase):
    def test_with_bias(self):
        pt = nn.Linear(20, 30)

        pt_input = torch.randn(128, 20)
        pt_output = pt(pt_input)

        cg = Linear(20, 30)
        cg.weight = pt.weight.detach().numpy()
        cg.bias = pt.bias.detach().numpy()

        cg_input = pt_input.detach().numpy()
        cg_output = torch.tensor(cg(cg_input))

        self.assertEqual(pt_output.shape, cg_output.shape)
        self.assertTrue(torch.allclose(pt_output, cg_output, rtol=1e-3))

    def test_without_bias(self):
        pt = nn.Linear(20, 30, bias=False)

        pt_input = torch.randn(128, 20)
        pt_output = pt(pt_input)

        cg = Linear(20, 30, bias=False)
        cg.weight = pt.weight.detach().numpy()

        cg_input = pt_input.detach().numpy()
        cg_output = torch.tensor(cg(cg_input))

        self.assertEqual(pt_output.shape, cg_output.shape)
        self.assertTrue(torch.allclose(pt_output, cg_output, rtol=1e-3))


class TestActivations(TestCase):
    def setUp(self) -> None:
        self.pt_in = torch.randn(5, 12)
        self.cg_in = np.vectorize(Value)(self.pt_in.detach().numpy())

    def test_tanh(self):
        pt = nn.Tanh()
        pt_out = pt(self.pt_in)
        cg = Tanh()
        cg_out = cg(self.cg_in)
        cg_ten = torch.tensor(np.vectorize(lambda x: x.data)(cg_out)).float()
        self.assertTrue(torch.allclose(pt_out, cg_ten, rtol=1e-3))

    def test_relu(self):
        pt = nn.ReLU()
        pt_out = pt(self.pt_in)
        cg = ReLU()
        cg_out = cg(self.cg_in)
        cg_ten = torch.tensor(np.vectorize(lambda x: x.data)(cg_out)).float()
        self.assertTrue(torch.allclose(pt_out, cg_ten, rtol=1e-3))

    def test_sigmoid(self):
        pt = nn.Sigmoid()
        pt_out = pt(self.pt_in)
        cg = Sigmoid()
        cg_out = cg(self.cg_in)
        cg_ten = torch.tensor(np.vectorize(lambda x: x.data)(cg_out)).float()
        self.assertTrue(torch.allclose(pt_out, cg_ten, rtol=1e-3))

    def test_softmax(self):
        pt = nn.Softmax(dim=1)
        pt_out = pt(self.pt_in)
        cg = Softmax()
        cg_out = cg(self.cg_in)
        cg_ten = torch.tensor(np.vectorize(lambda x: x.data)(cg_out)).float()
        self.assertTrue(torch.allclose(pt_out, cg_ten, rtol=1e-3))

    def test_log_softmax(self):
        pt = nn.LogSoftmax(dim=1)
        pt_out = pt(self.pt_in)
        cg = LogSoftmax()
        cg_out = cg(self.cg_in)
        cg_ten = torch.tensor(np.vectorize(lambda x: x.data)(cg_out)).float()
        self.assertTrue(torch.allclose(pt_out, cg_ten, rtol=1e-3))

    def test_batch_norm_1d(self):
        pt = nn.BatchNorm1d(12)
        pt_out = pt(self.pt_in)
        cg = BatchNorm1d(12)
        cg_out = cg(self.cg_in)
        cg_ten = torch.tensor(np.vectorize(lambda x: x.data)(cg_out)).float()
        self.assertTrue(torch.allclose(pt_out, cg_ten, rtol=1e-3))
