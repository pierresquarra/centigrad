import torch
from torch import nn
from unittest import TestCase
from centigrad.losses import MSELoss, CrossEntropyLoss


class TestLosses(TestCase):
    def setUp(self) -> None:
        self.pt_input = torch.randn(3, 5)
        self.pt_target = torch.randn(3, 5)
        self.cg_input = self.pt_input.detach().numpy()
        self.cg_target = self.pt_target.detach().numpy()

    def test_mse_loss(self):
        pt_loss = nn.MSELoss()
        pt_output = pt_loss(self.pt_input, self.pt_target).item()

        cg_loss = MSELoss()
        cg_output = cg_loss(self.cg_input, self.cg_target)
        self.assertAlmostEqual(pt_output, cg_output, delta=1e-6)

    def test_cross_entropy_loss(self):
        pt_loss = nn.CrossEntropyLoss()
        pt_output = pt_loss(self.pt_input, self.pt_target).item()

        cg_loss = CrossEntropyLoss()
        cg_output = cg_loss(self.cg_input, self.cg_target)
        self.assertAlmostEqual(pt_output, cg_output, delta=1e-6)
