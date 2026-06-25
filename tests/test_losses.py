import unittest
import torch
from src.losses.custom import KLDivLoss, UncertaintyLoss, ReconstructionLoss, SupConLoss

class TestCustomLosses(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.latent_dim = 128
        self.mu = torch.randn(self.batch_size, self.latent_dim)
        self.logvar = torch.randn(self.batch_size, self.latent_dim)
        
        self.img_shape = (self.batch_size, 3, 128, 128)
        self.x = torch.randn(*self.img_shape)
        self.x_hat = torch.randn(*self.img_shape)
        
    def test_kl_div_loss(self):
        criterion = KLDivLoss({})
        loss = criterion(self.mu, self.logvar)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0, "Loss phải là một scalar tensor")
        self.assertTrue(torch.isfinite(loss).all(), "Loss value bị nan/inf")

    def test_uncertainty_loss(self):
        criterion = UncertaintyLoss({})
        loss = criterion(self.logvar, lower_bound=-4.0, upper_bound=2.0)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0, "Loss phải là một scalar tensor")
        self.assertTrue(torch.isfinite(loss).all(), "Loss value bị nan/inf")
        self.assertTrue(loss.item() >= 0, "Penalty loss không được âm")

    def test_reconstruction_loss(self):
        criterion = ReconstructionLoss({})
        loss = criterion(self.x, self.x_hat)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0, "Loss phải là một scalar tensor")
        self.assertTrue(torch.isfinite(loss).all(), "Loss value bị nan/inf")
        self.assertTrue(loss.item() >= 0, "Reconstruction loss không được âm")

    def test_supcon_loss(self):
        criterion = SupConLoss({})
        features = torch.randn(self.batch_size, 64)
        labels = torch.randint(0, 10, (self.batch_size,))
        loss = criterion(features, labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0, "Loss phải là một scalar tensor")
        self.assertTrue(torch.isfinite(loss).all(), "Loss value bị nan/inf")

if __name__ == '__main__':
    unittest.main()
