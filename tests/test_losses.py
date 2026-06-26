import sys
import os
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.losses.custom import get_contrastive_loss

class TestContrastiveLosses(unittest.TestCase):
    def setUp(self):
        # Fake data: 4 samples, 2 classes (0, 0, 1, 1), embedding dim 64
        self.features = torch.randn(4, 64, requires_grad=True)
        self.labels = torch.tensor([0, 0, 1, 1])
        
    def _check_loss(self, loss_type):
        config = {'contrastive_type': loss_type}
        criterion = get_contrastive_loss(config)
        
        # We need to re-initialize features so gradients don't accumulate across tests
        features = self.features.clone().detach().requires_grad_(True)
        
        loss = criterion(features, self.labels)
        
        # Loss must be a scalar
        self.assertEqual(loss.dim(), 0, f"{loss_type} should return a scalar.")
        self.assertFalse(torch.isnan(loss).any(), f"{loss_type} returned NaN.")
        
        loss.backward()
        
        self.assertIsNotNone(features.grad, f"No gradients computed for {loss_type}.")
        self.assertFalse(torch.isnan(features.grad).any(), f"Gradients contain NaNs for {loss_type}.")
        
    def test_supcon_loss(self):
        self._check_loss('supcon')
        
    def test_ms_loss(self):
        self._check_loss('ms_loss')
        
    def test_infonce_loss(self):
        self._check_loss('infonce')

if __name__ == '__main__':
    unittest.main()
