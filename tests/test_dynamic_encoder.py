import unittest
import torch
from src.models.encoder import PalmEncoder

class TestDynamicEncoder(unittest.TestCase):
    def test_mock_encoder(self):
        config = {'backbone': 'mock', 'latent_dim': 128}
        encoder = PalmEncoder(config)
        x = torch.randn(2, 3, 32, 32)
        mu, logvar = encoder(x)
        self.assertEqual(mu.shape, (2, 128))
        self.assertEqual(logvar.shape, (2, 128))

    def test_resnet_encoder(self):
        config = {'backbone': 'resnet18', 'latent_dim': 64, 'pretrained': False}
        encoder = PalmEncoder(config)
        x = torch.randn(2, 3, 64, 64)
        mu, logvar = encoder(x)
        self.assertEqual(mu.shape, (2, 64))
        self.assertEqual(logvar.shape, (2, 64))

    def test_mobilenet_encoder(self):
        config = {'backbone': 'mobilenet_v2', 'latent_dim': 64, 'pretrained': False}
        encoder = PalmEncoder(config)
        x = torch.randn(2, 3, 64, 64)
        mu, logvar = encoder(x)
        self.assertEqual(mu.shape, (2, 64))
        self.assertEqual(logvar.shape, (2, 64))

if __name__ == '__main__':
    unittest.main()
