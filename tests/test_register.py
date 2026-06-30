import unittest
import torch
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.register import Registerer

# Mock Model to avoid loading heavy weights during tests
class MockProjector(torch.nn.Module):
    def forward(self, x):
        return x * 2.0  # Dummy projection

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = MockProjector()

    def forward(self, x, decode=False):
        batch_size = x.size(0)
        mu = torch.ones(batch_size, 128) * 0.5
        logvar = torch.zeros(batch_size, 128)
        return {'mu': mu, 'logvar': logvar}

class TestRegisterLogic(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MockModel().to(self.device)
        self.model.eval()

        self.mock_images = torch.randn(3, 1, 128, 128) # 3 images of 1 person

    def test_register_raw_mu(self):
        config = {'testing': {'use_raw_mu': True, 'optimize_r': False}}
        registerer = Registerer(self.model, self.device, config)
        
        result = registerer.register_person(self.mock_images)
        self.assertEqual(result.shape, (3, 128))
        self.assertTrue(torch.allclose(result, torch.tensor(0.5)))

    def test_register_projector_no_opt(self):
        config = {'testing': {'use_raw_mu': False, 'optimize_r': False}}
        registerer = Registerer(self.model, self.device, config)
        
        result = registerer.register_person(self.mock_images)
        self.assertEqual(result.shape, (3, 128))
        
        expected_val = 1.0 / (128 ** 0.5)
        self.assertTrue(torch.allclose(result[0, 0], torch.tensor(expected_val)))

if __name__ == '__main__':
    unittest.main()
