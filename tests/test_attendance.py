import unittest
import torch
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.attendance import Attendant

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

class TestAttendanceLogic(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MockModel().to(self.device)
        self.model.eval()

    def test_attendance_logic(self):
        config = {'testing': {'use_raw_mu': False, 'optimize_probe': False}}
        
        # Create a mock gallery
        gallery = {
            1: torch.ones(2, 128) / (128 ** 0.5), # Normalized vector for ID 1
            2: -torch.ones(2, 128) / (128 ** 0.5) # Normalized vector for ID 2
        }
        
        attendant = Attendant(self.model, gallery, self.device, config)
        
        # Test 1 image
        test_img = torch.randn(1, 1, 128, 128)
        best_id, best_score, id_scores, top5_ids = attendant.attend_image(test_img)
        
        # The test image will produce mu=0.5 -> projector=1.0 -> normalized = 1/sqrt(128)
        # Cosine sim with ID 1 should be exactly 1.0
        # Cosine sim with ID 2 should be exactly -1.0
        
        self.assertEqual(best_id, 1)
        self.assertAlmostEqual(best_score, 1.0, places=4)
        self.assertAlmostEqual(id_scores[2], -1.0, places=4)

if __name__ == '__main__':
    unittest.main()
