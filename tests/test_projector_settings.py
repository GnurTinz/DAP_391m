import unittest
import torch
import sys
import os

# Ensure the root project is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel

class TestProjectorSettings(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = [64, 64]
        self.dummy_input = torch.randn(self.batch_size, 3, *self.image_size)

    def get_base_config(self, use_decoder=False):
        return {
            'dataset': {
                'image_size': self.image_size
            },
            'encoder': {
                'backbone': 'mock', # Using mock to avoid loading large models in tests
                'latent_dim': 128
            },
            'decoder': {
                'use_decoder': use_decoder,
                'image_size': self.image_size,
                'skip_dropout': 0.0
            }
        }

    def _test_projector(self, model_class, use_decoder):
        settings = [
            # 1. No MLP (Identity)
            {
                'use_mlp': False,
                'proj_dim': 64 # Should be ignored and output dim should match latent_dim (128)
            },
            # 2. MLP with default activation (ReLU) and no hidden dims
            {
                'use_mlp': True,
                'proj_dim': 64,
                'hidden_dims': []
            },
            # 3. MLP with LeakyReLU and hidden dims
            {
                'use_mlp': True,
                'proj_dim': 64,
                'hidden_dims': [128, 64],
                'activation': 'LeakyReLU'
            },
            # 4. MLP with GELU
            {
                'use_mlp': True,
                'proj_dim': 32,
                'hidden_dims': [64],
                'activation': 'GELU'
            }
        ]
        
        for proj_config in settings:
            config = self.get_base_config(use_decoder)
            config['projector'] = proj_config
            
            try:
                model = model_class(config)
                model.eval()
                
                with torch.no_grad():
                    out = model(self.dummy_input, decode=use_decoder)
                
                self.assertIn('proj', out)
                proj_out = out['proj']
                
                # Check expected output dimension
                expected_dim = 128 if not proj_config.get('use_mlp', True) else proj_config.get('proj_dim', 64)
                self.assertEqual(proj_out.shape, (self.batch_size, expected_dim), 
                                 f"Failed on {model_class.__name__} with config {proj_config}")
                                 
            except Exception as e:
                self.fail(f"{model_class.__name__} failed with config {proj_config}. Error: {e}")

    def test_palm_model_projector(self):
        self._test_projector(ProbabilisticPalmModel, use_decoder=False)
        self._test_projector(ProbabilisticPalmModel, use_decoder=True)

    def test_unet_model_projector(self):
        self._test_projector(UNetPalmModel, use_decoder=True)

if __name__ == '__main__':
    unittest.main()
