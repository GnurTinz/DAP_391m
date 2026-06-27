import unittest
import torch
import os
import sys

# Thêm đường dẫn project_root vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet_model import UNetPalmModel

class TestUNetDynamicChannels(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_unet_default_channels(self):
        config = {
            'decoder': {
                'image_size': [128, 128],
                'use_decoder': True,
                # Không truyền unet_channels để test fallback mặc định (backward compatibility)
            },
            'encoder': {
                'latent_dim': 128,
                'backbone': 'mock'
            },
            'projector': {
                'use_mlp': False
            }
        }
        
        model = UNetPalmModel(config).to(self.device)
        # Verify if it falls back to 64, 128, 256, 512
        self.assertEqual(model.c1, 64)
        self.assertEqual(model.c4, 512)
        
        # Test forward pass with decoding
        x = torch.randn(2, 3, 128, 128).to(self.device)
        out = model(x, decode=True)
        
        self.assertIn('x_hat', out)
        self.assertEqual(out['x_hat'].shape, (2, 3, 128, 128))
        print("✔ U-Net Default Channels test passed!")

    def test_unet_lightweight_channels(self):
        config = {
            'decoder': {
                'image_size': [128, 128],
                'use_decoder': True,
                'unet_channels': [16, 32, 64, 128] # Cấu hình nhẹ
            },
            'encoder': {
                'latent_dim': 64,
                'backbone': 'mock'
            },
            'projector': {
                'use_mlp': False
            }
        }
        
        model = UNetPalmModel(config).to(self.device)
        # Verify dynamic binding
        self.assertEqual(model.c1, 16)
        self.assertEqual(model.c4, 128)
        
        # Test forward pass with decoding
        x = torch.randn(2, 3, 128, 128).to(self.device)
        out = model(x, decode=True)
        
        self.assertIn('x_hat', out)
        self.assertEqual(out['x_hat'].shape, (2, 3, 128, 128))
        print("✔ U-Net Lightweight Channels test passed!")
        
    def test_unet_custom_micro_channels(self):
        config = {
            'decoder': {
                'image_size': [64, 64],
                'use_decoder': True,
                'unet_channels': [8, 16, 32, 64] # Cấu hình siêu siêu nhỏ (micro)
            },
            'encoder': {
                'latent_dim': 32,
                'backbone': 'mock'
            },
            'projector': {
                'use_mlp': False
            }
        }
        
        model = UNetPalmModel(config).to(self.device)
        # Verify dynamic binding
        self.assertEqual(model.c1, 8)
        self.assertEqual(model.c4, 64)
        
        x = torch.randn(2, 3, 64, 64).to(self.device)
        out = model(x, decode=True)
        
        self.assertIn('x_hat', out)
        self.assertEqual(out['x_hat'].shape, (2, 3, 64, 64))
        print("✔ U-Net Custom Micro Channels test passed!")

if __name__ == '__main__':
    unittest.main()
