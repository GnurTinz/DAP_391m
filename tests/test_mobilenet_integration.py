import os
import sys
import torch
import unittest
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet_model import UNetPalmModel

class TestMobileNetIntegration(unittest.TestCase):
    def setUp(self):
        # Tải cấu hình chung và ghi đè phần model bằng unet_mobilenet
        self.config = OmegaConf.load("config/config.yaml")
        mobilenet_config = OmegaConf.load("config/model/unet_mobilenet.yaml")
        self.config.model = mobilenet_config
        
        # Bắt buộc khai báo image_size vì kiến trúc U-Net cần nội suy
        self.config.dataset = OmegaConf.create({"image_size": [128, 128]})
        
        self.model = UNetPalmModel(self.config.model)
        self.model.eval()

    def test_forward_pass_with_mobilenet(self):
        """
        Kiểm tra xem dữ liệu có chảy mượt mà qua MobileNet Backbone
        với in_channels = 3.
        """
        batch_size = 2
        channels = 3
        image_size = 128
        
        dummy_input = torch.randn(batch_size, channels, image_size, image_size)
        
        with torch.no_grad():
            outputs = self.model(dummy_input, decode=True)
            
        self.assertIn('mu', outputs)
        self.assertIn('logvar', outputs)
        self.assertIn('x_hat', outputs)
        
        expected_latent_dim = self.config.model.encoder.latent_dim
        self.assertEqual(outputs['mu'].shape, (batch_size, expected_latent_dim))
        self.assertEqual(outputs['x_hat'].shape, (batch_size, channels, image_size, image_size))
        
        p = self.model.projector(outputs['mu'])
        expected_proj_dim = self.config.model.projector.proj_dim
        self.assertEqual(p.shape, (batch_size, expected_proj_dim))

if __name__ == '__main__':
    unittest.main()
