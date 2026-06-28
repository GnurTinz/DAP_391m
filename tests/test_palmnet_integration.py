import os
import sys
import torch
import unittest
from omegaconf import OmegaConf

# Đảm bảo import được code trong src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet_model import UNetPalmModel

class TestPalmNetIntegration(unittest.TestCase):
    def setUp(self):
        # Tải cấu hình chung và ghi đè phần model bằng unet_palmnet_gabor
        self.config = OmegaConf.load("config/config.yaml")
        palmnet_config = OmegaConf.load("config/model/unet_palmnet_gabor.yaml")
        self.config.model = palmnet_config
        
        # Bắt buộc khai báo image_size vì kiến trúc U-Net cần nội suy
        self.config.dataset = OmegaConf.create({"image_size": 128})
        
        self.model = UNetPalmModel(self.config.model)
        self.model.eval()

    def test_forward_pass_with_palmnet(self):
        """
        Kiểm tra xem dữ liệu có chảy mượt mà qua PalmNet Backbone,
        sinh ra mu, logvar hợp lệ và U-Net Decoder có tái tạo được ảnh không.
        """
        batch_size = 2
        channels = 3
        image_size = 128
        
        # Tạo dữ liệu giả lập
        dummy_input = torch.randn(batch_size, channels, image_size, image_size)
        
        with torch.no_grad():
            outputs = self.model(dummy_input, decode=True)
            
        # 1. Kiểm tra không gian Latent (mu, logvar)
        self.assertIn('mu', outputs)
        self.assertIn('logvar', outputs)
        
        expected_latent_dim = self.config.model.encoder.latent_dim
        self.assertEqual(outputs['mu'].shape, (batch_size, expected_latent_dim))
        self.assertEqual(outputs['logvar'].shape, (batch_size, expected_latent_dim))
        
        # 2. Kiểm tra khối giải mã U-Net (x_hat)
        self.assertIn('x_hat', outputs)
        self.assertEqual(outputs['x_hat'].shape, (batch_size, channels, image_size, image_size))
        
        # 3. Kiểm tra khối Projector (nếu có sử dụng ArcFace / Contrastive)
        p = self.model.projector(outputs['mu'])
        expected_proj_dim = self.config.model.projector.proj_dim
        self.assertEqual(p.shape, (batch_size, expected_proj_dim))

    def test_forward_pass_with_grayscale_channels(self):
        """
        Kiểm tra mô hình có hỗ trợ linh hoạt số lượng channels đầu vào không (ví dụ in_channels = 1).
        """
        import copy
        # Clone config để không ảnh hưởng test khác
        custom_config = copy.deepcopy(self.config)
        custom_config.model.encoder.in_channels = 1
        
        # Khởi tạo model với config in_channels=1
        model_1_channel = UNetPalmModel(custom_config.model)
        model_1_channel.eval()
        
        batch_size = 2
        channels = 1
        image_size = 128
        
        # Tạo dữ liệu giả lập (ảnh grayscale)
        dummy_input = torch.randn(batch_size, channels, image_size, image_size)
        
        with torch.no_grad():
            outputs = model_1_channel(dummy_input, decode=True)
            
        self.assertIn('mu', outputs)
        self.assertIn('x_hat', outputs)
        # Đảm bảo output x_hat cũng là 1 channel
        self.assertEqual(outputs['x_hat'].shape, (batch_size, channels, image_size, image_size))

if __name__ == '__main__':
    unittest.main()
