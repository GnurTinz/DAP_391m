import unittest
import torch
import sys
import os

# Đảm bảo có thể import thư mục src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet_model import UNetPalmModel

class TestProbabilisticUNet(unittest.TestCase):
    def setUp(self):
        """Khởi tạo cấu hình và model trước mỗi bài test"""
        self.config = {
            'dataset': {
                'image_size': [64, 64] # Dùng size nhỏ để test nhanh
            },
            'encoder': {
                'backbone': 'mock', 
                'latent_dim': 128,
                'pretrained': False
            },
            'decoder': {
                'use_decoder': True,
                'image_size': [64, 64],
                'skip_dropout': 0.2
            },
            'projector': {
                'proj_dim': 64
            }
        }
        self.model = UNetPalmModel(self.config)
        self.model.eval()

    def test_forward_pass_shapes(self):
        """Kiểm tra xem forward pass có trả về đúng kích thước các tensor hay không"""
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 64, 64)
        
        with torch.no_grad():
            output = self.model(dummy_input, decode=True, temperature=1.0)
            
        # Kiểm tra keys trong output
        self.assertIn('mu', output)
        self.assertIn('logvar', output)
        self.assertIn('z', output)
        self.assertIn('x_hat', output)
        self.assertIn('proj', output)
        
        # Kiểm tra shape
        self.assertEqual(output['mu'].shape, (batch_size, 128))
        self.assertEqual(output['logvar'].shape, (batch_size, 128))
        self.assertEqual(output['z'].shape, (batch_size, 128))
        self.assertEqual(output['x_hat'].shape, (batch_size, 3, 64, 64))
        self.assertEqual(output['proj'].shape, (batch_size, 64))

    def test_no_nan_values(self):
        """Kiểm tra xem output (đặc biệt sau khi qua FiLM) có bị dính NaN hay không"""
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 64, 64)
        
        with torch.no_grad():
            output = self.model(dummy_input, decode=True)
            
        self.assertFalse(torch.isnan(output['x_hat']).any(), "Lỗi: x_hat sinh ra chứa giá trị NaN!")
        self.assertFalse(torch.isnan(output['mu']).any(), "Lỗi: mu chứa giá trị NaN!")

    def test_temperature_scaling(self):
        """Kiểm tra tham số temperature có thực sự làm thay đổi z hay không"""
        batch_size = 1
        dummy_input = torch.randn(batch_size, 3, 64, 64)
        
        with torch.no_grad():
            out1 = self.model(dummy_input, decode=False, temperature=1.0)
            out2 = self.model(dummy_input, decode=False, temperature=10.0)
            
        # Khi temperature thay đổi (và z được sample ngẫu nhiên), z1 và z2 phải khác nhau
        # Mặc dù mu và logvar phải giống hệt nhau (do cùng dummy_input)
        self.assertTrue(torch.allclose(out1['mu'], out2['mu']), "mu phải giống nhau với cùng input")
        self.assertFalse(torch.allclose(out1['z'], out2['z']), "z phải khác nhau do nhiễu và temperature")

    def test_custom_cnn_latent_encoder(self):
        """Kiểm tra mạng Latent Encoder sinh z với cấu trúc CNN động qua config"""
        custom_config = {
            'dataset': {
                'image_size': [32, 32]
            },
            'encoder': {
                'backbone': 'cnn',
                'hidden_dims': [8, 16, 32], # Tự do thay đổi số lớp
                'latent_dim': 64,
                'pretrained': False
            },
            'decoder': {
                'use_decoder': True,
                'image_size': [32, 32],
                'skip_dropout': 0.2
            },
            'projector': {
                'proj_dim': 32
            }
        }
        custom_model = UNetPalmModel(custom_config)
        custom_model.eval()
        
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 32, 32)
        
        with torch.no_grad():
            output = custom_model(dummy_input, decode=True)
            
        self.assertEqual(output['z'].shape, (batch_size, 64), "Latent dim không khớp với custom config")
        self.assertEqual(output['x_hat'].shape, (batch_size, 3, 32, 32), "Decode bị lỗi với custom CNN")

if __name__ == '__main__':
    unittest.main()
