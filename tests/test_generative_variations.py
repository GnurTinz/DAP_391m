import unittest
import torch
import os
import sys

# Thêm đường dẫn project_root vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet_model import UNetPalmModel

class TestGenerativeVariations(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_config = {
            'decoder': {
                'image_size': [128, 128],
                'use_decoder': True,
                'unet_channels': [16, 32, 64, 128] # Dùng bản nhẹ cho test chạy nhanh
            },
            'encoder': {
                'latent_dim': 64,
                'backbone': 'mock'
            },
            'projector': {
                'use_mlp': True,
                'hidden_dims': [64],
                'activation': 'ReLU',
                'proj_dim': 32
            }
        }
        
    def test_sampling_modes(self):
        """Test các chế độ sinh mẫu: stochastic, deterministic, symmetric"""
        model = UNetPalmModel(self.base_config).to(self.device)
        model.eval()
        
        x = torch.randn(2, 3, 128, 128).to(self.device)
        
        with torch.no_grad():
            # 1. Deterministic mode (z phải bằng mu)
            out_det = model(x, decode=False, sample_mode='deterministic')
            self.assertTrue(torch.allclose(out_det['z'], out_det['mu']), 
                            "Deterministic mode: z phải chính xác bằng mu")
            
            # 2. Stochastic mode
            out_stoc = model(x, decode=False, sample_mode='stochastic')
            self.assertFalse(torch.allclose(out_stoc['z'], out_stoc['mu']), 
                             "Stochastic mode: z không được giống hệt mu do có nhiễu epsilon")
            self.assertEqual(out_stoc['z'].shape, out_stoc['mu'].shape)
            
            # 3. Symmetric mode
            out_sym = model(x, decode=False, sample_mode='symmetric')
            self.assertEqual(out_sym['z'].shape, out_sym['mu'].shape)
            
        print("✔ Sinh mẫu (Sampling Modes) hoạt động chính xác!")

    def test_projector_mlp_variations(self):
        """Test biến thể Projector (bật/tắt MLP, đổi Activation)"""
        # 1. Dùng Identity (Tắt MLP)
        config_no_mlp = dict(self.base_config)
        config_no_mlp['projector'] = {'use_mlp': False}
        model_identity = UNetPalmModel(config_no_mlp).to(self.device)
        
        x = torch.randn(2, 3, 128, 128).to(self.device)
        out1 = model_identity(x, decode=False)
        self.assertTrue(torch.allclose(out1['proj'], out1['mu']), 
                        "Khi use_mlp=False, proj phải bằng mu (Identity)")
        
        # 2. Dùng MLP với LeakyReLU
        config_mlp = dict(self.base_config)
        config_mlp['projector'] = {
            'use_mlp': True,
            'hidden_dims': [128],
            'activation': 'LeakyReLU',
            'proj_dim': 32
        }
        model_mlp = UNetPalmModel(config_mlp).to(self.device)
        out2 = model_mlp(x, decode=False)
        self.assertEqual(out2['proj'].shape, (2, 32), 
                         "Khi use_mlp=True, chiều của proj phải bằng proj_dim")
                         
        print("✔ Biến thể Projector hoạt động chính xác!")

    def test_decoder_flag_and_film(self):
        """Test ngắt mạch Decoder để tối ưu lúc Inference (chỉ lấy features) và luồng FiLM"""
        model = UNetPalmModel(self.base_config).to(self.device)
        x = torch.randn(2, 3, 128, 128).to(self.device)
        
        # 1. decode=False (Không chạy nhánh giải mã)
        out_no_decode = model(x, decode=False)
        self.assertNotIn('x_hat', out_no_decode, "decode=False không được xuất x_hat")
        self.assertIn('z', out_no_decode)
        
        # 2. decode=True (Chạy qua nhánh FiLM và Decoder)
        out_decode = model(x, decode=True)
        self.assertIn('x_hat', out_decode, "decode=True phải sinh ra x_hat")
        self.assertEqual(out_decode['x_hat'].shape, (2, 3, 128, 128), "Kích thước ảnh tái tạo bị sai")
        
        print("✔ Ngắt mạch Decoder & FiLM skip connections hoạt động chính xác!")

if __name__ == '__main__':
    unittest.main()
