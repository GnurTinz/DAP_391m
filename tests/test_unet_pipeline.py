import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet_model import UNetPalmModel

class TestUNetPipelineSettings(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = [64, 64]
        self.latent_dim = 128
        self.dummy_input = torch.randn(self.batch_size, 3, *self.image_size)

    def get_base_config(self, use_decoder=True, skip_dropout=0.2):
        return {
            'dataset': {
                'image_size': self.image_size
            },
            'encoder': {
                'backbone': 'mock', 
                'latent_dim': self.latent_dim
            },
            'decoder': {
                'use_decoder': use_decoder,
                'image_size': self.image_size,
                'skip_dropout': skip_dropout
            },
            'projector': {
                'use_mlp': True,
                'hidden_dims': [128],
                'activation': 'ReLU',
                'proj_dim': 64
            }
        }

    def test_unet_with_decoder(self):
        config = self.get_base_config(use_decoder=True, skip_dropout=0.5)
        model = UNetPalmModel(config)
        model.eval()

        with torch.no_grad():
            out = model(self.dummy_input, decode=True)

        self.assertIn('mu', out)
        self.assertIn('logvar', out)
        self.assertIn('z', out)
        self.assertIn('proj', out)
        self.assertIn('x_hat', out)

        self.assertEqual(out['x_hat'].shape, (self.batch_size, 3, *self.image_size))
        self.assertEqual(out['z'].shape, (self.batch_size, self.latent_dim))

    def test_unet_without_decoder(self):
        config = self.get_base_config(use_decoder=False)
        model = UNetPalmModel(config)
        model.eval()

        with torch.no_grad():
            # Even if we pass decode=True, if use_decoder=False in config, 
            # it shouldn't produce x_hat according to current implementation
            out = model(self.dummy_input, decode=True)

        self.assertNotIn('x_hat', out)
        self.assertIn('mu', out)
        self.assertIn('z', out)

    def test_unet_sampling_modes(self):
        config = self.get_base_config(use_decoder=True)
        model = UNetPalmModel(config)
        model.eval()

        modes = ['stochastic', 'deterministic', 'symmetric']
        
        for mode in modes:
            with torch.no_grad():
                out = model(self.dummy_input, decode=False, sample_mode=mode, temperature=0.5)
            self.assertIn('z', out, f"Failed to generate z in mode: {mode}")
            self.assertEqual(out['z'].shape, (self.batch_size, self.latent_dim))

            if mode == 'deterministic':
                # In deterministic mode, z should be equal to mu
                self.assertTrue(torch.allclose(out['z'], out['mu']), "In deterministic mode, z should equal mu")

    def test_unet_forward_decode_false(self):
        config = self.get_base_config(use_decoder=True)
        model = UNetPalmModel(config)
        model.eval()

        with torch.no_grad():
            out = model(self.dummy_input, decode=False)
        
        self.assertNotIn('x_hat', out, "x_hat should not be generated when decode=False")
        self.assertIn('z', out)

    def test_gradient_flow(self):
        config = self.get_base_config(use_decoder=True)
        model = UNetPalmModel(config)
        
        # Đảm bảo đang ở chế độ train để tính toán gradient
        model.train()
        
        # Rút trích một vài tham số đại diện từ các khối mạng khác nhau
        # 1. Latent encoder
        latent_param = next(model.latent_encoder.parameters())
        # 2. Projector
        proj_param = next(model.projector.parameters())
        # 3. U-Net Encoder (Deterministic Skip Connections)
        enc_param = next(model.inc.parameters())
        # 4. FiLM modulation layer
        film_param = next(model.film_gamma3.parameters())
        # 5. U-Net Decoder
        dec_param = next(model.up1.parameters())

        # Forward pass
        out = model(self.dummy_input, decode=True)
        
        # Tạo hàm loss giả lập: Tổng của x_hat (tái tạo), proj (contrastive) và mu/logvar (KL)
        dummy_loss = out['x_hat'].mean() + out['proj'].mean() + out['mu'].mean() + out['logvar'].mean()
        
        # Clear gradient cũ (nếu có)
        model.zero_grad()
        
        # Backward pass
        dummy_loss.backward()
        
        # Kiểm tra xem gradient có đi qua toàn bộ pipeline và không bị đứt gãy không
        components = {
            "Latent Encoder": latent_param,
            "Projector": proj_param,
            "U-Net Encoder (Skips)": enc_param,
            "FiLM Layer": film_param,
            "U-Net Decoder": dec_param
        }
        
        for name, param in components.items():
            self.assertIsNotNone(param.grad, f"Gradient bị mất (None) tại {name}")
            
            grad_norm = param.grad.norm().item()
            self.assertGreater(grad_norm, 0.0, f"Gradient bằng 0 (Vanishing Gradient) tại {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"Gradient bị NaN tại {name}")

if __name__ == '__main__':
    unittest.main()
