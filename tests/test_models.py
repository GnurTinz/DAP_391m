import unittest
import torch
from src.models.encoder import PalmEncoder
from src.models.decoder import PalmDecoder
from src.models.verifier import PairVerifier
from src.models.palm_model import ProbabilisticPalmModel

class TestPalmModels(unittest.TestCase):
    def setUp(self):
        # Thiết lập config giả lập cho các module
        self.latent_dim = 128
        self.batch_size = 4
        
        self.encoder_config = {
            'backbone': 'mock', # Dùng mock backbone để test chạy nhanh
            'pretrained': False,
            'latent_dim': self.latent_dim
        }
        
        self.decoder_config = {
            'latent_dim': self.latent_dim
        }
        
        self.verifier_config = {
            'latent_dim': self.latent_dim,
            'hidden_dims': [64, 32]
        }
        
        self.model_config = {
            'encoder': self.encoder_config,
            'decoder': self.decoder_config,
            'verifier': self.verifier_config
        }
        
        # Tạo dữ liệu giả lập (mock data) với size 128x128 khớp với decoder và dataset
        self.dummy_input = torch.randn(self.batch_size, 3, 128, 128)

    def test_encoder(self):
        encoder = PalmEncoder(self.encoder_config)
        mu, logvar = encoder(self.dummy_input)
        
        # Kiểm tra shape của mu và logvar
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim), "Shape của mu không khớp!")
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim), "Shape của logvar không khớp!")
        
    def test_decoder(self):
        decoder = PalmDecoder(self.decoder_config)
        dummy_z = torch.randn(self.batch_size, self.latent_dim)
        
        # Decoder trong code yêu cầu z được reshape thành 256x8x8 trước deconv (16384 param). 
        # Cấu trúc Linear: fc = Linear(latent_dim, 256 * 8 * 8)
        # Đầu ra của decoder là 3 channels
        x_hat = decoder(dummy_z)
        
        self.assertEqual(x_hat.shape, (self.batch_size, 3, 128, 128), "Shape ảnh được tái tạo không khớp!")

    def test_verifier(self):
        verifier = PairVerifier(self.verifier_config)
        
        # Tạo 2 cặp mu, logvar giả lập
        mu1 = torch.randn(self.batch_size, self.latent_dim)
        logvar1 = torch.randn(self.batch_size, self.latent_dim)
        
        mu2 = torch.randn(self.batch_size, self.latent_dim)
        logvar2 = torch.randn(self.batch_size, self.latent_dim)
        
        score = verifier(mu1, logvar1, mu2, logvar2)
        
        # Output shape của MLP verifier là (batch_size, 1) vì linear cuối cùng là 1
        self.assertEqual(score.shape, (self.batch_size, 1), "Shape của điểm số verification không khớp!")

    def test_probabilistic_palm_model_forward(self):
        model = ProbabilisticPalmModel(self.model_config)
        
        # Chạy thử với cờ decode=True
        out = model(self.dummy_input, decode=True)
        
        # Kiểm tra có đủ các key cần thiết
        self.assertIn('mu', out)
        self.assertIn('logvar', out)
        self.assertIn('z', out)
        self.assertIn('x_hat', out)
        
        # Kiểm tra shape
        self.assertEqual(out['mu'].shape, (self.batch_size, self.latent_dim))
        self.assertEqual(out['logvar'].shape, (self.batch_size, self.latent_dim))
        self.assertEqual(out['z'].shape, (self.batch_size, self.latent_dim))
        self.assertEqual(out['x_hat'].shape, (self.batch_size, 3, 128, 128))

    def test_probabilistic_palm_model_verify(self):
        model = ProbabilisticPalmModel(self.model_config)
        
        mu1, logvar1 = torch.randn(self.batch_size, self.latent_dim), torch.randn(self.batch_size, self.latent_dim)
        mu2, logvar2 = torch.randn(self.batch_size, self.latent_dim), torch.randn(self.batch_size, self.latent_dim)
        
        score = model.verify(mu1, logvar1, mu2, logvar2)
        self.assertEqual(score.shape, (self.batch_size, 1))

if __name__ == '__main__':
    unittest.main()
