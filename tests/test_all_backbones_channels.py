import unittest
import torch
from omegaconf import OmegaConf
from src.models.unet_model import UNetPalmModel

class TestAllBackbonesChannels(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.load("config/config.yaml")
        
        # Initialize model config manually since OmegaConf.load doesn't resolve Hydra defaults
        self.config.model = OmegaConf.create({})
        
        # Ensure we have a decoder config
        self.config.model.decoder = OmegaConf.create({
            "use_decoder": True,
            "skip_dropout": 0.5,
            "image_size": [128, 128],
            "unet_channels": [16, 32, 64, 128] # Make it small for faster tests
        })
        self.config.model.projector = OmegaConf.create({
            "use_mlp": True,
            "hidden_dims": [64],
            "proj_dim": 32
        })

    def run_backbone_with_channels(self, backbone_name, in_channels, image_size=64):
        self.config.model.encoder = OmegaConf.create({
            "backbone": backbone_name,
            "in_channels": in_channels,
            "latent_dim": 64
        })
        
        self.config.model.decoder.image_size = [image_size, image_size]
        
        model = UNetPalmModel(self.config.model)
        model.eval()
        
        batch_size = 2
        dummy_input = torch.randn(batch_size, in_channels, image_size, image_size)
        
        with torch.no_grad():
            outputs = model(dummy_input, decode=True)
            
        self.assertIn('mu', outputs)
        self.assertIn('logvar', outputs)
        self.assertIn('x_hat', outputs)
        self.assertEqual(outputs['mu'].shape, (batch_size, 64))
        self.assertEqual(outputs['x_hat'].shape, (batch_size, in_channels, image_size, image_size))

    def test_mock_backbone(self):
        self.run_backbone_with_channels('mock', in_channels=1)
        self.run_backbone_with_channels('mock', in_channels=3)

    def test_cnn_backbone(self):
        self.run_backbone_with_channels('cnn', in_channels=1)
        self.run_backbone_with_channels('cnn', in_channels=3)

    def test_ccnet_backbone(self):
        self.run_backbone_with_channels('ccnet', in_channels=1)
        self.run_backbone_with_channels('ccnet', in_channels=4)

    def test_palmnet_backbone(self):
        self.run_backbone_with_channels('palmnet', in_channels=1)
        self.run_backbone_with_channels('palmnet', in_channels=3)

    def test_resnet_backbone(self):
        # resnet18 has conv1
        self.run_backbone_with_channels('resnet18', in_channels=1, image_size=128)
        self.run_backbone_with_channels('resnet18', in_channels=4, image_size=128)

    # def test_vgg_backbone(self):
    #     # vgg11 has features[0]
    #     self.run_backbone_with_channels('vgg11', in_channels=1, image_size=128)
    #     self.run_backbone_with_channels('vgg11', in_channels=2, image_size=128)

if __name__ == '__main__':
    unittest.main()
