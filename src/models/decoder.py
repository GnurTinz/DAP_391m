import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseDecoder(nn.Module):
    def __init__(self, latent_dim=256, image_size=(128, 128), hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Projection
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        
        # ConvTranspose blocks
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8)
        x = self.decoder(x)
        if list(x.shape[-2:]) != list(self.image_size):
            x = F.interpolate(x, size=tuple(self.image_size), mode='bilinear', align_corners=False)
        return x

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim=256, image_size=(128, 128), hidden_dims=None):
        super().__init__()
        self.image_size = image_size
        out_dim = 3 * image_size[0] * image_size[1]
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.net(z)
        x = x.view(-1, 3, self.image_size[0], self.image_size[1])
        return x

def build_decoder(config: dict):
    decoder_type = config.get('type', 'cnn_base')
    latent_dim = config.get('latent_dim', 256)
    image_size = config.get('image_size', [128, 128])
    hidden_dims = config.get('hidden_dims', None)
    
    if decoder_type == 'cnn_base':
        return CNNBaseDecoder(latent_dim, image_size, hidden_dims)
    elif decoder_type == 'mlp':
        return MLPDecoder(latent_dim, image_size, hidden_dims)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

class PalmDecoder(nn.Module):
    """
    Wrapper compatible with ProbabilisticPalmModel
    """
    def __init__(self, config: dict):
        super().__init__()
        self.model = build_decoder(config)
        
    def forward(self, z):
        return self.model(z)
