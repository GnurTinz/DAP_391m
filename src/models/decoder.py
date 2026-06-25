import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel

class PalmDecoder(BaseModel):
    """
    Decoder to reconstruct image from latent vector z.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.latent_dim = self.config.get('latent_dim', 256)
        self.image_size = self.config.get('image_size', [128, 128])
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        
        self.fc = nn.Linear(self.latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8)
        x = self.decoder(x)
        if list(x.shape[-2:]) != list(self.image_size):
            x = F.interpolate(x, size=tuple(self.image_size), mode='bilinear', align_corners=False)
        return x
