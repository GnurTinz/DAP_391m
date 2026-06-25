import torch
import torch.nn as nn
from .base import BaseModel

class PalmDecoder(BaseModel):
    """
    Decoder to reconstruct image from latent vector z.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.latent_dim = self.config.get('latent_dim', 256)
        
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
        return self.decoder(x)
