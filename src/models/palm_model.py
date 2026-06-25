import torch
import torch.nn as nn
from .base import BaseModel
from .encoder import PalmEncoder
from .decoder import PalmDecoder

class ProbabilisticPalmModel(BaseModel):
    """
    Main model that integrates Encoder and Decoder (optional).
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.encoder = PalmEncoder(config.get('encoder', {}))
        
        self.use_decoder = config.get('decoder', {}).get('use_decoder', True)
        if self.use_decoder:
            decoder_config = dict(config.get('decoder', {}))
            if 'latent_dim' not in decoder_config:
                decoder_config['latent_dim'] = config.get('encoder', {}).get('latent_dim', 256)
            if 'image_size' not in decoder_config:
                decoder_config['image_size'] = config.get('dataset', {}).get('image_size', [128, 128])
            self.decoder = PalmDecoder(decoder_config)
            
        # Light MLP (Projection Head) cho Contrastive Loss
        latent_dim = config.get('encoder', {}).get('latent_dim', 256)
        proj_dim = config.get('projector', {}).get('proj_dim', 128)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, proj_dim)
        )

    def reparameterize(self, mu, logvar, temperature=1.0):
        """
        Reparameterization trick: z = mu + sigma * epsilon * temperature
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * temperature

    def forward(self, x, decode=False, temperature=1.0):
        """
        Forward pass.
        If decode=True, also returns reconstructed image.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, temperature)
        
        out = {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'proj': self.projector(mu) # Đi qua Light MLP để phục vụ Contrastive Loss
        }
        
        if decode and self.use_decoder:
            x_hat = self.decoder(z)
            out['x_hat'] = x_hat
            
        return out
