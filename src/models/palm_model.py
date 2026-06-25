import torch
import torch.nn as nn
from .base import BaseModel
from .encoder import PalmEncoder
from .decoder import PalmDecoder
from .verifier import PairVerifier

class ProbabilisticPalmModel(BaseModel):
    """
    Main model that integrates Encoder, Decoder (optional), and Verifier.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.encoder = PalmEncoder(config.get('encoder', {}))
        
        self.use_decoder = config.get('decoder', {}).get('use_decoder', True)
        if self.use_decoder:
            self.decoder = PalmDecoder(config.get('decoder', {}))
            
        self.verifier = PairVerifier(config.get('verifier', {}))
        
        # Light MLP (Projection Head) cho Contrastive Loss
        latent_dim = config.get('encoder', {}).get('latent_dim', 256)
        proj_dim = config.get('projector', {}).get('proj_dim', 128)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim)
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, decode=False):
        """
        Forward pass.
        If decode=True, also returns reconstructed image.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
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
        
    def verify(self, mu1, logvar1, mu2, logvar2):
        """
        Run the pair verifier.
        """
        return self.verifier(mu1, logvar1, mu2, logvar2)
