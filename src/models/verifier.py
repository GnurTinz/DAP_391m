import torch
import torch.nn as nn
from .base import BaseModel

class PairVerifier(BaseModel):
    """
    MLP verifier for checking if two latent distributions belong to the same identity.
    Input features: [mu_1, mu_2, |mu_1 - mu_2|, mu_1 * mu_2, sigma_1, sigma_2, |sigma_1 - sigma_2|]
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.latent_dim = self.config.get('latent_dim', 256)
        hidden_dims = self.config.get('hidden_dims', [512, 256, 128])
        
        # 7 components as specified in the pipeline note
        in_features = 7 * self.latent_dim
        
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        # BCEWithLogitsLoss doesn't need Sigmoid at the end
        self.mlp = nn.Sequential(*layers)

    def forward(self, mu1, logvar1, mu2, logvar2):
        sigma1 = torch.exp(0.5 * logvar1)
        sigma2 = torch.exp(0.5 * logvar2)
        
        feat = torch.cat([
            mu1,
            mu2,
            torch.abs(mu1 - mu2),
            mu1 * mu2,
            sigma1,
            sigma2,
            torch.abs(sigma1 - sigma2)
        ], dim=1)
        
        return self.mlp(feat)
