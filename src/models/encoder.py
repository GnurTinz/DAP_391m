import torch
import torch.nn as nn
from .base import BaseModel
import torchvision.models as models

class PalmEncoder(BaseModel):
    """
    Encoder that outputs mu and log_var.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        backbone_name = self.config.get('backbone', 'resnet18')
        pretrained = self.config.get('pretrained', True)
        self.latent_dim = self.config.get('latent_dim', 256)
        
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            # Fallback for mock/skeleton
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            in_features = 64

        self.fc_mu = nn.Linear(in_features, self.latent_dim)
        self.fc_logvar = nn.Linear(in_features, self.latent_dim)

    def forward(self, x):
        features = self.backbone(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar
