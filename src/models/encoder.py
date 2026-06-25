import torch
import torch.nn as nn
from .base import BaseModel
import torchvision.models as models

class PalmEncoder(BaseModel):
    """
    Encoder that outputs mu and log_var.
    Supports pluggable backbones from torchvision.models.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        backbone_name = self.config.get('backbone', 'resnet18')
        pretrained = self.config.get('pretrained', True)
        self.latent_dim = self.config.get('latent_dim', 256)
        
        if backbone_name == 'mock':
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            in_features = 64
        elif hasattr(models, backbone_name):
            # Dynamic loading from torchvision
            model_func = getattr(models, backbone_name)
            # Some models don't support pretrained kwargs anymore, use weights. 
            # For simplicity, we just use pretrained=pretrained if supported.
            try:
                self.backbone = model_func(pretrained=pretrained)
            except TypeError:
                # Fallback for newer torchvision versions
                self.backbone = model_func(weights='DEFAULT' if pretrained else None)
            
            # Find the final classification layer to replace it and get in_features
            if hasattr(self.backbone, 'fc'): # ResNets
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            elif hasattr(self.backbone, 'classifier'):
                if isinstance(self.backbone.classifier, nn.Linear): # DenseNet
                    in_features = self.backbone.classifier.in_features
                    self.backbone.classifier = nn.Identity()
                elif isinstance(self.backbone.classifier, nn.Sequential): # MobileNet, VGG
                    # Try to find the last linear layer
                    last_layer = self.backbone.classifier[-1]
                    if isinstance(last_layer, nn.Linear):
                        in_features = last_layer.in_features
                        self.backbone.classifier[-1] = nn.Identity()
                    elif isinstance(last_layer, nn.Dropout):
                        # Sometimes there's dropout before linear in classifier
                        # Look one step back
                        if isinstance(self.backbone.classifier[-2], nn.Linear):
                            in_features = self.backbone.classifier[-2].in_features
                            self.backbone.classifier[-2] = nn.Identity()
                            self.backbone.classifier[-1] = nn.Identity() # Remove dropout too
                        else:
                            raise ValueError(f"Cannot parse classifier in {backbone_name}")
                    else:
                        raise ValueError(f"Cannot parse classifier sequential in {backbone_name}")
                else:
                    raise ValueError(f"Unknown classifier format in {backbone_name}")
            else:
                raise ValueError(f"Unsupported backbone structure for auto-feature extraction: {backbone_name}")
        else:
            raise ValueError(f"Backbone {backbone_name} not found in torchvision.models and is not 'mock'.")

        self.fc_mu = nn.Linear(in_features, self.latent_dim)
        self.fc_logvar = nn.Linear(in_features, self.latent_dim)

    def forward(self, x):
        features = self.backbone(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar
