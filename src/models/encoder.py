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
        self.latent_dim = self.config.get('latent_dim', 256)
        
        self.backbone, in_features = self._build_backbone(backbone_name)

        self.fc_mu = nn.Linear(in_features, self.latent_dim)
        self.fc_logvar = nn.Linear(in_features, self.latent_dim)

    def _build_backbone(self, backbone_name):
        if backbone_name == 'mock':
            return self._build_mock()
        elif backbone_name == 'cnn':
            return self._build_cnn()
        elif backbone_name == 'ccnet':
            return self._build_ccnet()
        elif backbone_name == 'palmnet':
            return self._build_palmnet()
        elif hasattr(models, backbone_name):
            return self._build_torchvision(backbone_name)
        else:
            raise ValueError(f"Backbone {backbone_name} not found in torchvision.models and is not 'mock'.")

    def _build_mock(self):
        in_channels = self.config.get('in_channels', 3)
        backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        return backbone, 64

    def _build_cnn(self):
        hidden_dims = self.config.get('hidden_dims', [32, 64, 128, 256])
        modules = []
        in_channels = self.config.get('in_channels', 3)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = h_dim
        
        backbone = nn.Sequential(
            *modules,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        return backbone, hidden_dims[-1]

    def _build_ccnet(self):
        from src.models.ccnet import CCNetBackbone
        in_channels = self.config.get('in_channels', 3)
        weight = self.config.get('ccnet_weight', 0.8)
        backbone = CCNetBackbone(in_channels=in_channels, weight=weight)
        return backbone, 2048

    def _build_palmnet(self):
        from src.models.palmnet import PalmNetBackbone
        in_channels = self.config.get('in_channels', 3)
        use_gabor_init = self.config.get('palmnet_gabor_init', True)
        backbone = PalmNetBackbone(in_channels=in_channels, use_gabor_init=use_gabor_init)
        return backbone, 64

    def _build_torchvision(self, backbone_name):
        pretrained = self.config.get('pretrained', True)
        in_channels = self.config.get('in_channels', 3)
        model_func = getattr(models, backbone_name)
        
        try:
            backbone = model_func(pretrained=pretrained)
        except TypeError:
            backbone = model_func(weights='DEFAULT' if pretrained else None)
            
        # Tự động thay đổi in_channels của lớp Conv đầu tiên nếu khác 3
        if in_channels != 3:
            if hasattr(backbone, 'conv1') and isinstance(backbone.conv1, nn.Conv2d):
                old_conv = backbone.conv1
                backbone.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, 
                                           kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                                           padding=old_conv.padding, bias=(old_conv.bias is not None))
            elif hasattr(backbone, 'features') and isinstance(backbone.features[0], nn.Conv2d):
                old_conv = backbone.features[0]
                backbone.features[0] = nn.Conv2d(in_channels, old_conv.out_channels, 
                                           kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                                           padding=old_conv.padding, bias=(old_conv.bias is not None))

        if hasattr(backbone, 'fc'): # ResNets
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier'):
            if isinstance(backbone.classifier, nn.Linear): # DenseNet
                in_features = backbone.classifier.in_features
                backbone.classifier = nn.Identity()
            elif isinstance(backbone.classifier, nn.Sequential): # MobileNet, VGG
                last_layer = backbone.classifier[-1]
                if isinstance(last_layer, nn.Linear):
                    in_features = last_layer.in_features
                    backbone.classifier[-1] = nn.Identity()
                elif isinstance(last_layer, nn.Dropout):
                    if isinstance(backbone.classifier[-2], nn.Linear):
                        in_features = backbone.classifier[-2].in_features
                        backbone.classifier[-2] = nn.Identity()
                        backbone.classifier[-1] = nn.Identity()
                    else:
                        raise ValueError(f"Cannot parse classifier in {backbone_name}")
                else:
                    raise ValueError(f"Cannot parse classifier sequential in {backbone_name}")
            else:
                raise ValueError(f"Unknown classifier format in {backbone_name}")
        else:
            raise ValueError(f"Unsupported backbone structure for auto-feature extraction: {backbone_name}")
            
        return backbone, in_features

    def forward(self, x):
        features = self.backbone(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar
