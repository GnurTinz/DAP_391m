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
            
        latent_dim = config.get('encoder', {}).get('latent_dim', 256)
        proj_dim = config.get('projector', {}).get('proj_dim', 128)
        use_mlp = config.get('projector', {}).get('use_mlp', True)
        hidden_dims = config.get('projector', {}).get('hidden_dims', [])
        act_name = config.get('projector', {}).get('activation', 'ReLU')

        # inference_proj_input: 'z' (default cho ProbabilisticPalmModel — trước đây hard-code z)
        # hoặc 'mu' (deterministic, ổn định hơn tại inference)
        self.inference_proj_input = config.get('projector', {}).get('inference_proj_input', 'z')

        if use_mlp:
            activation_cls = getattr(nn, act_name, nn.ReLU)
            layers = []
            in_dim = latent_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(activation_cls())
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, proj_dim))
            # norm_type: 'bn' (default, BatchNorm1d), 'ln' (LayerNorm, ổn hơn với batch=1),
            #            'none' (không norm)
            norm_type = config.get('projector', {}).get('norm_type', 'bn')
            use_bn    = config.get('projector', {}).get('use_bn', True)
            if not use_bn:
                norm_type = 'none'
            if norm_type == 'bn':
                layers.append(nn.BatchNorm1d(proj_dim))
            elif norm_type == 'ln':
                layers.append(nn.LayerNorm(proj_dim))
            self.projector = nn.Sequential(*layers)
        else:
            self.projector = nn.Identity()

    def reparameterize(self, mu, logvar, temperature=1.0, mode='stochastic'):
        """
        Reparameterization trick
        """
        if mode == 'deterministic':
            return mu
        elif mode == 'symmetric':
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return -mu + eps * std * temperature
            
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * temperature

    def get_proj_input(self, mu: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Trả về input đúng cho projector dựa trên inference_proj_input config.
        """
        return z if self.inference_proj_input == 'z' else mu

    def forward(self, x, decode=False, temperature=1.0, sample_mode='stochastic'):
        """
        Forward pass.
        If decode=True, also returns reconstructed image.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, temperature, mode=sample_mode)

        # Inference: tôn trọng inference_proj_input config
        proj_input = self.get_proj_input(mu, z)

        out = {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'proj': self.projector(proj_input)
        }
        
        if decode and self.use_decoder:
            x_hat = self.decoder(z)
            out['x_hat'] = x_hat
            
        return out
