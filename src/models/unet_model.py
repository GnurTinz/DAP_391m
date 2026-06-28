import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel
from .encoder import PalmEncoder

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # x2 is the skip connection from the encoder
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetPalmModel(BaseModel):
    """
    Probabilistic U-Net: 
    - Có skip connections để tái tạo ảnh cực nét.
    - Tại đáy U-Net (bottleneck), thông tin được chẻ ra thành mu và logvar.
    - Reparameterization -> vector z (dành cho Contrastive Loss).
    - Giải mã từ z kết hợp với các skip connections.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.image_size = config.get('decoder', {}).get('image_size', [128, 128])
        self.latent_dim = config.get('encoder', {}).get('latent_dim', 128)
        self.proj_dim = config.get('projector', {}).get('proj_dim', 128)
        self.use_decoder = config.get('decoder', {}).get('use_decoder', True)
        self.skip_dropout_rate = config.get('decoder', {}).get('skip_dropout', 0.2)
        
        unet_channels = config.get('decoder', {}).get('unet_channels', [64, 128, 256, 512])
        self.c1, self.c2, self.c3, self.c4 = unet_channels

        # --- ENCODER PATH ---
        self.inc = DoubleConv(3, self.c1)
        self.down1 = DownBlock(self.c1, self.c2)
        self.down2 = DownBlock(self.c2, self.c3)
        self.down3 = DownBlock(self.c3, self.c4)
        
        # Tính kích thước không gian tại bottleneck
        self.bottleneck_size = self.image_size[0] // 8
        self.bottleneck_size = max(self.bottleneck_size, 1)
        self.flatten_size = self.c4 * self.bottleneck_size * self.bottleneck_size
        
        # --- PROBABILISTIC BOTTLENECK ---
        # Sử dụng một mạng riêng biệt (Latent Encoder) để trích xuất mu và logvar
        # thay vì dùng chung thân mạng U-Net
        self.latent_encoder = PalmEncoder(config.get('encoder', {}))
        
        # --- FiLM MODULATION LAYERS ---
        # Điều khiển skip connection x3
        self.film_gamma3 = nn.Linear(self.latent_dim, self.c3)
        self.film_beta3 = nn.Linear(self.latent_dim, self.c3)
        
        # Điều khiển skip connection x2
        self.film_gamma2 = nn.Linear(self.latent_dim, self.c2)
        self.film_beta2 = nn.Linear(self.latent_dim, self.c2)
        
        # Projector for Contrastive Loss
        use_mlp = config.get('projector', {}).get('use_mlp', True)
        hidden_dims = config.get('projector', {}).get('hidden_dims', [])
        act_name = config.get('projector', {}).get('activation', 'ReLU')
        
        if use_mlp:
            activation_cls = getattr(nn, act_name, nn.ReLU)
            layers = []
            in_dim = self.latent_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(activation_cls())
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, self.proj_dim))
            # Bổ sung BatchNorm1d trước khi đi vào ArcFace (Tuyệt chiêu giúp hội tụ cực nhanh như Baseline 2)
            use_bn = config.get('projector', {}).get('use_bn', True)
            if use_bn:
                layers.append(nn.BatchNorm1d(self.proj_dim))
            self.projector = nn.Sequential(*layers)
        else:
            self.projector = nn.Identity()
            self.proj_dim = self.latent_dim # When using Identity, output dim matches latent dim
        
        # --- DECODER PATH ---
        if self.use_decoder:
            self.fc_dec = nn.Linear(self.latent_dim, self.flatten_size)
            self.up1 = UpBlock(self.c4, self.c3)
            self.up2 = UpBlock(self.c3, self.c2)
            self.up3 = UpBlock(self.c2, self.c1)
            self.outc = nn.Sequential(
                nn.Conv2d(self.c1, 3, kernel_size=1),
                nn.Tanh() # Đưa về [-1, 1]
            )
            self.skip_dropout = nn.Dropout2d(p=self.skip_dropout_rate)

    def reparameterize(self, mu, logvar, temperature=1.0, mode='stochastic'):
        if mode == 'deterministic':
            return mu
        elif mode == 'symmetric':
            # Đối xứng qua gốc toạ độ của không gian latent
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return -mu + eps * std * temperature
            
        # Mặc định stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * temperature

    def forward(self, x, decode=False, temperature=1.0, sample_mode='stochastic'):
        # 1. Trích xuất Latent distribution từ mạng riêng (Posterior/Prior Network)
        mu, logvar = self.latent_encoder(x)
        
        # Bóp logvar về biên an toàn để chống bùng nổ hàm mũ exp(logvar) trong KL loss 
        # và chống nhiễu loạn không gian latent z
        logvar = torch.clamp(logvar, min=-20, max=2.0)

        z = self.reparameterize(mu, logvar, temperature, mode=sample_mode)
        
        out = {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'proj': self.projector(mu)
        }
        
        if decode and self.use_decoder:
            out['x_hat'] = self.decode_from_z_and_x(z, x)
            
        return out

    def decode_from_z_and_x(self, z, x):
        """
        Giải mã ra ảnh x_hat từ vector tiềm ẩn z, sử dụng skip connections trích xuất từ ảnh x.
        """
        if not self.use_decoder:
            raise ValueError("Mô hình không sử dụng decoder.")
            
        # 1. Trích xuất Spatial Features từ Deterministic U-Net Encoder
        x1 = self.inc(x)       # Skip 1 (c1 channels)
        x2 = self.down1(x1)    # Skip 2 (c2 channels)
        x3 = self.down2(x2)    # Skip 3 (c3 channels)
        
        # 2. Giải mã từ latent z
        z_dec = self.fc_dec(z)
        z_dec = z_dec.view(-1, self.c4, self.bottleneck_size, self.bottleneck_size)
        
        # 3. Kết hợp z vào U-Net thông qua FiLM để điều khiển skip connections
        gamma3 = self.film_gamma3(z).view(-1, self.c3, 1, 1)
        beta3 = self.film_beta3(z).view(-1, self.c3, 1, 1)
        modulated_x3 = (1 + gamma3) * x3 + beta3
        
        gamma2 = self.film_gamma2(z).view(-1, self.c2, 1, 1)
        beta2 = self.film_beta2(z).view(-1, self.c2, 1, 1)
        modulated_x2 = (1 + gamma2) * x2 + beta2
        
        # Áp dụng Dropout2d cho toàn bộ các skip connection
        d_x3 = self.skip_dropout(modulated_x3)
        d_x2 = self.skip_dropout(modulated_x2)
        d_x1 = self.skip_dropout(x1)
        
        u1 = self.up1(z_dec, d_x3)
        u2 = self.up2(u1, d_x2)
        u3 = self.up3(u2, d_x1)
        
        x_hat = self.outc(u3)
        
        # Đảm bảo output size đúng bằng input size
        if list(x_hat.shape[-2:]) != list(self.image_size):
            x_hat = F.interpolate(x_hat, size=tuple(self.image_size), mode='bilinear', align_corners=False)
            
        return x_hat
