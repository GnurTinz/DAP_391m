import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel

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
        
        # --- ENCODER PATH ---
        self.inc = DoubleConv(3, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        
        # Tính kích thước không gian tại bottleneck
        self.bottleneck_size = self.image_size[0] // 8
        self.bottleneck_size = max(self.bottleneck_size, 1)
        self.flatten_size = 512 * self.bottleneck_size * self.bottleneck_size
        
        # --- PROBABILISTIC BOTTLENECK ---
        self.norm_flat = nn.LayerNorm(self.flatten_size)
        self.fc_mu = nn.Linear(self.flatten_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, self.latent_dim)
        
        # Light MLP for Contrastive Loss
        self.projector = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.proj_dim)
        )
        
        # --- DECODER PATH ---
        if self.use_decoder:
            self.fc_dec = nn.Linear(self.latent_dim, self.flatten_size)
            self.up1 = UpBlock(512, 256)
            self.up2 = UpBlock(256, 128)
            self.up3 = UpBlock(128, 64)
            self.outc = nn.Sequential(
                nn.Conv2d(64, 3, kernel_size=1),
                nn.Tanh() # Đưa về [-1, 1]
            )
            self.skip_dropout = nn.Dropout2d(p=self.skip_dropout_rate)

    def reparameterize(self, mu, logvar, temperature=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * temperature

    def forward(self, x, decode=False, temperature=1.0):
        # Encoder (Lưu lại skip connections)
        x1 = self.inc(x)       # Skip 1 (64 channels)
        x2 = self.down1(x1)    # Skip 2 (128 channels)
        x3 = self.down2(x2)    # Skip 3 (256 channels)
        x4 = self.down3(x3)    # Bottleneck input (512 channels)
        
        # Bottleneck (Probabilistic)
        x4_flat = x4.view(x4.size(0), -1)
        x4_flat = self.norm_flat(x4_flat)
        mu = self.fc_mu(x4_flat)
        logvar = self.fc_logvar(x4_flat)
        
        # Bóp logvar về biên an toàn để chống bùng nổ hàm mũ exp(logvar) trong KL loss 
        # và chống nhiễu loạn không gian latent z
        logvar = torch.clamp(logvar, min=-20, max=2.0)

        z = self.reparameterize(mu, logvar, temperature)
        
        out = {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'proj': self.projector(mu)
        }
        
        if decode and self.use_decoder:
            # Giải mã từ latent z
            z_dec = self.fc_dec(z)
            z_dec = z_dec.view(-1, 512, self.bottleneck_size, self.bottleneck_size)
            
            # Kết hợp với skip connections từ encoder
            # Áp dụng Dropout2d để triệt tiêu ngẫu nhiên một số kênh đặc trưng (feature maps)
            # Buộc Decoder phải phụ thuộc vào z để giải mã, chống Posterior Collapse
            d_x3 = self.skip_dropout(x3)
            d_x2 = self.skip_dropout(x2)
            d_x1 = self.skip_dropout(x1)
            
            u1 = self.up1(z_dec, d_x3)
            u2 = self.up2(u1, d_x2)
            u3 = self.up3(u2, d_x1)
            
            x_hat = self.outc(u3)
            
            # Đảm bảo output size đúng bằng input size
            if list(x_hat.shape[-2:]) != list(self.image_size):
                x_hat = F.interpolate(x_hat, size=tuple(self.image_size), mode='bilinear', align_corners=False)
                
            out['x_hat'] = x_hat
            
        return out
