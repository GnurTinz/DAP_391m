import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseDecoder(nn.Module):
    def __init__(self, latent_dim=256, image_size=(128, 128), hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        # Tính toán kích thước khởi tạo (dựa trên 4 lớp ConvTranspose2d với stride=2 -> scale 16 lần)
        self.init_size = image_size[0] // 16
        self.init_size = max(self.init_size, 1) # Đảm bảo ít nhất là 1x1
        
        # Projection
        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)
        
        # ConvTranspose blocks
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, self.init_size, self.init_size)
        x = self.decoder(x)
        if list(x.shape[-2:]) != list(self.image_size):
            x = F.interpolate(x, size=tuple(self.image_size), mode='bilinear', align_corners=False)
        return x

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim=256, image_size=(128, 128), hidden_dims=None):
        super().__init__()
        self.image_size = image_size
        out_dim = 3 * image_size[0] * image_size[1]
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.net(z)
        x = x.view(-1, 3, self.image_size[0], self.image_size[1])
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DetailedResNetDecoder(nn.Module):
    """
    Decoder sử dụng Residual Blocks và ConvTranspose để sinh ảnh sắc nét, chi tiết hơn (Tránh mờ nhòe).
    """
    def __init__(self, latent_dim=256, image_size=(128, 128), hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.init_size = image_size[0] // 16
        self.init_size = max(self.init_size, 1)
        
        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)
        
        self.decoder = nn.Sequential(
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            ResidualBlock(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, self.init_size, self.init_size)
        x = self.decoder(x)
        if list(x.shape[-2:]) != list(self.image_size):
            x = F.interpolate(x, size=tuple(self.image_size), mode='bilinear', align_corners=False)
        return x

def build_decoder(config: dict):
    decoder_type = config.get('type', 'cnn_base')
    latent_dim = config.get('latent_dim', 256)
    image_size = config.get('image_size', [128, 128])
    hidden_dims = config.get('hidden_dims', None)
    
    if decoder_type == 'cnn_base':
        return CNNBaseDecoder(latent_dim, image_size, hidden_dims)
    elif decoder_type == 'mlp':
        return MLPDecoder(latent_dim, image_size, hidden_dims)
    elif decoder_type == 'resnet':
        return DetailedResNetDecoder(latent_dim, image_size, hidden_dims)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

class PalmDecoder(nn.Module):
    """
    Wrapper compatible with ProbabilisticPalmModel
    """
    def __init__(self, config: dict):
        super().__init__()
        self.model = build_decoder(config)
        
    def forward(self, z):
        return self.model(z)
