import torch
import torch.nn as nn

class TestTimeVerifier(nn.Module):
    """
    MLP Verifier cho giai đoạn Test-Time và Verification.
    Nhận đầu vào là [Z, r] (concat), trả về xác suất mẫu là Positive.
    """
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z, r):
        # r có thể là 1 vector, được expand_as(z) nếu cần ở bên ngoài, 
        # hoặc broadcast concat trực tiếp
        r_expanded = r.expand(z.size(0), -1)
        feat = torch.cat([z, r_expanded], dim=1)
        return self.net(feat)
