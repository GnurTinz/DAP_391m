import unittest
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. Định nghĩa các Module (Mockup)
# ==========================================

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        # Mockup CNN encoder
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Giả sử ảnh đầu vào 64x64, sau 2 lần pool -> 16x16
        self.fc_mu = nn.Linear(32 * 16 * 16, latent_dim)
        self.fc_sigma = nn.Linear(32 * 16 * 16, latent_dim)
        
    def forward(self, x):
        features = self.conv(x)
        mu = self.fc_mu(features)
        log_sigma = self.fc_sigma(features) # Thường xuất log(sigma) để ổn định
        sigma = torch.exp(log_sigma)
        return mu, sigma

class Reparameterize(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 16 * 16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 32, 16, 16)
        x_hat = self.deconv(x)
        return x_hat

class LightMLP(nn.Module):
    """ Dùng cho Contrastive Learning (Push/Pull) """
    def __init__(self, latent_dim=128, proj_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim)
        )
        
    def forward(self, z):
        return self.mlp(z)

class VerificationMLP(nn.Module):
    """ Dùng cho bước Verification (Accept/Reject) """
    def __init__(self, latent_dim=128):
        super().__init__()
        # Đầu vào có thể là concat của query và candidate/r (2 * latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Trả về xác suất match
        )
        
    def forward(self, query_mu, candidate_r):
        combined = torch.cat([query_mu, candidate_r], dim=1)
        score = self.mlp(combined)
        return score

# ==========================================
# 2. Chương trình Test (Unittest style)
# ==========================================

class TestPipelineModules(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.channels = 3
        self.H = 64
        self.W = 64
        self.latent_dim = 128
        self.proj_dim = 64

        # Initialize mock modules
        self.encoder = Encoder(self.channels, self.latent_dim)
        self.reparam = Reparameterize()
        self.decoder = Decoder(self.latent_dim, self.channels)
        self.light_mlp = LightMLP(self.latent_dim, self.proj_dim)
        self.verification_mlp = VerificationMLP(self.latent_dim)

    def test_training_pipeline(self):
        # 0. Input data
        x = torch.randn(self.batch_size, self.channels, self.H, self.W)
        
        # 1. Encode
        mu, sigma = self.encoder(x)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim), "Shape của mu không đúng")
        self.assertEqual(sigma.shape, (self.batch_size, self.latent_dim), "Shape của sigma không đúng")
        
        # 2. Sample (Reparameterize)
        z = self.reparam(mu, sigma)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim), "Shape của z không đúng")
        
        # 3. Decode
        x_hat = self.decoder(z)
        self.assertEqual(x_hat.shape, x.shape, "Shape của ảnh tái tạo x_hat không đúng")
        
        # 4. Contrastive Projection
        proj = self.light_mlp(z)
        self.assertEqual(proj.shape, (self.batch_size, self.proj_dim), "Shape của projection không đúng")

    def test_inference_finding_r(self):
        # Giả lập query embedding (x_new)
        query_mu = torch.randn(1, self.latent_dim)
        
        # Giả lập database (Db)
        num_candidates = 100
        database = torch.randn(num_candidates, self.latent_dim)
        
        # 1. Tìm ứng viên gần nhất (Attend / Find similar)
        dists = torch.norm(database - query_mu, dim=1)
        best_idx = torch.argmin(dists)
        r_candidate = database[best_idx].unsqueeze(0) # r lấy từ DB
        
        self.assertEqual(r_candidate.shape, (1, self.latent_dim), "Shape của r_candidate từ DB không đúng")
        
        # 2. Optimization (Finding r - Test-time optimization loop mock)
        r_optim = r_candidate.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([r_optim], lr=0.01)
        
        # Lưu lại bản gốc để kiểm tra xem tham số có thực sự được update không
        original_r = r_optim.clone()
        
        # Giả lập 1 step tối ưu r
        optimizer.zero_grad()
        score = self.verification_mlp(query_mu, r_optim)
        loss = 1.0 - score # Mục tiêu: Tối đa hóa score => giảm loss
        loss.backward()
        optimizer.step()
        
        self.assertEqual(r_optim.shape, (1, self.latent_dim), "Shape của r_optim bị lỗi sau backward")
        self.assertFalse(torch.allclose(r_optim, original_r), "r_optim không được cập nhật sau quá trình optimize")
        
        # 3. Final Verification
        final_score = self.verification_mlp(query_mu, r_optim.detach())
        self.assertTrue(0 <= final_score.item() <= 1.0, "Score phải nằm trong [0, 1]")

if __name__ == "__main__":
    unittest.main()
