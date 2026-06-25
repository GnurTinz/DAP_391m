import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.models.palm_model import ProbabilisticPalmModel

class TestTimeVerifier(nn.Module):
    """
    MLP Verifier cho giai đoạn Test-Time.
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

def optimize_representation(model, image, config, device, num_samples=512, steps=50, lr=0.01, alpha=0.1):
    """
    Thực thi bài toán tối ưu r theo sơ đồ pipeline.
    """
    model.eval()
    with torch.no_grad():
        # Lấy mu, sigma từ ảnh
        # image shape: (1, C, H, W)
        out = model(image, decode=False)
        mu_q = out['mu'].detach() # (1, latent_dim)
        logvar_q = out['logvar'].detach()
        sigma_q = torch.exp(0.5 * logvar_q) # (1, latent_dim)
        latent_dim = mu_q.size(1)

    # 1. Sinh mẫu Positive (Từ phân phối q(z|x))
    # N mẫu positive
    eps_pos = torch.randn(num_samples, latent_dim, device=device)
    z_pos = mu_q + sigma_q * eps_pos
    y_pos = torch.ones(num_samples, 1, device=device)

    # 2. Sinh mẫu Negative (Từ phân phối chuẩn N(0, I) hoặc Hard Negatives tuỳ chọn)
    z_neg = torch.randn(num_samples, latent_dim, device=device)
    y_neg = torch.zeros(num_samples, 1, device=device)

    # Gộp data
    z_all = torch.cat([z_pos, z_neg], dim=0)
    y_all = torch.cat([y_pos, y_neg], dim=0)

    # 3. Khởi tạo r và Verifier
    r = nn.Parameter(mu_q.clone())
    verifier = TestTimeVerifier(latent_dim, hidden_dim=128).to(device)
    
    # 4. Tối ưu hoá
    optimizer = optim.Adam([
        {'params': verifier.parameters(), 'lr': lr},
        {'params': [r], 'lr': lr}
    ])
    criterion = nn.BCEWithLogitsLoss()

    print(f"Bắt đầu tối ưu r (Khởi tạo r giống mu_q). Số bước: {steps}")
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Đưa vào MLP
        logits = verifier(z_all, r)
        
        # Tính BCE Loss
        bce_loss = criterion(logits, y_all)
        
        # Tính L2 penalty để r không trôi quá xa mu
        l2_penalty = torch.norm(r - mu_q, p=2) ** 2
        
        # Tổng Loss
        total_loss = bce_loss + alpha * l2_penalty
        
        total_loss.backward()
        optimizer.step()
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step [{step+1}/{steps}] - BCE: {bce_loss.item():.4f}, L2_pen: {l2_penalty.item():.4f}, Total Loss: {total_loss.item():.4f}")
            
    print("Hoàn tất tối ưu r!")
    
    # Khoảng cách di chuyển của r so với ban đầu
    distance_moved = torch.norm(r - mu_q, p=2).item()
    print(f"Khoảng cách r đã dịch chuyển so với mu ban đầu: {distance_moved:.4f}")
    
    return r.detach()

def main():
    parser = argparse.ArgumentParser(description="Find optimal representation r (Test-Time Optimization)")
    parser.add_argument('--config', type=str, default='config/mnist.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (optional for dummy test)')
    parser.add_argument('--samples', type=int, default=512, help='Number of pos/neg samples (N)')
    parser.add_argument('--steps', type=int, default=50, help='Number of optimization steps (T)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    # Khởi tạo mô hình
    model = ProbabilisticPalmModel(config).to(device)
    
    # Load checkpoint nếu có
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Đang tải checkpoint từ {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Tải checkpoint thành công.")
    else:
        print("Cảnh báo: Không có checkpoint nào được cung cấp. Chạy với mô hình khởi tạo ngẫu nhiên để Test!")

    # Tạo 1 ảnh Dummy Data (1, C, H, W)
    img_size = config.get('dataset', {}).get('image_size', [128, 128])
    dummy_image = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
    
    print("\nBắt đầu quy trình giả lập Inference (FINDING r)...")
    optimized_r = optimize_representation(
        model=model,
        image=dummy_image,
        config=config,
        device=device,
        num_samples=args.samples,
        steps=args.steps,
        lr=0.01,
        alpha=0.1
    )
    
    print(f"\nVector r cuối cùng có shape: {optimized_r.shape}")

if __name__ == '__main__':
    main()
