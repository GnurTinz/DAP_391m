import torch
import torch.nn as nn
import torch.optim as optim
from src.models.verifier import TestTimeVerifier

def optimize_r_from_latent(mu_q, logvar_q, device, verifier=None, freeze_net=False, num_samples=512, steps=50, lr=0.01, alpha=0.1, verbose=True):
    """
    Thực thi bài toán tối ưu r theo sơ đồ pipeline nhưng từ latent (mu, logvar).
    """
    sigma_q = torch.exp(0.5 * logvar_q) # (1, latent_dim)
    latent_dim = mu_q.size(1)

    # 1. Sinh mẫu Positive (Từ phân phối q(z|x))
    eps_pos = torch.randn(num_samples, latent_dim, device=device)
    z_pos = mu_q + sigma_q * eps_pos
    y_pos = torch.ones(num_samples, 1, device=device)

    # 2. Sinh mẫu Negative
    z_neg = torch.randn(num_samples, latent_dim, device=device)
    y_neg = torch.zeros(num_samples, 1, device=device)

    z_all = torch.cat([z_pos, z_neg], dim=0)
    y_all = torch.cat([y_pos, y_neg], dim=0)

    # 3. Khởi tạo r và Verifier
    r = nn.Parameter(mu_q.clone())
    if verifier is None:
        verifier = TestTimeVerifier(latent_dim, hidden_dim=128).to(device)
    
    if freeze_net:
        # Nếu freeze_net = True, chỉ tối ưu r
        for param in verifier.parameters():
            param.requires_grad = False
        optimizer = optim.Adam([
            {'params': [r], 'lr': lr}
        ])
    else:
        # Tối ưu cả r và verifier
        for param in verifier.parameters():
            param.requires_grad = True
        optimizer = optim.Adam([
            {'params': verifier.parameters(), 'lr': lr},
            {'params': [r], 'lr': lr}
        ])
    criterion = nn.BCEWithLogitsLoss()

    if verbose:
        print(f"Bắt đầu tối ưu r (Khởi tạo r giống mu_q). Số bước: {steps}")
    
    for step in range(steps):
        optimizer.zero_grad()
        logits = verifier(z_all, r)
        bce_loss = criterion(logits, y_all)
        l2_penalty = torch.norm(r - mu_q, p=2) ** 2
        total_loss = bce_loss + alpha * l2_penalty
        
        total_loss.backward()
        optimizer.step()
        
        if verbose and ((step + 1) % 10 == 0 or step == 0):
            print(f"Step [{step+1}/{steps}] - BCE: {bce_loss.item():.4f}, L2_pen: {l2_penalty.item():.4f}, Total Loss: {total_loss.item():.4f}")
            
    if verbose:
        print("Hoàn tất tối ưu r!")
        distance_moved = torch.norm(r - mu_q, p=2).item()
        print(f"Khoảng cách r đã dịch chuyển so với mu ban đầu: {distance_moved:.4f}")
    
    return r.detach(), verifier, z_pos

def optimize_representation(model, image, config, device, num_samples=512, steps=50, lr=0.01, alpha=0.1):
    """
    Thực thi bài toán tối ưu r theo sơ đồ pipeline từ ảnh.
    """
    model.eval()
    with torch.no_grad():
        out = model(image, decode=False)
        mu_q = out['mu'].detach() # (1, latent_dim)
        logvar_q = out['logvar'].detach()
        
    r, verifier, z_pos = optimize_r_from_latent(mu_q, logvar_q, device, verifier=None, freeze_net=False, num_samples=num_samples, steps=steps, lr=lr, alpha=alpha, verbose=True)
    return r
