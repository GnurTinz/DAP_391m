import torch
import torch.nn as nn
import torch.optim as optim
from src.models.verifier import TestTimeVerifier

def optimize_r_from_latent(mu_q, logvar_q, device, verifier=None, freeze_net=False, num_samples=512, steps=50, lr=0.01, alpha=0.1, config=None, verbose=True):
    """
    Thực thi bài toán tối ưu r theo sơ đồ pipeline nhưng từ latent (mu, logvar).
    Nhận các thiết lập sampling (pos_strategy, pos_temp, neg_strategy, neg_temp) từ config YAML.
    """
    if config is None:
        config = {}
        
    rep_config = config.get('represent', {})
    pos_temp = rep_config.get('pos_temperature', 1.0)
    pos_strategy = rep_config.get('pos_strategy', 'stochastic')
    neg_temp = rep_config.get('neg_temperature', 1.0)
    neg_strategy = rep_config.get('neg_strategy', 'random')

    sigma_q = torch.exp(0.5 * logvar_q) # (1, latent_dim)
    latent_dim = mu_q.size(1)

    # 1. Sinh mẫu Positive
    if pos_strategy == 'deterministic':
        z_pos = mu_q.expand(num_samples, -1)
    else: # stochastic mặc định
        eps_pos = torch.randn(num_samples, latent_dim, device=device)
        z_pos = mu_q + pos_temp * sigma_q * eps_pos
    y_pos = torch.ones(num_samples, 1, device=device)

    # 2. Sinh mẫu Negative
    if neg_strategy == 'spherical':
        import math
        # Sử dụng logic Latent Rotation (xoay vector tiềm ẩn) giống y hệt generate_from_latent trong ImageGenerator
        mu_norm = torch.norm(mu_q, p=2, dim=1, keepdim=True) + 1e-8
        mu_unit = mu_q / mu_norm
        
        # Góc xoay theta trải từ 45 độ (pi/4) đến 180 độ (pi)
        theta = (math.pi / 4) + (math.pi * 3 / 4) * torch.rand(num_samples, 1, device=device)
        
        # Tìm hướng ngẫu nhiên trực giao với mu để tạo mặt phẳng xoay (batched)
        v = torch.randn(num_samples, latent_dim, device=device)
        v_proj = torch.sum(v * mu_unit, dim=1, keepdim=True) * mu_unit
        v_ortho = v - v_proj
        v_ortho_unit = v_ortho / (torch.norm(v_ortho, p=2, dim=1, keepdim=True) + 1e-8)
        
        # Xoay mu trên mặt phẳng 2D
        z_rot = mu_norm * (mu_unit * torch.cos(theta) + v_ortho_unit * torch.sin(theta))
        
        # Thêm chút nhiễu bất định (giống generate_images)
        eps_neg = torch.randn(num_samples, latent_dim, device=device)
        z_neg = z_rot + neg_temp * eps_neg * sigma_q
    else:
        # Default 'random': Sinh ngẫu nhiên từ phân phối Gaussian
        z_neg = torch.randn(num_samples, latent_dim, device=device) * neg_temp
        
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
    Hỗ trợ config yaml:
    - mode: 'single' (tối ưu từ 1 ảnh) hoặc 'average' (tính z trung bình của 2 ảnh).
    - re_encode: True (decode z ra ảnh x_hat, sau đó dùng encoder trích xuất lại mu, logvar từ x_hat rồi mới tìm r) hoặc False.
    """
    model.eval()
    rep_config = config.get('represent', {})
    mode = rep_config.get('mode', 'single')
    re_encode = rep_config.get('re_encode', False)
    
    with torch.no_grad():
        if mode == 'average' and image.size(0) >= 2:
            print("Chế độ 'average': Tính trung bình không gian Latent của 2 ảnh.")
            x1 = image[0:1]
            x2 = image[1:2]
            
            out1 = model(x1, decode=True)
            out2 = model(x2, decode=True)
            
            mu_avg = (out1['mu'] + out2['mu']) / 2.0
            logvar_avg = (out1['logvar'] + out2['logvar']) / 2.0
            
            # Sinh ảnh biểu diễn x_hat từ mu_avg
            if hasattr(model, 'decode_from_z_and_x'): # U-Net
                x_avg = (x1 + x2) / 2.0
                x_hat = model.decode_from_z_and_x(mu_avg, x_avg)
            else: # VAE
                x_hat = model.decoder(mu_avg)
                
            target_mu = mu_avg
            target_logvar = logvar_avg
            
        else:
            if mode == 'average':
                print("Cảnh báo: mode='average' yêu cầu batch size >= 2. Tự động chuyển về 'single'.")
            
            out = model(image[0:1], decode=True)
            target_mu = out['mu']
            target_logvar = out['logvar']
            x_hat = out.get('x_hat', None)
            
        if re_encode and x_hat is not None:
            print("Chế độ 're_encode': Đang dùng Feature Extractor trích xuất lại đặc trưng từ ảnh sinh ra (x_hat)...")
            out_re = model(x_hat, decode=False)
            target_mu = out_re['mu'].detach()
            target_logvar = out_re['logvar'].detach()
        else:
            target_mu = target_mu.detach()
            target_logvar = target_logvar.detach()
        
    r, verifier, z_pos = optimize_r_from_latent(target_mu, target_logvar, device, verifier=None, freeze_net=False, num_samples=num_samples, steps=steps, lr=lr, alpha=alpha, config=config, verbose=True)
    return r
