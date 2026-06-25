import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLoss

class KLDivLoss(BaseLoss):
    """
    KL Divergence between learned distribution and N(0, I).
    """
    def forward(self, mu, logvar):
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld / mu.size(0)

class UncertaintyLoss(BaseLoss):
    """
    Penalize sigma from becoming too small or too large.
    """
    def forward(self, logvar, lower_bound=-4.0, upper_bound=2.0):
        mean_logvar = logvar.mean()
        penalty = F.relu(lower_bound - mean_logvar) + F.relu(mean_logvar - upper_bound)
        return penalty

class ReconstructionLoss(BaseLoss):
    """
    L1 or L2 loss for reconstruction.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        loss_type = config.get('type', 'mse')
        if loss_type.lower() == 'l1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()

    def forward(self, x, x_hat):
        return self.criterion(x_hat, x)

class SupConLoss(BaseLoss):
    """
    Supervised Contrastive Loss.
    Kéo các mẫu cùng danh tính lại gần nhau, đẩy các mẫu khác danh tính ra xa trên không gian cầu.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        # Lấy temperature từ config, mặc định 0.1
        self.temperature = config.get('temperature', 0.1)

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # 1. Chuẩn hóa L2 để đưa features lên unit sphere (không gian cầu)
        features = F.normalize(features, p=2, dim=1)
        
        # 2. Tính ma trận cosine similarity và chia cho temperature
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 3. Tạo mask cho các mẫu cùng label (Positive pairs)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 4. Loại bỏ phần đường chéo chính (Self-contrast) để không so sánh mẫu với chính nó
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # 5. Xử lý numerical stability (tránh overflow khi tính exp)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # 6. Tính log_prob = log( exp(logits) / sum(exp(logits)) )
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # 7. Tính mean log-likelihood over positive
        num_positives = mask.sum(1)
        valid_anchors = num_positives > 0 # Chỉ xét các anchor có ít nhất 1 positive sample
        
        if not valid_anchors.any():
            # Nếu trong batch toàn các danh tính riêng biệt (không có cặp dương nào)
            return torch.tensor(0.0, requires_grad=True, device=device)
            
        mean_log_prob_pos = (mask[valid_anchors] * log_prob[valid_anchors]).sum(1) / num_positives[valid_anchors]
        
        # 8. Contrastive loss = -mean_log_prob_pos.mean()
        loss = -mean_log_prob_pos.mean()
        return loss
