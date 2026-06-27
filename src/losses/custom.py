import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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

class MultiSimilarityLoss(BaseLoss):
    """
    Multi-Similarity Loss with Self-Miner.
    Push and pull based on absolute similarity pairs with strong weighting.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.alpha = config.get('ms_alpha', 2.0)
        self.beta = config.get('ms_beta', 50.0)
        self.base = config.get('ms_base', 0.5)

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # L2 normalize
        features = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(features, features.T)
        
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        mask_neg = 1.0 - mask_pos
        
        # Loại bỏ đường chéo
        logits_mask = torch.ones_like(mask_pos) - torch.eye(batch_size, device=device)
        mask_pos = mask_pos * logits_mask
        
        loss = []
        for i in range(batch_size):
            pos_sims = sim_matrix[i][mask_pos[i].bool()]
            neg_sims = sim_matrix[i][mask_neg[i].bool()]
            
            if len(pos_sims) == 0 or len(neg_sims) == 0:
                continue
                
            # Hard mining using MS Loss weighting
            pos_loss = (1.0 / self.alpha) * torch.log(1 + torch.sum(torch.exp(-self.alpha * (pos_sims - self.base))))
            neg_loss = (1.0 / self.beta) * torch.log(1 + torch.sum(torch.exp(self.beta * (neg_sims - self.base))))
            
            loss.append(pos_loss + neg_loss)
            
        if len(loss) == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)
            
        return torch.stack(loss).mean()

class InfoNCELoss(BaseLoss):
    """
    InfoNCE Loss (Tương đương NT-Xent loss).
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.temperature = config.get('temperature', 0.1)

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        features = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos[mask.sum(1) > 0].mean()
        
        if torch.isnan(loss) or mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)
            
        return loss

class ArcFaceLoss(BaseLoss):
    """
    ArcFace Loss cho Metric Learning.
    Yêu cầu config['arcface'] phải có 'num_classes' và 'embedding_size'.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        arcface_cfg = config.get('arcface', {})
        self.s = arcface_cfg.get('s', 30.0)
        self.m = arcface_cfg.get('m', 0.50)
        
        self.num_classes = arcface_cfg.get('num_classes', 100)
        self.embedding_size = arcface_cfg.get('embedding_size', 512)
        
        self.weight = nn.Parameter(torch.FloatTensor(self.num_classes, self.embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.ce = nn.CrossEntropyLoss()
        
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, features, labels):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), 1e-9, 1.0))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return self.ce(output, labels)


def get_contrastive_loss(config: dict):
    loss_type = config.get('contrastive_type', 'supcon')
    if loss_type == 'ms_loss':
        return MultiSimilarityLoss(config)
    elif loss_type == 'infonce':
        return InfoNCELoss(config)
    elif loss_type == 'arcface':
        return ArcFaceLoss(config)
    else:
        return SupConLoss(config)

