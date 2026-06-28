import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import math
from .base import BaseLoss

try:
    from pytorch_metric_learning import losses as pml_losses
except ImportError:
    raise ImportError("Vui lòng cài đặt pytorch-metric-learning: pip install pytorch-metric-learning")

class KLDivLoss(BaseLoss):
    """
    KL Divergence between learned distribution and N(0, I).
    """
    def forward(self, mu, logvar):
        
        # Phân phối học được q(z|x)
        std = torch.exp(0.5 * logvar)
        q = Normal(mu, std)
        
        # Phân phối chuẩn mục tiêu p(z) = N(0, 1)
        p = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        
        # KL(q || p)
        # Sum theo chiều features, sau đó lấy mean theo batch size
        kld = kl_divergence(q, p).sum(dim=-1).mean()
        return kld

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
    Supervised Contrastive Loss using pytorch-metric-learning.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.temperature = config.get('temperature', 0.1)
        self.loss_fn = pml_losses.SupConLoss(temperature=self.temperature)

    def forward(self, features, labels):
        # normalize features (SupConLoss in pml expects unnormalized, but standard practice normalizes)
        features = F.normalize(features, p=2, dim=1)
        return self.loss_fn(features, labels)

class MultiSimilarityLoss(BaseLoss):
    """
    Multi-Similarity Loss using pytorch-metric-learning.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        alpha = config.get('ms_alpha', 2.0)
        beta = config.get('ms_beta', 50.0)
        base = config.get('ms_base', 0.5)
        self.loss_fn = pml_losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        return self.loss_fn(features, labels)

class InfoNCELoss(BaseLoss):
    """
    InfoNCE Loss (NT-Xent) using pytorch-metric-learning.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.temperature = config.get('temperature', 0.1)
        self.loss_fn = pml_losses.NTXentLoss(temperature=self.temperature)

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        return self.loss_fn(features, labels)

class ArcFaceLoss(BaseLoss):
    """
    ArcFace Loss cho Metric Learning sử dụng pytorch-metric-learning.
    Tự động tính và lưu trữ logits để tương thích với engine cũ.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        arcface_cfg = config.get('arcface', {})
        s = arcface_cfg.get('s', 30.0)
        m = arcface_cfg.get('m', 0.50)
        
        num_classes = arcface_cfg.get('num_classes', 100)
        embedding_size = arcface_cfg.get('embedding_size', 512)
        
        self.loss_fn = pml_losses.ArcFaceLoss(
            num_classes=num_classes, 
            embedding_size=embedding_size, 
            margin=math.degrees(m), # pml arcface margin is in degrees
            scale=s
        )

    def forward(self, features, labels):
        loss = self.loss_fn(features, labels)
        
        # Mô phỏng lại last_logits để truyền ra ngoài cho Accuracy metric
        with torch.no_grad():
            W = self.loss_fn.W
            cosine = F.linear(F.normalize(features, p=2, dim=1), F.normalize(W, p=2, dim=0).T)
            self.last_logits = cosine * self.loss_fn.scale
            
        return loss

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

