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
        self.criterion = nn.L1Loss()

    def forward(self, x, x_hat):
        return self.criterion(x_hat, x)

class SupConLoss(BaseLoss):
    """
    Supervised Contrastive Loss wrapper (simplified mock for skeleton).
    In practice, implement full SupCon logic.
    """
    def forward(self, features, labels):
        # Mock implementation returning scalar tensor
        return torch.tensor(0.0, requires_grad=True, device=features.device)
