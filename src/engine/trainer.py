import torch
from torch.utils.data import DataLoader
from ..models.palm_model import ProbabilisticPalmModel
from ..losses.custom import KLDivLoss, ReconstructionLoss, UncertaintyLoss

class Trainer:
    """
    Main training loop for the Probabilistic PalmPrint Model.
    """
    def __init__(self, model: ProbabilisticPalmModel, train_loader: DataLoader, config: dict):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        
        # Losses
        self.kl_loss = KLDivLoss(config.get('losses', {}))
        self.rec_loss = ReconstructionLoss(config.get('losses', {}))
        self.unc_loss = UncertaintyLoss(config.get('losses', {}))
        
        # Hyperparams
        loss_cfg = config.get('losses', {})
        self.lambda_rec = loss_cfg.get('lambda_rec', 0.1)
        self.beta_kl = loss_cfg.get('beta_kl', 0.01)
        self.lambda_unc = loss_cfg.get('lambda_unc', 0.1)
        
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 100)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=train_cfg.get('learning_rate', 1e-3),
            weight_decay=train_cfg.get('weight_decay', 1e-4)
        )

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                # Forward
                outputs = self.model(images, decode=True)
                mu = outputs['mu']
                logvar = outputs['logvar']
                
                # Calculate losses
                kl = self.kl_loss(mu, logvar)
                unc = self.unc_loss(logvar)
                
                loss = self.beta_kl * kl + self.lambda_unc * unc
                
                if 'x_hat' in outputs:
                    rec = self.rec_loss(images, outputs['x_hat'])
                    loss += self.lambda_rec * rec
                    
                # Add SupCon / Id / Pair losses here based on stage
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                    
            print(f"Epoch {epoch} completed. Average Loss: {total_loss / len(self.train_loader):.4f}")
