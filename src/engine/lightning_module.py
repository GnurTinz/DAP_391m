import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.losses.custom import KLDivLoss, ReconstructionLoss, UncertaintyLoss, SupConLoss
from src.engine.loss_scheduler import LossSchedulerManager

class GenerativeLightningModule(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # 1. Khởi tạo Model
        model_config = config.get('model', {})
        if 'decoder' not in model_config:
            model_config['decoder'] = {}
        model_type = model_config.get('type', 'probabilistic')
        
        if model_type == 'unet':
            self.model = UNetPalmModel(model_config)
        else:
            self.model = ProbabilisticPalmModel(model_config)
        self.model.use_decoder = True 
            
        # 2. Khởi tạo Losses
        loss_cfg = config.get('losses', {})
        self.kl_loss = KLDivLoss(loss_cfg)
        self.rec_loss = ReconstructionLoss(loss_cfg)
        self.unc_loss = UncertaintyLoss(loss_cfg)
        self.supcon_loss = SupConLoss(loss_cfg)
        
        # 3. Khởi tạo Schedulers & Default Weights
        self.loss_scheduler = LossSchedulerManager(config.get('loss_schedules', {}))
        
        self.beta_kl = loss_cfg.get('beta_kl', 0.01)
        self.lambda_rec = loss_cfg.get('lambda_rec', 1.0)
        self.lambda_unc = loss_cfg.get('lambda_unc', 0.1)
        self.lambda_con = loss_cfg.get('lambda_con', 0.5)

    def forward(self, x, decode=True):
        return self.model(x, decode=decode)

    def _update_loss_weights(self):
        epoch = self.current_epoch
        global_step = self.global_step
        
        weights = self.loss_scheduler.get_weights(epoch, global_step)
        if 'beta_kl' in weights: self.beta_kl = weights['beta_kl']
        if 'lambda_rec' in weights: self.lambda_rec = weights['lambda_rec']
        if 'lambda_con' in weights: self.lambda_con = weights['lambda_con']
        if 'lambda_unc' in weights: self.lambda_unc = weights['lambda_unc']
        
        self.log('Weights/beta_kl', self.beta_kl, on_step=True)
        self.log('Weights/lambda_rec', self.lambda_rec, on_step=True)
        self.log('Weights/lambda_con', self.lambda_con, on_step=True)
        self.log('Weights/lambda_unc', self.lambda_unc, on_step=True)

    def shared_step(self, batch, batch_idx, stage='train'):
        images, labels = batch
        outputs = self(images, decode=True)
        
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        kl = self.kl_loss(mu, logvar)
        unc = self.unc_loss(logvar)
        total_loss = self.beta_kl * kl + self.lambda_unc * unc
        
        rec = torch.tensor(0.0, device=self.device)
        if 'x_hat' in outputs:
            rec = self.rec_loss(images, outputs['x_hat'])
            total_loss += self.lambda_rec * rec
            
        con = torch.tensor(0.0, device=self.device)
        if 'proj' in outputs:
            con = self.supcon_loss(outputs['proj'], labels)
            total_loss += self.lambda_con * con
            
        # Logging
        self.log(f'{stage}/Total_Loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'{stage}/Recon_Loss', rec, on_step=False, on_epoch=True)
        self.log(f'{stage}/KL_Loss', kl, on_step=False, on_epoch=True)
        self.log(f'{stage}/Con_Loss', con, on_step=False, on_epoch=True)
        
        if batch_idx == 0:
            if 'x_hat' in outputs:
                log_images = (images[:4] + 1) / 2.0
                log_xhat = (outputs['x_hat'][:4] + 1) / 2.0
                try:
                    self.logger.experiment.add_images(f'{stage}/Original', log_images, self.current_epoch)
                    self.logger.experiment.add_images(f'{stage}/Reconstructed', log_xhat, self.current_epoch)
                except Exception:
                    pass
                    
        return total_loss

    def training_step(self, batch, batch_idx):
        self._update_loss_weights()
        return self.shared_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage='val')

    def configure_optimizers(self):
        train_cfg = self.config.get('training', {})
        optimizer_name = train_cfg.get('optimizer', 'AdamW')
        lr = train_cfg.get('learning_rate', 0.0002)
        weight_decay = train_cfg.get('weight_decay', 1e-4)
        
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            
        scheduler_name = train_cfg.get('scheduler', 'CosineAnnealingLR')
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.get('epochs', 30))
            return [optimizer], [scheduler]
            
        return optimizer
