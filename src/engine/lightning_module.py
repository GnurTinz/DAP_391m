import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.losses.custom import KLDivLoss, ReconstructionLoss, UncertaintyLoss, get_contrastive_loss
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
        
        self.use_contrastive = loss_cfg.get('use_contrastive', True)
        if self.use_contrastive:
            # Tự động lấy số chiều chiếu (proj_dim) truyền vào ArcFace
            proj_dim = model_config.get('projector', {}).get('proj_dim', 128)
            if not model_config.get('projector', {}).get('use_mlp', True):
                proj_dim = model_config.get('encoder', {}).get('latent_dim', 128)
                
            if 'arcface' not in loss_cfg:
                loss_cfg['arcface'] = {}
            # Ép embedding_size của ArcFace theo cấu hình model (ví dụ 32) thay vì 512 mặc định
            loss_cfg['arcface']['embedding_size'] = proj_dim
            
            # Ép num_classes của ArcFace theo cấu hình dataset
            num_classes = config.get('dataset', {}).get('num_classes', 100)
            loss_cfg['arcface']['num_classes'] = num_classes
            
            self.contrastive_loss = get_contrastive_loss(loss_cfg)
            print(f"🚀 Sử dụng hàm mất mát đẩy/kéo (Metric Learning): {self.contrastive_loss.__class__.__name__} (dim={proj_dim}, classes={num_classes}) 🚀")
        else:
            self.contrastive_loss = None
            print("⚠️ Contrastive Loss đã bị TẮT theo cấu hình YAML. ⚠️")
        # 3. Khởi tạo Schedulers & Default Weights
        self.loss_scheduler = LossSchedulerManager(config.get('loss_schedules', {}))
        
        self.beta_kl = loss_cfg.get('beta_kl', 0.01)
        self.lambda_rec = loss_cfg.get('lambda_rec', 1.0)
        self.lambda_unc = loss_cfg.get('lambda_unc', 0.1)
        self.lambda_con = loss_cfg.get('lambda_con', 0.5)
        
        # Cung cấp tensor mẫu để PyTorch Lightning ghi cấu trúc mạng ra TensorBoard
        img_size = config.get('dataset', {}).get('image_size', [128, 128])
        self.example_input_array = torch.randn(1, 3, img_size[0], img_size[1])

    def forward(self, x, decode=True, sampling_strategy='reconstruction'):
        # Lấy thông số từ YAML config theo strategy
        sampling_cfg = self.config.get('sampling', {}).get(sampling_strategy, {})
        mode = sampling_cfg.get('mode', 'stochastic')
        temperature = sampling_cfg.get('temperature', 1.0)

        # Cấu hình Curriculum Learning
        switch_to_z_step = self.config.get('loss_schedules', {}).get('switch_to_z_step', None)

        # Đổ cấu hình xuống model
        if 'global_step' in self.model.forward.__code__.co_varnames:
            out = self.model(x, decode=decode, temperature=temperature, sample_mode=mode, 
                             global_step=self.global_step, switch_to_z_step=switch_to_z_step)
        elif 'sample_mode' in self.model.forward.__code__.co_varnames:
            out = self.model(x, decode=decode, temperature=temperature, sample_mode=mode)
        elif 'temperature' in self.model.forward.__code__.co_varnames:
            out = self.model(x, decode=decode, temperature=temperature)
        else:
            out = self.model(x, decode=decode)
        import torch
        if torch.jit.is_tracing():
            # Khi TensorBoard Tracer quét qua, ta ép trả về một tuple thuần tuý để tránh lỗi Dict
            # Trả về mu, logvar, proj, x_hat
            return out.get('mu'), out.get('logvar'), out.get('proj'), out.get('x_hat')
        return out

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
        # Quá trình train bắt buộc dùng strategy='training' để chuẩn phân phối (stochastic, T=1.0)
        outputs = self(images, decode=True, sampling_strategy='training')
        
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
        if self.use_contrastive and 'proj' in outputs and self.contrastive_loss is not None:
            con = self.contrastive_loss(outputs['proj'], labels)
            total_loss += self.lambda_con * con
            
            # Nếu hàm loss có sinh ra logits (vd: ArcFaceLoss), ta tính thêm Accuracy
            if hasattr(self.contrastive_loss, 'last_logits'):
                logits = self.contrastive_loss.last_logits
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean()
                prog_bar_details = self.config.get('logging', {}).get('prog_bar_details', True)
                self.log(f'{stage}/Accuracy', acc, prog_bar=prog_bar_details, on_step=True, on_epoch=True)
            
        # Cấu hình hiển thị Progress Bar (Colab thường cần gọn gàng)
        prog_bar_details = self.config.get('logging', {}).get('prog_bar_details', True)
        
        # Logging
        self.log(f'{stage}/Total_Loss', total_loss, prog_bar=prog_bar_details, on_step=True, on_epoch=True)
        self.log(f'{stage}/Recon_Loss', rec, prog_bar=prog_bar_details, on_step=True, on_epoch=True)
        self.log(f'{stage}/KL_Loss', kl, prog_bar=prog_bar_details, on_step=True, on_epoch=True)
        self.log(f'{stage}/Con_Loss', con, prog_bar=prog_bar_details, on_step=True, on_epoch=True)
        
        if batch_idx == 0:
            if 'x_hat' in outputs:
                log_images = (images[:4] + 1) / 2.0
                log_xhat = (outputs['x_hat'][:4] + 1) / 2.0
                try:
                    self.logger.experiment.add_images(f'{stage}/Original', log_images, self.current_epoch)
                    self.logger.experiment.add_images(f'{stage}/Reconstructed', log_xhat, self.current_epoch)
                except Exception:
                    pass
                
                # Lưu file ảnh vật lý ra ổ cứng
                import os
                import torchvision.utils as vutils
                version_dir = self.logger.log_dir if self.logger else "logs/unversioned_results"
                img_dir = os.path.join(version_dir, "epoch_samples")
                os.makedirs(img_dir, exist_ok=True)
                
                # Ghép ảnh gốc (dòng trên) và ảnh tái tạo (dòng dưới)
                comparison = torch.cat([log_images, log_xhat], dim=0)
                vutils.save_image(comparison, os.path.join(img_dir, f"{stage}_epoch_{self.current_epoch:03d}.png"), nrow=4)
                    
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
        elif scheduler_name == 'StepLR':
            step_size = train_cfg.get('scheduler_step_size', 10)
            gamma = train_cfg.get('scheduler_gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            return [optimizer], [scheduler]
        elif scheduler_name == 'MultiStepLR':
            milestones = train_cfg.get('scheduler_milestones', [15, 25])
            gamma = train_cfg.get('scheduler_gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            return [optimizer], [scheduler]
            
        return optimizer
