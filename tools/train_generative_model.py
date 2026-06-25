import os
import argparse
import yaml
import logging
import ssl
from datetime import datetime

# Bypass SSL verification for downloading datasets on Windows
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets.palm_dataset import PalmPrintDataset
from src.datasets.mnist_dataset import MNISTDataset
from src.datasets.sampler import get_sampler
from src.models.palm_model import ProbabilisticPalmModel
from src.engine.trainer import Trainer
from src.losses.custom import SupConLoss
from src.engine.loss_scheduler import LossSchedulerManager

class GenerativeTrainer(Trainer):
    """
    Kế thừa từ Trainer để train riêng VAE + Contrastive Loss.
    """
    def __init__(self, model, train_loader, config, logger=None, writer=None, device='cpu', ckpt_prefix='vae_con_model'):
        super().__init__(model, train_loader, config, logger)
        self.writer = writer
        self.device = device
        self.ckpt_prefix = ckpt_prefix
        
        # Khởi tạo thêm Contrastive Loss
        self.supcon_loss = SupConLoss(config.get('losses', {}))
        self.lambda_con = config.get('losses', {}).get('lambda_con', 0.5)
        
        # Scheduler
        self.loss_scheduler = LossSchedulerManager(config.get('loss_schedules', {}))
        
        # Đảm bảo model lên đúng device
        self.model = self.model.to(self.device)
        self.start_epoch = 0
        self.best_loss = float('inf')

    def train(self):
        self.model.train()
        best_loss = self.best_loss
        os.makedirs(os.path.join('logs', 'checkpoints'), exist_ok=True)
        log_interval = self.config.get('training', {}).get('log_interval', 10)
        save_interval = self.config.get('training', {}).get('save_interval', 1)
        
        for epoch in range(self.start_epoch, self.epochs):
            # Update loss weights for the epoch if any scheduler only uses epoch
            weights = self.loss_scheduler.get_weights(epoch)
            if 'beta_kl' in weights: self.beta_kl = weights['beta_kl']
            if 'lambda_rec' in weights: self.lambda_rec = weights['lambda_rec']
            if 'lambda_con' in weights: self.lambda_con = weights['lambda_con']
            if 'lambda_unc' in weights: self.lambda_unc = weights['lambda_unc']
            
            if self.logger:
                self.logger.info(f"--- Epoch {epoch+1}/{self.epochs} Loss Weights ---")
                self.logger.info(f"beta_kl: {self.beta_kl:.4f}, lambda_rec: {self.lambda_rec:.4f}, "
                                 f"lambda_con: {self.lambda_con:.4f}, lambda_unc: {self.lambda_unc:.4f}")
                                 
            epoch_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}]")
            for batch_idx, (images, labels) in enumerate(pbar):
                if epoch == 0 and batch_idx == 0:
                    if self.logger:
                        self.logger.info(f"--- First Batch Info ---")
                        self.logger.info(f"x (Images) shape: {images.shape}, dtype: {images.dtype}")
                        self.logger.info(f"y (Labels) shape: {labels.shape}, dtype: {labels.dtype}")
                        self.logger.info(f"x min: {images.min().item():.4f}, max: {images.max().item():.4f}")
                        self.logger.info(f"------------------------")
                        
                images, labels = images.to(self.device), labels.to(self.device)
                
                global_step = epoch * len(self.train_loader) + batch_idx
                
                # Update weights for this step
                step_weights = self.loss_scheduler.get_weights(epoch, global_step)
                if 'beta_kl' in step_weights: self.beta_kl = step_weights['beta_kl']
                if 'lambda_rec' in step_weights: self.lambda_rec = step_weights['lambda_rec']
                if 'lambda_con' in step_weights: self.lambda_con = step_weights['lambda_con']
                if 'lambda_unc' in step_weights: self.lambda_unc = step_weights['lambda_unc']

                self.optimizer.zero_grad()
                
                # Forward
                outputs = self.model(images, decode=True)
                mu = outputs['mu']
                logvar = outputs['logvar']
                
                # 1. Các loss cơ bản của VAE từ lớp cha
                kl = self.kl_loss(mu, logvar)
                unc = self.unc_loss(logvar)
                
                total_loss = self.beta_kl * kl + self.lambda_unc * unc
                
                rec = torch.tensor(0.0, device=self.device)
                if 'x_hat' in outputs:
                    rec = self.rec_loss(images, outputs['x_hat'])
                    total_loss += self.lambda_rec * rec
                    
                # 2. Thêm SupConLoss (Contrastive) tính trên output của Light MLP
                proj = outputs['proj']
                con = self.supcon_loss(proj, labels)
                total_loss += self.lambda_con * con
                
                # Backward
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                
                pbar.set_postfix({
                    'Total': f"{total_loss.item():.4f}",
                    'Rec': f"{rec.item():.4f}",
                    'KL': f"{kl.item():.4f}",
                    'Con': f"{con.item():.4f}"
                })
                
                # Logging theo log_interval
                if batch_idx % log_interval == 0:
                    if self.logger:
                        self.logger.debug(f"Epoch [{epoch+1}/{self.epochs}], Batch [{batch_idx}/{len(self.train_loader)}] "
                                          f"Total Loss: {total_loss.item():.4f}, Recon: {rec.item():.4f}, "
                                          f"KL: {kl.item():.4f}, Con: {con.item():.4f}")
                    if self.writer:
                        self.writer.add_scalar('Batch/Total_Loss', total_loss.item(), global_step)
                        self.writer.add_scalar('Batch/Recon_Loss', rec.item(), global_step)
                        self.writer.add_scalar('Batch/KL_Loss', kl.item(), global_step)
                        self.writer.add_scalar('Batch/Con_Loss', con.item(), global_step)
                        
                        # Log the running lambdas
                        self.writer.add_scalar('Weights/beta_kl', self.beta_kl, global_step)
                        self.writer.add_scalar('Weights/lambda_rec', self.lambda_rec, global_step)
                        self.writer.add_scalar('Weights/lambda_con', self.lambda_con, global_step)
                        self.writer.add_scalar('Weights/lambda_unc', self.lambda_unc, global_step)

            # Tổng kết trung bình mỗi epoch
            avg_loss = epoch_loss / len(self.train_loader)
            if self.logger:
                self.logger.info(f"Epoch [{epoch+1}/{self.epochs}] | Avg Loss: {avg_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{self.epochs}] | Avg Loss: {avg_loss:.4f}")
                
            if self.writer:
                self.writer.add_scalar('Epoch/Total_Loss', avg_loss, epoch)
                if 'x_hat' in outputs:
                    self.writer.add_images('Epoch/Original_Images', images[:4], epoch)
                    self.writer.add_images('Epoch/Reconstructed_Images', outputs['x_hat'][:4], epoch)

            # Checkpoint best
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join('logs', 'checkpoints', f"{self.ckpt_prefix}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': best_loss
                }, best_path)
                if self.logger:
                    self.logger.info(f"Saved new best model with loss: {best_loss:.4f}")
            
            # Checkpoint last
            if (epoch + 1) % save_interval == 0 or epoch == self.epochs - 1:
                last_path = os.path.join('logs', 'checkpoints', f"{self.ckpt_prefix}_last.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': best_loss
                }, last_path)


def setup_logging(config):
    log_dir = config.get('logging', {}).get('log_dir', 'logs/experiments')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('logging', {}).get('experiment_name', 'vae_train')
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logger = logging.getLogger('VAETrainLogger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger, log_dir, timestamp, experiment_name

def main():
    parser = argparse.ArgumentParser(description="Train VAE (Generative Model) + Contrastive Loss")
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger, log_dir, timestamp, exp_name = setup_logging(config)
    logger.info(f"Loaded config from: {args.config}")
    
    tb_dir = os.path.join('logs', 'tensorboard', f"{exp_name}_{timestamp}")
    if config.get('logging', {}).get('enable_tensorboard', True):
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        writer = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Dataset
    data_dir = config.get('dataset', {}).get('data_dir', 'data/PolyU')
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers = config.get('training', {}).get('num_workers', 4)
    
    dataset_name = config.get('dataset', {}).get('name', 'PolyU')
    if dataset_name.upper() == 'MNIST':
        train_dataset = MNISTDataset(data_dir=data_dir, config=config.get('dataset', {}), is_train=True)
    elif dataset_name.upper() == 'POLYU':
        train_dataset = PalmPrintDataset(data_dir=data_dir, config=config.get('dataset', {}), is_train=True)
    else:
        raise ValueError(f"Dataset {dataset_name} không được hỗ trợ!")

    # Sampler setup
    use_sampler = config.get('training', {}).get('use_sampler', True)
    if use_sampler:
        sampler_type = config.get('training', {}).get('sampler_type', 'pk_sampler')
        p = config.get('training', {}).get('sampler_p', None)
        k = config.get('training', {}).get('sampler_k', None)
        
        try:
            sampler = get_sampler(
                sampler_type=sampler_type, 
                labels=train_dataset.get_labels(), 
                batch_size=batch_size, 
                p=p, k=k
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_sampler=sampler,
                num_workers=num_workers
            )
            logger.info(f"Đang sử dụng sampler: {sampler_type}")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo sampler: {e}. Chuyển về DataLoader mặc định.")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )

    # Model
    model_config = config.get('model', {})
    if 'decoder' not in model_config:
        model_config['decoder'] = {}
    model_config['decoder']['image_size'] = config.get('dataset', {}).get('image_size', [128, 128])
    model = ProbabilisticPalmModel(model_config)
    model.use_decoder = True 
    
    logger.info("=== MODEL ARCHITECTURE ===")
    logger.info(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    
    # Train qua GenerativeTrainer (Kế thừa từ Base Trainer)
    ckpt_prefix = f"vae_con_model_{exp_name}_{timestamp}"
    trainer = GenerativeTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        logger=logger,
        writer=writer,
        device=device,
        ckpt_prefix=ckpt_prefix
    )
    
    # Load checkpoint for resume if provided
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint '{args.resume}' to resume training...")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.start_epoch = checkpoint['epoch'] + 1
            trainer.best_loss = checkpoint.get('best_loss', float('inf'))
            logger.info(f"Resumed from epoch {checkpoint['epoch']} with best_loss {trainer.best_loss:.4f}")
        else:
            trainer.model.load_state_dict(checkpoint)
            logger.info("Loaded legacy checkpoint without optimizer state. Starting from epoch 0.")
    
    logger.info("Bắt đầu quá trình huấn luyện VAE + Contrastive...")
    trainer.train()

    logger.info("Training completed.")
    
    if writer:
        writer.close()

if __name__ == '__main__':
    main()
