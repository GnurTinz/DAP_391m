import os
import argparse
import yaml
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.palm_dataset import PalmPrintDataset
from src.datasets.mnist_dataset import MNISTDataset
from src.models.palm_model import ProbabilisticPalmModel
from src.engine.trainer import Trainer
from src.losses.custom import SupConLoss

class GenerativeTrainer(Trainer):
    """
    Kế thừa từ Trainer để train riêng VAE + Contrastive Loss.
    """
    def __init__(self, model, train_loader, config, logger=None, writer=None, device='cpu'):
        super().__init__(model, train_loader, config, logger)
        self.writer = writer
        self.device = device
        
        # Khởi tạo thêm Contrastive Loss
        self.supcon_loss = SupConLoss(config.get('losses', {}))
        self.lambda_con = config.get('losses', {}).get('lambda_con', 0.5)
        
        # Đảm bảo model lên đúng device
        self.model = self.model.to(self.device)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
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
                global_step = epoch * len(self.train_loader) + batch_idx
                
                # Logging mỗi 10 batch
                if batch_idx % 10 == 0:
                    if self.writer:
                        self.writer.add_scalar('Batch/Total_Loss', total_loss.item(), global_step)
                        self.writer.add_scalar('Batch/Recon_Loss', rec.item(), global_step)
                        self.writer.add_scalar('Batch/KL_Loss', kl.item(), global_step)
                        self.writer.add_scalar('Batch/Con_Loss', con.item(), global_step)

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


def setup_logging(config):
    log_dir = config.get('logging', {}).get('log_dir', 'logs/experiments')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('logging', {}).get('experiment_name', 'vae_train')
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logger = logging.getLogger('VAETrainLogger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger, log_dir, timestamp, experiment_name

def main():
    parser = argparse.ArgumentParser(description="Train VAE (Generative Model) + Contrastive Loss")
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger, log_dir, timestamp, exp_name = setup_logging(config)
    logger.info(f"Loaded config from: {args.config}")
    
    tb_dir = os.path.join('logs', 'tensorboard', f"{exp_name}_{timestamp}")
    writer = SummaryWriter(log_dir=tb_dir) if config.get('logging', {}).get('enable_tensorboard', True) else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Dataset
    data_dir = config.get('dataset', {}).get('data_dir', 'data/PolyU')
    batch_size = config.get('dataset', {}).get('batch_size', 32)
    num_workers = config.get('dataset', {}).get('num_workers', 4)
    
    dataset_name = config.get('dataset', {}).get('name', 'PolyU')
    if dataset_name.upper() == 'MNIST':
        train_dataset = MNISTDataset(data_dir=data_dir, config=config.get('dataset', {}), is_train=True)
    else:
        train_dataset = PalmPrintDataset(data_dir=data_dir, config=config.get('dataset', {}), is_train=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Model
    model = ProbabilisticPalmModel(config.get('model', {}))
    model.use_decoder = True 
    
    # Train qua GenerativeTrainer (Kế thừa từ Base Trainer)
    trainer = GenerativeTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        logger=logger,
        writer=writer,
        device=device
    )
    
    logger.info("Bắt đầu quá trình huấn luyện VAE + Contrastive...")
    trainer.train()

    # Save Model
    os.makedirs('logs/checkpoints', exist_ok=True)
    ckpt_path = os.path.join('logs', 'checkpoints', f"vae_con_model_{exp_name}_{timestamp}.pth")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Training completed. Model saved at: {ckpt_path}")
    
    if writer:
        writer.close()

if __name__ == '__main__':
    main()
