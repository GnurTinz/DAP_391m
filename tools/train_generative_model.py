import os
import argparse
import yaml
import logging
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.palm_dataset import PalmPrintDataset
from src.models.palm_model import ProbabilisticPalmModel
from src.losses.custom import KLDivLoss, ReconstructionLoss

def setup_logging(config):
    """ Thiết lập logging lưu ra file và console """
    log_dir = config.get('logging', {}).get('log_dir', 'logs/experiments')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('logging', {}).get('experiment_name', 'vae_train')
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Cấu hình logger
    logger = logging.getLogger('VAETrainLogger')
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger, log_dir, timestamp, experiment_name

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Train VAE (Generative Model) for PalmPrint")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Đường dẫn tới file config YAML')
    args = parser.parse_args()

    # 2. Load Config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 3. Setup Logging & TensorBoard
    logger, log_dir, timestamp, exp_name = setup_logging(config)
    logger.info(f"Loaded config from: {args.config}")
    
    tb_dir = os.path.join('logs', 'tensorboard', f"{exp_name}_{timestamp}")
    writer = SummaryWriter(log_dir=tb_dir) if config.get('logging', {}).get('enable_tensorboard', True) else None
    if writer:
        logger.info(f"TensorBoard logging enabled at: {tb_dir}")

    # 4. Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 5. Dataset & DataLoader
    data_dir = config.get('dataset', {}).get('data_dir', 'data/PolyU')
    batch_size = config.get('dataset', {}).get('batch_size', 32)
    num_workers = config.get('dataset', {}).get('num_workers', 4)
    
    train_dataset = PalmPrintDataset(data_dir=data_dir, config=config.get('dataset', {}), is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    logger.info(f"Initialized train DataLoader with {len(train_dataset)} samples.")

    # 6. Model Setup (Only VAE parts: Encoder + Decoder)
    model = ProbabilisticPalmModel(config.get('model', {}))
    # Chắc chắn bật cờ use_decoder cho quá trình train VAE
    model.use_decoder = True 
    model = model.to(device)
    logger.info("Initialized ProbabilisticPalmModel (VAE mode).")

    # 7. Optimizer & Losses Setup
    lr = config.get('training', {}).get('learning_rate', 1e-3)
    weight_decay = config.get('training', {}).get('weight_decay', 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    recon_loss_fn = ReconstructionLoss(config.get('losses', {}))
    kl_loss_fn = KLDivLoss(config.get('losses', {}))
    
    lambda_rec = config.get('losses', {}).get('lambda_rec', 1.0)
    beta_kl = config.get('losses', {}).get('beta_kl', 0.01)

    # 8. Training Loop
    epochs = config.get('training', {}).get('epochs', 50)
    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_kl_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: lấy latent và ảnh tái tạo
            out = model(images, decode=True)
            mu, logvar, x_hat = out['mu'], out['logvar'], out['x_hat']
            
            # Tính losses
            l_rec = recon_loss_fn(images, x_hat)
            l_kl = kl_loss_fn(mu, logvar)
            
            total_loss = (lambda_rec * l_rec) + (beta_kl * l_kl)
            
            # Backward
            total_loss.backward()
            optimizer.step()
            
            # Logging
            epoch_loss += total_loss.item()
            epoch_rec_loss += l_rec.item()
            epoch_kl_loss += l_kl.item()

            if writer and batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Batch/Total_Loss', total_loss.item(), global_step)
                writer.add_scalar('Batch/Recon_Loss', l_rec.item(), global_step)
                writer.add_scalar('Batch/KL_Loss', l_kl.item(), global_step)

        # Trung bình loss mỗi epoch
        avg_loss = epoch_loss / len(train_loader)
        avg_rec = epoch_rec_loss / len(train_loader)
        avg_kl = epoch_kl_loss / len(train_loader)
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Recon: {avg_rec:.4f} | KL: {avg_kl:.4f}")
        
        if writer:
            writer.add_scalar('Epoch/Total_Loss', avg_loss, epoch)
            writer.add_scalar('Epoch/Recon_Loss', avg_rec, epoch)
            writer.add_scalar('Epoch/KL_Loss', avg_kl, epoch)
            
            # Ghi nhận hình ảnh (original vs reconstructed) vào cuối epoch
            writer.add_images('Epoch/Original_Images', images[:4], epoch)
            writer.add_images('Epoch/Reconstructed_Images', x_hat[:4], epoch)

    # 9. Save Model
    os.makedirs('logs/checkpoints', exist_ok=True)
    ckpt_path = os.path.join('logs', 'checkpoints', f"vae_model_{exp_name}_{timestamp}.pth")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Training completed. Model saved at: {ckpt_path}")
    
    if writer:
        writer.close()

if __name__ == '__main__':
    main()
