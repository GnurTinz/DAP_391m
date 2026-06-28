import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import DatasetFactory
from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.engine.represent import optimize_r_in_projected_space
from tools.test_pipeline import set_seed, setup_logger, load_models

def build_gallery(palm_model, dataloader, device, logger, config, output_dir="."):
    """
    Trích xuất mu, logvar từ tập Train (hoặc load từ cache). 
    Tối ưu r trong không gian Projected Vector cho từng ảnh.
    """
    use_saved_features = config.get('testing', {}).get('use_saved_features', False)
    features_save_path = os.path.join(output_dir, 'extracted_features.pt')
    
    if use_saved_features and os.path.isfile(features_save_path):
        logger.info(f"Loading extracted features from {features_save_path}...")
        features = torch.load(features_save_path, map_location=device)
        db_mu = features['db_mu']
        db_logvar = features['db_logvar']
        db_labels = features['db_labels']
    else:
        logger.info("Extracting Train features from dataset...")
        db_mu = []
        db_logvar = []
        db_labels = []
        
        palm_model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting Train Features"):
                images = images.to(device)
                outputs = palm_model(images, decode=False)
                db_mu.append(outputs['mu'].cpu())
                db_logvar.append(outputs['logvar'].cpu())
                db_labels.append(labels.cpu())
                
        db_mu = torch.cat(db_mu, dim=0).to(device)
        db_logvar = torch.cat(db_logvar, dim=0).to(device)
        db_labels = torch.cat(db_labels, dim=0).to(device)
        
        # Lưu lại để lần sau tái sử dụng
        torch.save({
            'db_mu': db_mu,
            'db_logvar': db_logvar,
            'db_labels': db_labels
        }, features_save_path)
        logger.info(f"Features saved to {features_save_path}")
    
    unique_labels = torch.unique(db_labels)
    gallery = {}
    
    loss_type = config.get('testing', {}).get('gallery_loss', 'bce')
    num_samples = config.get('testing', {}).get('gallery_samples', 256)
    max_steps = config.get('testing', {}).get('gallery_max_steps', 100)
    lr = config.get('testing', {}).get('gallery_lr', 0.01)
    num_workers = config.get('testing', {}).get('gallery_workers', 1)
    
    logger.info(f"Optimizing R using {loss_type.upper()} loss (Max steps: {max_steps}, LR: {lr}, Workers: {num_workers})...")
    
    def process_label(label):
        idx = (db_labels == label).nonzero(as_tuple=True)[0]
        idx_others = (db_labels != label).nonzero(as_tuple=True)[0]
        
        mu_c = db_mu[idx]
        logvar_c = db_logvar[idx]
        mu_others = db_mu[idx_others]
        logvar_others = db_logvar[idx_others]
        
        max_images = config.get('testing', {}).get('gallery_max_images', 8)
        limit = min(len(mu_c), max_images)
        
        r_cs = []
        for i in range(limit):
            r_c = optimize_r_in_projected_space(
                mu_c[i:i+1], logvar_c[i:i+1], mu_others, logvar_others, 
                model=palm_model, device=device, config=config, 
                num_samples=num_samples, loss_type=loss_type, max_steps=max_steps, lr=lr,
                verbose=(num_workers == 1)
            )
            r_cs.append(r_c)
            
        return label.item(), torch.stack(r_cs) # shape [N_images, proj_dim]

    import sys
    try:
        if num_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_label, label): label for label in unique_labels}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Optimizing Gallery R (Parallel)"):
                    label_item, r_stack = future.result()
                    gallery[label_item] = r_stack
        else:
            for label in tqdm(unique_labels, desc="Optimizing Gallery R"):
                label_item, r_stack = process_label(label)
                gallery[label_item] = r_stack
    except KeyboardInterrupt:
        logger.warning("\n[!] Bị gián đoạn bởi người dùng (Ctrl+C). Dừng tiến trình ngay lập tức.")
        os._exit(1)
        
    logger.info(f"Database built with {len(gallery)} unique identities.")
    return gallery

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    seed = config.get('seed', 42)
    checkpoint_path = config.get('checkpoint', '')
    
    import re
    version_dir = "logs/unversioned_results"
    if checkpoint_path:
        match = re.search(r'(.*[\\/]version_\d+)', checkpoint_path.replace('\\', '/'))
        if match:
            version_dir = match.group(1)
            
            # Auto-load config_backup.yaml
            backup_cfg_path = os.path.join(version_dir, 'config_backup.yaml')
            if os.path.isfile(backup_cfg_path):
                try:
                    backup_cfg = OmegaConf.load(backup_cfg_path)
                    if 'model' in backup_cfg:
                        config['model'] = OmegaConf.to_container(backup_cfg['model'], resolve=True)
                    if 'dataset' in backup_cfg and 'image_size' in backup_cfg['dataset']:
                        if 'dataset' not in config:
                            config['dataset'] = {}
                        config['dataset']['image_size'] = backup_cfg['dataset']['image_size']
                except Exception as e:
                    pass

    output_dir = os.path.join(version_dir, 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    
    if 'testing' not in config:
        config['testing'] = {
            'gallery_loss': 'bce',
            'gallery_samples': 256
        }

    logger, log_file, timestamp = setup_logger(output_dir, 'build_gallery')
    logger.info(f"Logs will be saved to: {log_file}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    palm_model = load_models(config, device, logger, checkpoint_path)
    
    data_dir = config.get('dataset', {}).get('data_dir', 'data/PolyU')
    dataset_name = config.get('dataset', {}).get('name', 'PolyU')
    batch_size = config.get('testing', {}).get('batch_size', 32)
    num_workers = config.get('testing', {}).get('num_workers', 2)
    
    logger.info("=== Dataset Configuration ===")
    logger.info(f"Dataset Name: {dataset_name}")
    logger.info(f"Data Directory: {data_dir}")
    
    if dataset_name.upper() == 'POLYU':
        dataset_name = 'PalmPrintDataset'
    elif dataset_name.upper() == 'MNIST':
        dataset_name = 'MNISTDataset'
        
    try:
        # Gallery từ tập Train
        train_dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=config.get('dataset', {}), is_train=True)
    except ValueError as e:
        logger.error(str(e))
        return

    gallery_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    gallery = build_gallery(palm_model, gallery_loader, device, logger, config, output_dir=output_dir)
    
    # Save gallery
    gallery_save_path = os.path.join(output_dir, 'gallery.pt')
    torch.save(gallery, gallery_save_path)
    logger.info(f"Gallery saved successfully to {gallery_save_path}")

if __name__ == '__main__':
    main()
