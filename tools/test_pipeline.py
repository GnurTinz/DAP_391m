import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import random
import csv
from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets import DatasetFactory
from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.engine.represent import optimize_r_in_projected_space
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_dir, experiment_name):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logger = logging.getLogger('TestPipelineLogger')
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger, log_file, timestamp

def load_models(config, device, logger, checkpoint_path=None):
    logger.info("Initializing models...")
    
    model_config = config.get('model', {})
    if 'decoder' not in model_config:
        model_config['decoder'] = {}
    model_config['decoder']['image_size'] = config.get('dataset', {}).get('image_size', [128, 128])
    
    model_type = model_config.get('type', 'probabilistic')
    if model_type == 'unet':
        palm_model = UNetPalmModel(model_config).to(device)
    else:
        palm_model = ProbabilisticPalmModel(model_config).to(device)
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'palm_model_state_dict' in checkpoint:
            state_dict = checkpoint['palm_model_state_dict']
        elif 'state_dict' in checkpoint: # PyTorch Lightning checkpoint
            state_dict = checkpoint['state_dict']
            # Xóa prefix 'model.' nếu có (do LightningModule thường wrap model bên trong)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            # Fallback nếu không có prefix model.
            if len(state_dict) == 0:
                 state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        try:
            palm_model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded weights.")
        except Exception as e:
            logger.warning(f"Could not load palm_model weights: {e}")
    else:
        logger.warning("No valid checkpoint provided. Models are initialized randomly.")
        
    palm_model.eval()
    return palm_model

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
            
            # Auto-load config_backup.yaml to match the model shape exactly
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
                    print(f"Successfully loaded model settings from {backup_cfg_path}")
                except Exception as e:
                    print(f"Warning: failed to load {backup_cfg_path}: {e}")
            
    output_dir = os.path.join(version_dir, 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    
    if 'testing' not in config:
        config['testing'] = {
            'verify_threshold': 0.5,
            'margin': 0.05,
            'max_uncertainty': 2.0,
            'use_uncertainty': True,
            'gallery_loss': 'bce',
            'gallery_samples': 256
        }

    logger, log_file, timestamp = setup_logger(output_dir, 'test_pipeline')
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
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Image Size: {config.get('dataset', {}).get('image_size', 'Unknown')}")
    logger.info("==============================")
    
    if dataset_name.upper() == 'POLYU':
        dataset_name = 'PalmPrintDataset'
    elif dataset_name.upper() == 'MNIST':
        dataset_name = 'MNISTDataset'
        
    try:
        # Gallery từ tập Train
        train_dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=config.get('dataset', {}), is_train=True)
        # Probe từ tập Test/Val
        test_dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=config.get('dataset', {}), is_train=False)
    except ValueError as e:
        logger.error(str(e))
        return

    gallery_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    query_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    csv_path = os.path.join(output_dir, f"test_results_{timestamp}.csv")
    csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Query_ID', 'True_Label', 'Predicted_Label', 'Best_Score', 'Uncertainty', 'Decision', 'Reason', 'Is_Correct'])
    
    from tools.build_gallery import build_gallery
    from tools.evaluate_probe import evaluate_probe
    
    use_saved = config.get('testing', {}).get('use_saved_gallery', False)
    gallery_save_path = os.path.join(output_dir, 'gallery.pt')
    
    if use_saved and os.path.isfile(gallery_save_path):
        logger.info(f"Loading saved gallery from {gallery_save_path}")
        gallery = torch.load(gallery_save_path, map_location=device)
    else:
        gallery = build_gallery(palm_model, gallery_loader, device, logger, config, output_dir=output_dir)
        torch.save(gallery, gallery_save_path)
        logger.info(f"Gallery saved successfully to {gallery_save_path}")
        
    evaluate_probe(palm_model, gallery, query_loader, device, logger, config, csv_writer)
    
    csv_file.close()
    logger.info("Test pipeline finished successfully.")

if __name__ == '__main__':
    main()
