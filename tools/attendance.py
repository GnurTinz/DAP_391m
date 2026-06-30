import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
import re
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import DatasetFactory
from tools.utils import set_seed, setup_logger, load_models
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

class Attendant:
    def __init__(self, model, gallery, device, config):
        self.model = model
        self.gallery = gallery
        self.device = device
        self.config = config
        
        self.use_raw_mu = config.get('testing', {}).get('use_raw_mu', True)
        self.optimize_probe = config.get('testing', {}).get('optimize_probe', False)
        
        # Build gallery tensors
        gallery_rs_flat = []
        self.gallery_labels = []
        for k in self.gallery.keys():
            r_c = self.gallery[k]
            if r_c.dim() == 1:
                r_c = r_c.unsqueeze(0)
            for i in range(r_c.size(0)):
                gallery_rs_flat.append(r_c[i])
                self.gallery_labels.append(k)
                
        self.gallery_rs = torch.stack(gallery_rs_flat).to(self.device)
        if self.use_raw_mu:
            self.gallery_rs = torch.nn.functional.normalize(self.gallery_rs, p=2, dim=1)
            self.optimize_probe = False # raw_mu doesnt use optimize_probe currently

    def attend_image(self, image):
        image = image.to(self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image, decode=False)
            mu = outputs['mu']
            
            if not self.optimize_probe:
                if self.use_raw_mu:
                    p_probe = torch.nn.functional.normalize(mu, p=2, dim=1)
                else:
                    p_probe = self.model.projector(mu)
                    p_probe = torch.nn.functional.normalize(p_probe, p=2, dim=1)
            else:
                # Nếu optimize probe thì cần logic phức tạp hơn (tương tự như bài cũ)
                # Tạm thời hỗ trợ non-optimize cho nhanh
                if self.use_raw_mu:
                    p_probe = torch.nn.functional.normalize(mu, p=2, dim=1)
                else:
                    p_probe = self.model.projector(mu)
                    p_probe = torch.nn.functional.normalize(p_probe, p=2, dim=1)
                    
        # Tính cosine similarity với gallery
        scores = torch.mm(p_probe, self.gallery_rs.t()).squeeze(0) # (Total_Templates,)
        
        # Lấy max score cho từng identity
        unique_labels = list(set(self.gallery_labels))
        id_scores = {}
        for lbl in unique_labels:
            indices = [i for i, x in enumerate(self.gallery_labels) if x == lbl]
            id_scores[lbl] = scores[indices].max().item()
            
        best_id = max(id_scores, key=id_scores.get)
        best_score = id_scores[best_id]
        
        # Rank-5
        sorted_ids = sorted(id_scores.keys(), key=lambda x: id_scores[x], reverse=True)
        top5_ids = sorted_ids[:5]
        
        return best_id, best_score, id_scores, top5_ids


def calculate_eer(y_true, y_scores):
    if len(set(y_true)) < 2:
        return 0.0
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100

def process_attendance(palm_model, test_dataset, gallery, device, logger, config):
    attendant = Attendant(palm_model, gallery, device, config)
    
    total = 0
    correct_1 = 0
    correct_5 = 0
    
    y_true = []
    y_scores = []
    
    logger.info("Running attendance evaluation on test dataset...")
    for img, label in tqdm(test_dataset, desc="Evaluating"):
        true_label = label.item() if isinstance(label, torch.Tensor) else label
        
        best_id, best_score, id_scores, top5_ids = attendant.attend_image(img)
        
        total += 1
        if best_id == true_label:
            correct_1 += 1
            
        if true_label in top5_ids:
            correct_5 += 1
            
        # For EER calculation
        for gal_lbl, score in id_scores.items():
            y_true.append(1 if gal_lbl == true_label else 0)
            y_scores.append(score)
            
    acc_1 = (correct_1 / total) * 100
    acc_5 = (correct_5 / total) * 100
    eer = calculate_eer(y_true, y_scores)
    
    logger.info("=== KẾT QUẢ ĐÁNH GIÁ (ATTENDANCE PIPELINE) ===")
    logger.info(f"Total Probes: {total}")
    logger.info(f"Matching Accuracy (Rank-1): {acc_1:.2f}%")
    logger.info(f"Matching Accuracy (Rank-5): {acc_5:.2f}%")
    logger.info(f"Equal Error Rate (EER): {eer:.2f}%")
    logger.info("================================================")
    

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    seed = config.get('seed', 42)
    checkpoint_path = config.get('checkpoint', '')
    
    version_dir = "logs/unversioned_results"
    if checkpoint_path:
        match = re.search(r'(.*[\\/]version_\d+)', checkpoint_path.replace('\\', '/'))
        if match:
            version_dir = match.group(1)
            # Load run_config.yaml from registration step
            run_cfg_path = os.path.join(version_dir, 'run_config.yaml')
            if os.path.isfile(run_cfg_path):
                try:
                    run_cfg = OmegaConf.load(run_cfg_path)
                    for k, v in run_cfg.items():
                        config[k] = OmegaConf.to_container(v, resolve=True) if isinstance(v, DictConfig) else v
                    print(f"Loaded configuration from {run_cfg_path}")
                except Exception as e:
                    print(f"Warning: failed to load {run_cfg_path}: {e}")
                    
            # Load config_backup.yaml to get model shape
            backup_cfg_path = os.path.join(version_dir, 'config_backup.yaml')
            if os.path.isfile(backup_cfg_path):
                try:
                    backup_cfg = OmegaConf.load(backup_cfg_path)
                    if 'model' in backup_cfg:
                        config['model'] = OmegaConf.to_container(backup_cfg['model'], resolve=True)
                except: pass

    # Override with CLI arguments
    cli_cfg = OmegaConf.to_container(cfg, resolve=True)
    if 'testing' not in config:
        config['testing'] = {}
    for k, v in cli_cfg.get('testing', {}).items():
        config['testing'][k] = v

    os.makedirs(version_dir, exist_ok=True)
    set_seed(seed)

    logger, log_file, timestamp = setup_logger(version_dir, 'attendance_pipeline')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    palm_model = load_models(config, device, logger, checkpoint_path)
    
    data_dir = config.get('dataset', {}).get('data_dir', 'data/PolyU')
    dataset_name = config.get('dataset', {}).get('name', 'PolyU')
    if dataset_name.upper() == 'POLYU':
        dataset_name = 'PalmPrintDataset'
    elif dataset_name.upper() == 'MNIST':
        dataset_name = 'MNISTDataset'
        
    gallery_path = os.path.join(version_dir, f'gallery_{dataset_name.lower()}.pt')
    if not os.path.isfile(gallery_path):
        logger.error(f"Gallery file {gallery_path} not found! Please run register.py first.")
        return
        
    logger.info(f"Loading gallery from {gallery_path}")
    gallery = torch.load(gallery_path, map_location=device)
    
    logger.info(f"Dataset Name: {dataset_name}")
    try:
        test_dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=config.get('dataset', {}), is_train=False)
    except ValueError as e:
        logger.error(str(e))
        return

    process_attendance(palm_model, test_dataset, gallery, device, logger, config)
    
    # Save the updated configuration
    run_cfg_path = os.path.join(version_dir, 'run_config.yaml')
    with open(run_cfg_path, 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(config, f, allow_unicode=True)

if __name__ == '__main__':
    main()
