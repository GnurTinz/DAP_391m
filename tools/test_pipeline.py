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

def build_gallery(palm_model, dataloader, device, logger, config):
    """
    Trích xuất mu, logvar từ tập Train. 
    Tối ưu r trong không gian Projected Vector.
    """
    logger.info("Building enrollment database (Gallery) from Train set...")
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
    
    unique_labels = torch.unique(db_labels)
    gallery = {}
    
    loss_type = config.get('testing', {}).get('gallery_loss', 'bce')
    num_samples = config.get('testing', {}).get('gallery_samples', 256)
    
    logger.info(f"Optimizing R using {loss_type.upper()} loss...")
    
    for label in tqdm(unique_labels, desc="Optimizing Gallery R"):
        idx = (db_labels == label).nonzero(as_tuple=True)[0]
        idx_others = (db_labels != label).nonzero(as_tuple=True)[0]
        
        # Lấy trung bình mu_c, logvar_c của person
        mu_c = db_mu[idx].mean(dim=0).unsqueeze(0)
        logvar_c = db_logvar[idx].mean(dim=0).unsqueeze(0)
        
        mu_others = db_mu[idx_others]
        logvar_others = db_logvar[idx_others]
        
        r_c = optimize_r_in_projected_space(
            mu_c, logvar_c, mu_others, logvar_others, 
            model=palm_model, device=device, config=config, 
            num_samples=num_samples, loss_type=loss_type
        )
        
        gallery[label.item()] = r_c
        
    logger.info(f"Database built with {len(gallery)} unique identities.")
    return gallery

def evaluate_probe(palm_model, gallery, query_loader, device, logger, config, csv_writer):
    """
    Chấm công 2 bước: Tính Cosine Similarity, check Max Uncertainty.
    """
    logger.info("Starting Probe inference...")
    
    threshold = config.get('testing', {}).get('verify_threshold', 0.5)
    margin = config.get('testing', {}).get('margin', 0.1)
    u_max = config.get('testing', {}).get('max_uncertainty', 2.0)
    use_uncertainty = config.get('testing', {}).get('use_uncertainty', True)
    
    gallery_labels = list(gallery.keys())
    gallery_rs = torch.stack([gallery[k] for k in gallery_labels]).to(device) # (Num_IDs, Proj_Dim)
    
    total_queries = 0
    correct_accepts = 0
    false_accepts = 0
    false_rejects = 0
    true_rejects = 0
    
    y_true = []
    y_scores = []
    
    palm_model.eval()
    with torch.no_grad():
        for images, labels in tqdm(query_loader, desc="Inference"):
            images = images.to(device)
            labels = labels.tolist()
            
            outputs = palm_model(images, decode=False)
            mu_q_batch = outputs['mu']
            logvar_q_batch = outputs['logvar']
            
            # P_probe (Deterministic Projection)
            p_probe_batch = palm_model.projector(mu_q_batch)
            p_probe_batch = torch.nn.functional.normalize(p_probe_batch, p=2, dim=1)
            
            for i in range(images.size(0)):
                total_queries += 1
                logvar_q = logvar_q_batch[i]
                p_probe = p_probe_batch[i:i+1] # (1, Proj_Dim)
                true_label = labels[i]
                
                # Tính Uncertainty
                sigma_q = torch.exp(0.5 * logvar_q)
                uncertainty = sigma_q.mean().item()
                
                decision = "REJECT"
                reason = ""
                predicted_label = -1
                best_score = 0.0
                
                # 1. Đo độ tương đồng Cosine
                sim_scores = torch.mm(p_probe, gallery_rs.t()).squeeze(0) # (Num_IDs,)
                topk_scores, topk_idx = torch.topk(sim_scores, k=min(2, len(gallery_labels)))
                
                best_score = topk_scores[0].item()
                predicted_label = gallery_labels[topk_idx[0].item()]
                top2_score = topk_scores[1].item() if len(topk_scores) > 1 else -1.0
                
                # Tính Metrics EER (Chỉ lấy điểm số cao nhất thuộc class tương ứng nếu có)
                if true_label in gallery_labels:
                    idx_true = gallery_labels.index(true_label)
                    score_true = sim_scores[idx_true].item()
                    y_true.append(1)
                    y_scores.append(score_true)
                    
                    # Thêm 1 mẫu negative (người khác có điểm cao nhất)
                    idx_false = topk_idx[0].item() if topk_idx[0].item() != idx_true else (topk_idx[1].item() if len(topk_idx)>1 else idx_true)
                    if idx_false != idx_true:
                        y_true.append(0)
                        y_scores.append(sim_scores[idx_false].item())
                else:
                    y_true.append(0)
                    y_scores.append(best_score)
                
                # 2. Đưa ra Quyết định (Decision Logic)
                if best_score > threshold:
                    if (best_score - top2_score) > margin:
                        if use_uncertainty and uncertainty > u_max:
                            reason = "High Uncertainty"
                        else:
                            decision = "ACCEPT"
                            reason = "Passed"
                    else:
                        reason = "Margin too small"
                else:
                    reason = "Score below threshold"
                    predicted_label = -1
                        
                is_correct = (decision == "ACCEPT" and predicted_label == true_label)
                
                if true_label in gallery_labels:
                    if decision == "ACCEPT":
                        if predicted_label == true_label:
                            correct_accepts += 1
                        else:
                            false_accepts += 1
                    else:
                        false_rejects += 1
                else:
                    if decision == "ACCEPT":
                        false_accepts += 1
                    else:
                        true_rejects += 1
                
                if csv_writer:
                    csv_writer.writerow([
                        total_queries, true_label, predicted_label,
                        f"{best_score:.4f}", f"{uncertainty:.4f}",
                        decision, reason, is_correct
                    ])
                
    logger.info("=== Inference Summary ===")
    logger.info(f"Total Queries: {total_queries}")
    logger.info(f"Correct Accepts (TAR): {correct_accepts}")
    logger.info(f"False Accepts (FAR): {false_accepts}")
    logger.info(f"False Rejects (FRR): {false_rejects}")
    logger.info(f"True Rejects (Unknowns): {true_rejects}")
    
    accuracy = (correct_accepts / total_queries) * 100 if total_queries > 0 else 0
    logger.info(f"Rank-1 Accuracy: {accuracy:.2f}%")
    
    if len(y_true) > 0 and len(set(y_true)) > 1:
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        logger.info(f"Equal Error Rate (EER): {eer*100:.2f}%")
    else:
        logger.info("Equal Error Rate (EER): N/A (Not enough classes)")

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
    
    gallery = build_gallery(palm_model, gallery_loader, device, logger, config)
    evaluate_probe(palm_model, gallery, query_loader, device, logger, config, csv_writer)
    
    csv_file.close()
    logger.info("Test pipeline finished successfully.")

if __name__ == '__main__':
    main()
