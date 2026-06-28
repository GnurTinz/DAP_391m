import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import numpy as np
import random
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import DatasetFactory
from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from tools.test_pipeline import set_seed, setup_logger, load_models

def evaluate_probe(palm_model, gallery, query_loader, device, logger, config, csv_writer):
    """
    Chấm công 2 bước: Tính Cosine Similarity, check Max Uncertainty.
    """
    logger.info("Starting Probe inference...")
    
    threshold = config.get('testing', {}).get('verify_threshold', 0.5)
    margin = config.get('testing', {}).get('margin', 0.1)
    u_max = config.get('testing', {}).get('max_uncertainty', 2.0)
    use_uncertainty = config.get('testing', {}).get('use_uncertainty', True)
    
    # Đã hỗ trợ đa mẫu (multiple templates per person)
    gallery_labels_flat = []
    gallery_rs_flat = []
    
    for k in gallery.keys():
        r_c = gallery[k]
        if r_c.dim() == 1:
            r_c = r_c.unsqueeze(0)
        for i in range(r_c.size(0)):
            gallery_rs_flat.append(r_c[i])
            gallery_labels_flat.append(k)
            
    gallery_rs = torch.stack(gallery_rs_flat).to(device) # (Total_Templates, Proj_Dim)
    gallery_labels = list(gallery.keys())
    
    total_queries = 0
    total_known_queries = 0
    correct_accepts = 0
    false_accepts = 0
    false_rejects = 0
    true_rejects = 0
    
    y_true = []
    y_scores = []
    
    optimize_probe = config.get('testing', {}).get('optimize_probe', True)
    loss_type = config.get('testing', {}).get('gallery_loss', 'bce')
    num_samples = config.get('testing', {}).get('gallery_samples', 256)
    max_steps = config.get('testing', {}).get('gallery_max_steps', 100)
    lr = config.get('testing', {}).get('gallery_lr', 0.01)
    
    palm_model.eval()
    from src.engine.represent import optimize_r_in_projected_space
    
    for images, labels in tqdm(query_loader, desc="Inference"):
        images = images.to(device)
        labels = labels.tolist()
        
        with torch.no_grad():
            outputs = palm_model(images, decode=False)
            mu_q_batch = outputs['mu']
            logvar_q_batch = outputs['logvar']
            
            if not optimize_probe:
                p_probe_batch = palm_model.projector(mu_q_batch)
                p_probe_batch = torch.nn.functional.normalize(p_probe_batch, p=2, dim=1)
                
        for i in range(images.size(0)):
            total_queries += 1
            true_label = labels[i]
            
            mu_q = mu_q_batch[i:i+1]
            logvar_q = logvar_q_batch[i:i+1]
            uncertainty = torch.exp(0.5 * logvar_q).mean().item()
            
            if optimize_probe:
                idx_others = [j for j in range(images.size(0)) if j != i]
                if len(idx_others) > 0:
                    mu_others = mu_q_batch[idx_others]
                    logvar_others = logvar_q_batch[idx_others]
                else:
                    mu_others = None
                    logvar_others = None
                    
                p_probe = optimize_r_in_projected_space(
                    mu_q, logvar_q, mu_others, logvar_others,
                    model=palm_model, device=device, config=config,
                    num_samples=num_samples, loss_type=loss_type, max_steps=max_steps, lr=lr,
                    verbose=False
                ).unsqueeze(0)
            else:
                p_probe = p_probe_batch[i:i+1]
                
            decision = "REJECT"
            reason = ""
            predicted_label = -1
            best_score = 0.0
            
            # 1. Đo độ tương đồng Cosine
            sim_scores = torch.mm(p_probe, gallery_rs.t()).squeeze(0) # (Total_Templates,)
            topk_scores, topk_idx = torch.topk(sim_scores, k=min(2, len(gallery_labels_flat)))
            
            best_score = topk_scores[0].item()
            predicted_label = gallery_labels_flat[topk_idx[0].item()]
            top2_score = topk_scores[1].item() if len(topk_scores) > 1 else -1.0
            
            # Tính Metrics EER (Lấy điểm số cao nhất thuộc class tương ứng nếu có)
            if true_label in gallery_labels:
                # Tìm điểm cao nhất của đúng true_label
                idx_true_all = [idx for idx, val in enumerate(gallery_labels_flat) if val == true_label]
                score_true = sim_scores[idx_true_all].max().item()
                y_true.append(1)
                y_scores.append(score_true)
                
                # Thêm 1 mẫu negative (người khác có điểm cao nhất)
                # Tìm index đầu tiên thuộc người khác
                idx_false = -1
                for i_idx in topk_idx:
                    if gallery_labels_flat[i_idx.item()] != true_label:
                        idx_false = i_idx.item()
                        break
                if idx_false != -1:
                    y_true.append(0)
                    y_scores.append(sim_scores[idx_false].item())
            else:
                y_true.append(0)
                y_scores.append(best_score)
            
            # 2. Đưa ra Quyết định (Bỏ qua Verifier, chỉ lấy Predicted Label)
            decision = "ACCEPT"
            reason = "Rank-1 Search"
            # predicted_label đã được gán sẵn là top 1 similarity
            
            is_correct = (predicted_label == true_label)
            
            if true_label in gallery_labels:
                total_known_queries += 1
                if is_correct:
                    correct_accepts += 1
                else:
                    false_accepts += 1
            else:
                # Mẫu ngoài (Unknown) mà bị gán bừa vào Gallery -> False Accept
                false_accepts += 1
            
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
    logger.info(f"Overall Accuracy: {accuracy:.2f}%")
    
    classification_acc = (correct_accepts / total_known_queries) * 100 if total_known_queries > 0 else 0
    logger.info(f"Classification Accuracy (Closed-Set): {classification_acc:.2f}%")
    
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
            'verify_threshold': 0.5,
            'margin': 0.05,
            'max_uncertainty': 2.0,
            'use_uncertainty': True
        }

    logger, log_file, timestamp = setup_logger(output_dir, 'evaluate_probe')
    logger.info(f"Logs will be saved to: {log_file}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    palm_model = load_models(config, device, logger, checkpoint_path)
    
    gallery_save_path = os.path.join(output_dir, 'gallery.pt')
    if not os.path.isfile(gallery_save_path):
        logger.error(f"Gallery file not found at {gallery_save_path}! Please run build_gallery.py first.")
        return
        
    logger.info(f"Loading gallery from {gallery_save_path}")
    gallery = torch.load(gallery_save_path, map_location=device)
    
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
        # Probe từ tập Test/Val
        test_dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=config.get('dataset', {}), is_train=False)
    except ValueError as e:
        logger.error(str(e))
        return

    query_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    csv_path = os.path.join(output_dir, f"test_results_{timestamp}.csv")
    csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Query_ID', 'True_Label', 'Predicted_Label', 'Best_Score', 'Uncertainty', 'Decision', 'Reason', 'Is_Correct'])
    
    evaluate_probe(palm_model, gallery, query_loader, device, logger, config, csv_writer)
    
    csv_file.close()
    logger.info("Evaluation finished successfully.")

if __name__ == '__main__':
    main()
