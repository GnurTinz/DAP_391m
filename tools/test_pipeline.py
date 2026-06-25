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
from src.engine.represent import optimize_r_from_latent

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_dir, experiment_name):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logger = logging.getLogger('TestPipelineLogger')
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File Handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger, log_file, timestamp

def load_models(config, device, logger, checkpoint_path=None):
    """Initialize and load models."""
    logger.info("Initializing models...")
    
    # 1. Probabilistic Palm Model
    model_config = config.get('model', {})
    if 'decoder' not in model_config:
        model_config['decoder'] = {}
    model_config['decoder']['image_size'] = config.get('dataset', {}).get('image_size', [128, 128])
    palm_model = ProbabilisticPalmModel(model_config).to(device)
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Assuming checkpoint might contain both or just palm_model
        if 'palm_model_state_dict' in checkpoint:
            palm_model.load_state_dict(checkpoint['palm_model_state_dict'])
        else:
            # Fallback to model_state_dict for backward compatibility
            try:
                palm_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            except Exception as e:
                logger.warning(f"Could not load palm_model weights strictly: {e}")
    else:
        logger.warning("No valid checkpoint provided. Models are initialized randomly.")
        
    palm_model.eval()
    
    return palm_model

def build_database(palm_model, dataloader, device, logger):
    """Enrollment phase: extract mu and logvar for all identities to build a gallery."""
    logger.info("Building enrollment database (Gallery)...")
    db_mu = []
    db_logvar = []
    db_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Enrollment"):
            images = images.to(device)
            outputs = palm_model(images, decode=False)
            db_mu.append(outputs['mu'].cpu())
            db_logvar.append(outputs['logvar'].cpu())
            db_labels.append(labels.cpu())
            
    db_mu = torch.cat(db_mu, dim=0)
    db_logvar = torch.cat(db_logvar, dim=0)
    db_labels = torch.cat(db_labels, dim=0)
    
    # Aggregate per identity (Prototype computing)
    unique_labels = torch.unique(db_labels)
    gallery = {}
    
    global_verifier = None
    
    for i, label in enumerate(unique_labels):
        idx = (db_labels == label).nonzero(as_tuple=True)[0]
        # Mean of mus
        mu_c = db_mu[idx].mean(dim=0).unsqueeze(0).to(device)
        logvar_c = db_logvar[idx].mean(dim=0).unsqueeze(0).to(device)
        
        # Tối ưu r cho database. Train verifier ở person đầu tiên, freeze từ person thứ 2.
        r_c, global_verifier, _ = optimize_r_from_latent(
            mu_c, logvar_c, device, 
            verifier=global_verifier, 
            freeze_net=(i > 0),
            num_samples=256, steps=20, verbose=False
        )
        
        gallery[label.item()] = {
            'r': r_c.squeeze(0)
        }
        
    logger.info(f"Database built with {len(gallery)} unique identities.")
    return gallery, global_verifier

def run_pipeline_inference(palm_model, global_verifier, query_loader, gallery, device, logger, config, csv_writer):
    """
    Simulate the end-to-end inference pipeline:
    1. Extract latent distribution (mu_q, logvar_q)
    2. Retrieve top-K candidates
    3. Verify pairs
    4. Decision making
    """
    logger.info("Starting inference pipeline on queries...")
    
    # Verification thresholds from config
    threshold = config.get('testing', {}).get('verify_threshold', 0.5)
    margin = config.get('testing', {}).get('margin', 0.1)
    u_max = config.get('testing', {}).get('max_uncertainty', 2.0)
    top_k = config.get('testing', {}).get('top_k', 5)
    
    gallery_labels = list(gallery.keys())
    gallery_rs = torch.stack([gallery[k]['r'] for k in gallery_labels]) # (Num_IDs, Latent_Dim)
    
    total_queries = 0
    correct_accepts = 0
    false_accepts = 0
    false_rejects = 0
    true_rejects = 0 # If open-set is implemented
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(query_loader, desc="Inference")):
            images = images.to(device)
            labels = labels.tolist()
            
            # 1. Encode query
            outputs = palm_model(images, decode=False)
            mu_q_batch = outputs['mu']
            logvar_q_batch = outputs['logvar']
            
            for i in range(images.size(0)):
                total_queries += 1
                mu_q = mu_q_batch[i]
                logvar_q = logvar_q_batch[i]
                true_label = labels[i]
                
                # Uncertainty (Sigma)
                sigma_q = torch.exp(0.5 * logvar_q)
                uncertainty = sigma_q.mean().item()
                
                if uncertainty > u_max:
                    # Reject due to high uncertainty
                    decision = "REJECT"
                    reason = "High Uncertainty"
                    predicted_label = -1
                    best_score = 0.0
                else:
                    # 2. Tìm r_q cho query, luôn luôn freeze verifier
                    r_q, _, z_pos = optimize_r_from_latent(
                        mu_q.unsqueeze(0), logvar_q.unsqueeze(0), device, 
                        verifier=global_verifier, 
                        freeze_net=True,
                        num_samples=256, steps=20, verbose=False
                    )
                    
                    # 3. Retrieval bằng khoảng cách giữa r_q và các r_c
                    distances = torch.norm(gallery_rs - r_q, dim=1)
                    topk_dist, topk_idx = torch.topk(distances, k=min(top_k, len(gallery_labels)), largest=False)
                    
                    candidate_scores = []
                    candidate_labels = []
                    
                    # 4. Verification bằng TestTimeVerifier đã train
                    for k_idx in topk_idx:
                        c_label = gallery_labels[k_idx.item()]
                        r_c = gallery[c_label]['r'].unsqueeze(0)
                        
                        # Truyền z_pos (X_new) vào z, r_c vào r như sơ đồ Verification
                        with torch.no_grad():
                            final_logits = global_verifier(z_pos, r_c)
                            score_prob = torch.sigmoid(final_logits).mean().item()
                            
                        candidate_scores.append(score_prob)
                        candidate_labels.append(c_label)
                        
                    # 4. Decision Logic
                    # Sort candidates by verification score descending
                    sorted_pairs = sorted(zip(candidate_labels, candidate_scores), key=lambda x: x[1], reverse=True)
                    top1_label, top1_score = sorted_pairs[0]
                    top2_score = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0.0
                    
                    decision = "REJECT"
                    reason = ""
                    predicted_label = -1
                    
                    if top1_score > threshold:
                        if (top1_score - top2_score) > margin:
                            decision = "ACCEPT"
                            predicted_label = top1_label
                            reason = "Passed"
                        else:
                            reason = "Margin too small"
                    else:
                        reason = "Score below threshold"
                        
                    best_score = top1_score
                
                # Evaluation Metrics
                is_correct = (decision == "ACCEPT" and predicted_label == true_label)
                
                if true_label in gallery_labels: # Closed-set / Known ID
                    if decision == "ACCEPT":
                        if predicted_label == true_label:
                            correct_accepts += 1
                        else:
                            false_accepts += 1
                    else:
                        false_rejects += 1
                else: # Open-set / Unknown ID
                    if decision == "ACCEPT":
                        false_accepts += 1
                    else:
                        true_rejects += 1
                
                # Write to CSV
                csv_writer.writerow([
                    total_queries,
                    true_label,
                    predicted_label,
                    f"{best_score:.4f}",
                    f"{uncertainty:.4f}",
                    decision,
                    reason,
                    is_correct
                ])
                
    logger.info("=== Inference Summary ===")
    logger.info(f"Total Queries: {total_queries}")
    logger.info(f"Correct Accepts (TAR): {correct_accepts}")
    logger.info(f"False Accepts (FAR): {false_accepts}")
    logger.info(f"False Rejects (FRR): {false_rejects}")
    logger.info(f"True Rejects (Unknowns): {true_rejects}")
    
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
            
    output_dir = os.path.join(version_dir, 'test_results')
    os.makedirs(output_dir, exist_ok=True)

    # Set seed
    set_seed(seed)
    
    # Set default test configs if not present
    if 'testing' not in config:
        config['testing'] = {
            'verify_threshold': 0.5,
            'margin': 0.1,
            'max_uncertainty': 2.0,
            'top_k': 5
        }

    # Setup Logging
    logger, log_file, timestamp = setup_logger(output_dir, 'test_pipeline')
    logger.info(f"Random seed set to: {seed}")
    logger.info(f"Logs will be saved to: {log_file}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load Models
    palm_model = load_models(config, device, logger, checkpoint_path)
    
    # Datasets setup
    # In a real scenario, we should have a gallery dataloader and a query dataloader
    # For simulation, we use the test set and split it into gallery and query
    data_dir = config.get('dataset', {}).get('data_dir', 'data/PolyU')
    dataset_name = config.get('dataset', {}).get('name', 'PolyU')
    batch_size = config.get('testing', {}).get('batch_size', 32)
    num_workers = config.get('testing', {}).get('num_workers', 2)
    
    logger.info(f"Loading {dataset_name} dataset from {data_dir}...")
    # For backward compatibility with old configs
    if dataset_name.upper() == 'POLYU':
        dataset_name = 'PalmPrintDataset'
    elif dataset_name.upper() == 'MNIST':
        dataset_name = 'MNISTDataset'
        
    try:
        test_dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=config.get('dataset', {}), is_train=False)
    except ValueError as e:
        logger.error(str(e))
        return

    # Simulate Gallery and Query splits
    # E.g. First sample of each class goes to Gallery, rest to Query
    gallery_indices = []
    query_indices = []
    seen_labels = set()
    
    labels = test_dataset.get_labels() if hasattr(test_dataset, 'get_labels') else [test_dataset[i][1] for i in range(len(test_dataset))]
    
    for i, label in enumerate(labels):
        if label not in seen_labels:
            gallery_indices.append(i)
            seen_labels.add(label)
        else:
            query_indices.append(i)
            
    gallery_subset = torch.utils.data.Subset(test_dataset, gallery_indices)
    query_subset = torch.utils.data.Subset(test_dataset, query_indices)
    
    gallery_loader = DataLoader(gallery_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    query_loader = DataLoader(query_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Setup CSV Writer
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"test_results_{timestamp}.csv")
    csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Query_ID', 'True_Label', 'Predicted_Label', 'Best_Score', 'Uncertainty', 'Decision', 'Reason', 'Is_Correct'])
    logger.info(f"Results will be saved to CSV: {csv_path}")
    
    # Run Pipeline
    gallery, global_verifier = build_database(palm_model, gallery_loader, device, logger)
    run_pipeline_inference(palm_model, global_verifier, query_loader, gallery, device, logger, config, csv_writer)
    
    csv_file.close()
    logger.info("Test pipeline finished successfully.")

if __name__ == '__main__':
    main()
