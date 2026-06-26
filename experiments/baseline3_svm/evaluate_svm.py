import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
import numpy as np
from sklearn.svm import LinearSVC
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.datasets import DatasetFactory

def train_svm_for_person(mu, logvar, num_samples=512):
    """
    Thay vì dùng Mạng Neural MLP Verifier để tìm điểm r,
    ta dùng SVM tuyến tính để tìm siêu mặt phẳng (hyperplane) phân tách z_pos và z_neg.
    Trả về mô hình SVM đã huấn luyện.
    """
    latent_dim = mu.size(1)
    sigma = torch.exp(0.5 * logvar)
    
    # 1. Sinh z_pos (stochastic)
    eps_pos = torch.randn(num_samples, latent_dim, device=mu.device)
    z_pos = mu + sigma * eps_pos
    y_pos = np.ones(num_samples)
    
    # 2. Sinh z_neg (random)
    z_neg = torch.randn(num_samples, latent_dim, device=mu.device)
    y_neg = np.zeros(num_samples)
    
    X = torch.cat([z_pos, z_neg], dim=0).cpu().numpy()
    y = np.concatenate([y_pos, y_neg])
    
    clf = LinearSVC(C=1.0, max_iter=2000, dual=False)
    clf.fit(X, y)
    return clf

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    checkpoint_path = config.get('checkpoint', '')
    rep_cfg = config.get('represent', {})
    samples = rep_cfg.get('num_samples', 512)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    # Khởi tạo mô hình Feature Extractor
    checkpoint_data = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location=device)

    model_config = config.get('model', {})
    if checkpoint_data and 'hyper_parameters' in checkpoint_data and 'model' in checkpoint_data['hyper_parameters']:
        model_config = checkpoint_data['hyper_parameters']['model']
        
    model_type = model_config.get('type', 'default')
    
    if 'decoder' not in model_config:
        model_config['decoder'] = {}
    if 'image_size' not in model_config['decoder']:
        model_config['decoder']['image_size'] = config.get('dataset', {}).get('image_size', [128, 128])
    
    if model_type == 'unet':
        model = UNetPalmModel(model_config).to(device)
    else:
        model = ProbabilisticPalmModel(model_config).to(device)
    
    if checkpoint_data:
        state_dict = checkpoint_data.get('model_state_dict', checkpoint_data.get('state_dict', checkpoint_data))
        clean_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    # Load dataset
    dataset_config = config.get('dataset', {})
    data_dir = dataset_config.get('data_dir', 'data/collect')
    dataset_name = dataset_config.get('name', 'OwnOriginalDataset')
    
    print(f"\nĐang tải dữ liệu từ {dataset_name} ({data_dir})...")
    dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=dataset_config, is_train=False)
    
    person_indices = {}
    if hasattr(dataset, 'samples') and len(dataset.samples) > 0:
        for i, item in enumerate(dataset.samples):
            label = item[1]
            if label not in person_indices:
                person_indices[label] = []
            person_indices[label].append(i)
            
    train_indices = {}
    val_indices = {}
    for label, indices in person_indices.items():
        split_idx = max(1, len(indices) // 2)
        train_indices[label] = indices[:split_idx]
        val_indices[label] = indices[split_idx:]
        
    print(f"Tổng số Person: {len(person_indices)}")

    # ==========================================
    # 1. XÂY DỰNG GALLERY (HUẤN LUYỆN SVM CHO TỪNG PERSON)
    # ==========================================
    print("\n" + "="*50)
    print(" BASELINE 3: XÂY DỰNG GALLERY (TRAIN SVM CHO MỖI PERSON) ")
    print("="*50)
    
    gallery_svms = {}
    pbar = tqdm(train_indices.items(), desc="Train SVM (Gallery)")
    for person_id, indices in pbar:
        idx_to_use = indices[:2]
        if dataset is not None and hasattr(dataset, 'samples'):
            images = [dataset[idx][0] for idx in idx_to_use]
            batch_images = torch.stack(images).to(device)
        else:
            img_size = dataset_config.get('image_size', [128, 128])
            batch_images = torch.randn(len(idx_to_use), 3, img_size[0], img_size[1]).to(device)
            
        with torch.no_grad():
            out = model(batch_images, decode=False)
            mu_val = out['mu']
            logvar_val = out['logvar']
            mu_avg = mu_val.mean(dim=0, keepdim=True)
            logvar_avg = logvar_val.mean(dim=0, keepdim=True)
            
        # Huấn luyện SVM
        clf = train_svm_for_person(mu_avg, logvar_avg, num_samples=samples)
        gallery_svms[person_id] = clf

    # ==========================================
    # 2. ĐÁNH GIÁ TRÊN TẬP VAL (MATCHING VỚI SVM)
    # ==========================================
    print("\n" + "="*50)
    print(" KẾT QUẢ SO SÁNH VAL VÀ TRAIN (SVM MATCHING) ")
    print("="*50)
    
    correct_matches = 0
    total_probes = 0
    
    pbar_val = tqdm(val_indices.items(), desc="Đánh giá Probe")
    for probe_label, indices in pbar_val:
        if len(indices) == 0:
            continue
            
        idx_to_use = indices[:2]
        images = [dataset[idx][0] for idx in idx_to_use]
        batch_images = torch.stack(images).to(device)
            
        with torch.no_grad():
            out = model(batch_images, decode=False)
            mu_probe = out['mu'].mean(dim=0, keepdim=True).cpu().numpy() # (1, latent_dim)
            
        best_match = None
        best_score = -float('inf')
        
        # Thử đưa mu_probe qua tất cả các mô hình SVM trong Gallery
        for gal_label, clf in gallery_svms.items():
            # decision_function trả về khoảng cách đại số từ mẫu tới siêu mặt phẳng.
            # Điểm càng cao (> 0) thì càng chắc chắn thuộc về class 1 (Positive)
            score = clf.decision_function(mu_probe)[0]
            if score > best_score:
                best_score = score
                best_match = gal_label
                
        is_correct = best_match == probe_label
        if is_correct:
            correct_matches += 1
        total_probes += 1
        
    accuracy = (correct_matches / total_probes) * 100 if total_probes > 0 else 0
    print(f"\n=> Matching Accuracy (Rank-1) using Linear SVM: {accuracy:.2f}% ({correct_matches}/{total_probes})")
    print("Hoàn tất đánh giá Baseline 3!")

if __name__ == '__main__':
    main()
