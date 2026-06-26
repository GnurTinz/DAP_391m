import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.datasets import DatasetFactory

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    checkpoint_path = config.get('checkpoint', '')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    # Khởi tạo mô hình
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
    
    # Load checkpoint
    if checkpoint_data:
        state_dict = checkpoint_data.get('model_state_dict', checkpoint_data.get('state_dict', checkpoint_data))
        clean_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
    else:
        print("Cảnh báo: Không có checkpoint hợp lệ.")
    model.eval()

    # Load file Gallery mu
    if checkpoint_path:
        version_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    else:
        version_dir = "logs/unversioned_results"
        
    gallery_path = os.path.join(version_dir, 'gallery_mu.pt')
    if not os.path.exists(gallery_path):
        print(f"Lỗi: Không tìm thấy file {gallery_path}! Hãy chạy build_gallery_mu.py trước.")
        sys.exit(1)
        
    print(f"Đang tải Gallery mu từ {gallery_path}...")
    gallery_data = torch.load(gallery_path, map_location=device)
    gallery_mu = gallery_data['gallery_mu']

    # Khởi tạo Dataset
    dataset_config = config.get('dataset', {})
    data_dir = dataset_config.get('data_dir', 'data/collect')
    dataset_name = dataset_config.get('name', 'OwnOriginalDataset')
    
    print(f"\nĐang tải dữ liệu từ {dataset_name} ({data_dir})...")
    dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=dataset_config, is_train=False)
    
    val_person_indices = {}
    if hasattr(dataset, 'samples') and len(dataset.samples) > 0:
        person_indices = {}
        for i, item in enumerate(dataset.samples):
            label = item[1]
            if label not in person_indices:
                person_indices[label] = []
            person_indices[label].append(i)
            
        for label, indices in person_indices.items():
            split_idx = max(1, len(indices) // 2)
            val_person_indices[label] = indices[split_idx:]
    else:
        val_person_indices = {0: [1]}
        
    print(f"Tổng số Person trên tập Val: {len([v for v in val_person_indices.values() if len(v)>0])}")

    # ==========================================
    # Trích xuất mu trên tập Val (Probe)
    # ==========================================
    print("\n" + "="*50)
    print(" BASELINE 1: TRÍCH XUẤT PROBE MU TỪ TẬP VAL ")
    print("="*50)
    
    probe_mu = {}
    
    pbar = tqdm(val_person_indices.items(), desc="Xử lý Person")
    for person_id, indices in pbar:
        if len(indices) == 0:
            continue
            
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
            mu_avg = mu_val.mean(dim=0, keepdim=True)
            
        probe_mu[person_id] = mu_avg.cpu().detach()
        
    # ==========================================
    # Đánh giá Khớp mu (Matching)
    # ==========================================
    print("\n" + "="*50)
    print(" KẾT QUẢ SO SÁNH VAL VÀ TRAIN (BASELINE MATCHING) ")
    print("="*50)
    
    correct_matches = 0
    total_probes = len(probe_mu)
    
    for probe_label, p_mu in probe_mu.items():
        best_match = None
        best_dist = float('inf')
        
        for gal_label, g_mu in gallery_mu.items():
            dist = torch.norm(p_mu - g_mu, p=2).item()
            if dist < best_dist:
                best_dist = dist
                best_match = gal_label
                
        is_correct = best_match == probe_label
        if is_correct:
            correct_matches += 1
        
    accuracy = (correct_matches / total_probes) * 100 if total_probes > 0 else 0
    print(f"=> Matching Accuracy (Rank-1) using raw mu: {accuracy:.2f}% ({correct_matches}/{total_probes})")
    
    print("\nChú ý: Phương pháp Baseline này dùng trực tiếp mu, không có Test-time Verifier.")
    print("Vì không sinh ra các phân bố Positive/Negative, nên chúng ta không tính EER được ở đây.")
    print("\nHoàn tất toàn bộ chu trình đánh giá Baseline!")

if __name__ == '__main__':
    main()
