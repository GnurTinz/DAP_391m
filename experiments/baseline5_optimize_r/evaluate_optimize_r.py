import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.datasets import DatasetFactory
from experiments.baseline2_resnet_arcface.model import ResNetArcFace

class SimpleProjector(nn.Module):
    def __init__(self, in_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def optimize_r_input(features, num_epochs=100, lr=0.01):
    """
    Hướng 1: Tối ưu r cộng trực tiếp vào đầu vào (Z + r).
    features: [N, 512] (dữ liệu gallery của 1 người).
    Mục tiêu: Tìm r sao cho các vector (Z + r) chụm lại gần nhau nhất (giảm variance).
    (Đây là một dummy task để chứng minh việc r hội tụ).
    """
    latent_dim = features.size(1)
    r = nn.Parameter(torch.zeros(1, latent_dim, device=features.device))
    optimizer = optim.Adam([r], lr=lr)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        modified_Z = features + r
        mean_Z = modified_Z.mean(dim=0, keepdim=True)
        loss = torch.mean((modified_Z - mean_Z)**2)
        loss += 0.01 * torch.mean(r**2)
        loss.backward()
        optimizer.step()
        
    return r.detach()

def optimize_r_projected(features, projector, num_epochs=100, lr=0.01):
    """
    Hướng 2: Tối ưu r trong không gian Projected.
    features: [N, 512]
    projector: Module phi tuyến
    Mục tiêu: Cố định Projector, tìm r trong không gian P (out_dim) sao cho r gần với P nhất.
    """
    out_dim = 256
    r = nn.Parameter(torch.zeros(1, out_dim, device=features.device))
    optimizer = optim.Adam([r], lr=lr)
    
    with torch.no_grad():
        projected_Z = projector(features)
        
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = torch.mean((projected_Z - r)**2)
        loss.backward()
        optimizer.step()
        
    return r.detach()

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    save_dir = "logs/baseline5_optimize_r"
    os.makedirs(save_dir, exist_ok=True)
    
    b2_save_dir = "logs/baseline2_resnet"
    meta_path = os.path.join(b2_save_dir, "meta.pt")
    weight_path = os.path.join(b2_save_dir, "checkpoint_resnet_arcface.pth")
    
    if not os.path.exists(meta_path) or not os.path.exists(weight_path):
        print("Vui lòng chạy baseline 2 trước!")
        sys.exit(1)
        
    meta = torch.load(meta_path)
    model = ResNetArcFace(num_classes=meta['num_classes'], feature_dim=512, pretrained=False).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device).get('model_state_dict', torch.load(weight_path, map_location=device)))
    model.eval()

    projector = SimpleProjector(512, 256).to(device)
    projector.eval() 

    dataset_config = config.get('dataset', {})
    dataset = DatasetFactory.create(dataset_config.get('name', 'OwnOriginalDataset'), data_dir=dataset_config.get('data_dir', 'data/collect'), config=dataset_config, is_train=False)
    
    person_indices = {}
    for i, item in enumerate(dataset.samples):
        person_indices.setdefault(item[1], []).append(i)
            
    train_indices = {k: v[:max(1, len(v)//2)] for k, v in person_indices.items()}
    val_indices = {k: v[max(1, len(v)//2):] for k, v in person_indices.items()}

    print("\n[GIAI ĐOẠN 1] Tìm kiếm r bằng Gradient Descent (Tối ưu hóa)...")
    gallery_r_input = {}
    gallery_r_proj = {}
    
    for person_id, indices in tqdm(train_indices.items()):
        images = torch.stack([dataset[idx][0] for idx in indices]).to(device)
        with torch.no_grad():
            features = model(images)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
        r_in = optimize_r_input(features)
        gallery_r_input[person_id] = (features.mean(dim=0, keepdim=True) + r_in)
        
        r_proj = optimize_r_projected(features, projector)
        gallery_r_proj[person_id] = r_proj
        
    print("\n[GIAI ĐOẠN 2] Đánh giá Probe trên tập Val...")
    correct_input = 0
    correct_proj = 0
    total = 0
    
    y_true_in = []
    y_scores_in = []
    y_true_proj = []
    y_scores_proj = []
    
    for probe_label, indices in tqdm(val_indices.items()):
        if not indices: continue
        images = torch.stack([dataset[idx][0] for idx in indices]).to(device)
        with torch.no_grad():
            probe_feat = model(images)
            probe_feat = torch.nn.functional.normalize(probe_feat, p=2, dim=1)
            probe_feat_mean = probe_feat.mean(dim=0, keepdim=True)
            probe_proj_mean = projector(probe_feat_mean)
            
        best_sim_in = -float('inf')
        best_label_in = None
        best_sim_proj = -float('inf')
        best_label_proj = None
        
        for gal_label in gallery_r_input.keys():
            sim_in = torch.nn.functional.cosine_similarity(probe_feat_mean, gallery_r_input[gal_label]).item()
            sim_proj = torch.nn.functional.cosine_similarity(probe_proj_mean, gallery_r_proj[gal_label]).item()
            
            y_true_in.append(1 if gal_label == probe_label else 0)
            y_scores_in.append(sim_in)
            
            y_true_proj.append(1 if gal_label == probe_label else 0)
            y_scores_proj.append(sim_proj)
            
            if sim_in > best_sim_in:
                best_sim_in = sim_in
                best_label_in = gal_label
                
            if sim_proj > best_sim_proj:
                best_sim_proj = sim_proj
                best_label_proj = gal_label
        
        if best_label_in == probe_label: correct_input += 1
        if best_label_proj == probe_label: correct_proj += 1
        total += 1
        
    acc_in = (correct_input / total) * 100 if total > 0 else 0
    acc_proj = (correct_proj / total) * 100 if total > 0 else 0
    
    from sklearn.metrics import roc_curve
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    
    def calculate_eer(y_true, y_scores):
        if len(y_true) > 0 and len(set(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            return eer * 100
        return 0.0
        
    eer_in = calculate_eer(y_true_in, y_scores_in)
    eer_proj = calculate_eer(y_true_proj, y_scores_proj)

    # Ghi chú kết quả ra file txt đầy đủ
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    result_txt_path = os.path.join(save_dir, "eval_results.txt")
    with open(result_txt_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{current_time}] === KẾT QUẢ ĐÁNH GIÁ (BASELINE 5 - Optimize r) ===\n")
        f.write(f"Dataset: {config.get('dataset', {}).get('name', 'OwnOriginalDataset')} ({config.get('dataset', {}).get('data_dir', 'data/collect')})\n")
        f.write(f"Total Persons: {len(person_indices)}\n")
        f.write(f"Total Probes: {total}\n")
        f.write(f"Matching Accuracy (Tối ưu r ở Input: Z + r): {acc_in:.2f}%\n")
        f.write(f"Equal Error Rate  (Tối ưu r ở Input: Z + r): {eer_in:.2f}%\n")
        f.write(f"Matching Accuracy (Tối ưu r ở Projected Space): {acc_proj:.2f}%\n")
        f.write(f"Equal Error Rate  (Tối ưu r ở Projected Space): {eer_proj:.2f}%\n")
        f.write("="*70 + "\n")
    print(f"Đã ghi kết quả đánh giá chi tiết tại: {result_txt_path}")

if __name__ == '__main__':
    main()
