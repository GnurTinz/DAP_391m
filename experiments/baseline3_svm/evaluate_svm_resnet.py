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

from src.datasets import DatasetFactory
from experiments.baseline2_resnet_arcface.model import ResNetArcFace

def train_svm_for_resnet_person(feat, num_samples=512, noise_std=0.05):
    """
    Giả lập quá trình tạo SVM Verifier cho ResNet.
    Do ResNet không có logvar, ta cộng một lượng noise nhỏ để sinh z_pos.
    z_neg được lấy ngẫu nhiên từ không gian chuẩn.
    """
    latent_dim = feat.size(1)
    
    # 1. Sinh z_pos (stochastic bằng cách cộng noise)
    eps_pos = torch.randn(num_samples, latent_dim, device=feat.device)
    z_pos = feat + noise_std * eps_pos
    y_pos = np.ones(num_samples)
    
    # 2. Sinh z_neg (random giống như baseline3 gốc)
    z_neg = torch.randn(num_samples, latent_dim, device=feat.device)
    y_neg = np.zeros(num_samples)
    
    X = torch.cat([z_pos, z_neg], dim=0).cpu().numpy()
    y = np.concatenate([y_pos, y_neg])
    
    clf = LinearSVC(C=1.0, max_iter=2000, dual=False)
    clf.fit(X, y)
    return clf

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    save_dir = "logs/baseline3_svm"
    os.makedirs(save_dir, exist_ok=True)
    
    b2_save_dir = "logs/baseline2_resnet"
    meta_path = os.path.join(b2_save_dir, "meta.pt")
    weight_path = os.path.join(b2_save_dir, "checkpoint_resnet_arcface.pth")
    
    if not os.path.exists(meta_path) or not os.path.exists(weight_path):
        print("Lỗi: Không tìm thấy model hoặc meta data của baseline 2. Vui lòng chạy baseline 2 trước!")
        sys.exit(1)
        
    meta = torch.load(meta_path)
    num_classes = meta['num_classes']
    
    model = ResNetArcFace(num_classes=num_classes, feature_dim=512, pretrained=False).to(device)
    
    checkpoint = torch.load(weight_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
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
    print(" BASELINE 3: XÂY DỰNG GALLERY (TRAIN SVM CHO MỖI PERSON) TỪ RESNET ")
    print("="*50)
    
    gallery_svms = {}
    pbar = tqdm(train_indices.items(), desc="Train SVM (Gallery)")
    for person_id, indices in pbar:
        idx_to_use = indices[:2] # Giống hệt evaluate_svm.py (chỉ dùng tối đa 2 ảnh để average)
        if dataset is not None and hasattr(dataset, 'samples'):
            images = [dataset[idx][0] for idx in idx_to_use]
            batch_images = torch.stack(images).to(device)
            
            with torch.no_grad():
                features = model(batch_images)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                feat_avg = features.mean(dim=0, keepdim=True)
                feat_avg = torch.nn.functional.normalize(feat_avg, p=2, dim=1)
                
            clf = train_svm_for_resnet_person(feat_avg, num_samples=512)
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
            features = model(batch_images)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            probe_feat = features.mean(dim=0, keepdim=True).cpu().numpy()
            
        best_match = None
        best_score = -float('inf')
        
        # Thử đưa probe_feat qua tất cả các mô hình SVM trong Gallery
        for gal_label, clf in gallery_svms.items():
            score = clf.decision_function(probe_feat)[0]
            if score > best_score:
                best_score = score
                best_match = gal_label
                
        is_correct = best_match == probe_label
        if is_correct:
            correct_matches += 1
        total_probes += 1
        
    accuracy = (correct_matches / total_probes) * 100 if total_probes > 0 else 0
    print(f"\n=> Matching Accuracy (Rank-1) using ResNet+SVM: {accuracy:.2f}% ({correct_matches}/{total_probes})")
    
    # Ghi chú kết quả ra file txt
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    result_txt_path = os.path.join(save_dir, "eval_results.txt")
    with open(result_txt_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{current_time}] === KẾT QUẢ ĐÁNH GIÁ (BASELINE 3 - ResNet+SVM) ===\n")
        f.write(f"Dataset: {dataset_name} ({data_dir})\n")
        f.write(f"Total Persons: {len(person_indices)}\n")
        f.write(f"Total Probes: {total_probes}\n")
        f.write(f"Matching Accuracy (Rank-1): {accuracy:.2f}%\n")
        f.write("="*70 + "\n")
    print(f"Đã ghi kết quả đánh giá tại: {result_txt_path}")

if __name__ == '__main__':
    main()
