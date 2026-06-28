import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.datasets import DatasetFactory
from experiments.baseline2_resnet_arcface.model import ResNetArcFace

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    save_dir = "logs/baseline2_resnet"
    meta_path = os.path.join(save_dir, "meta.pt")
    weight_path = os.path.join(save_dir, "checkpoint_resnet_arcface.pth")
    
    if not os.path.exists(meta_path) or not os.path.exists(weight_path):
        print("Lỗi: Không tìm thấy model hoặc meta data. Vui lòng chạy train.py trước!")
        sys.exit(1)
        
    meta = torch.load(meta_path)
    num_classes = meta['num_classes']
    
    # Init model
    model = ResNetArcFace(num_classes=num_classes, feature_dim=512, pretrained=False).to(device)
    
    # Xử lý load dict an toàn (hỗ trợ cả checkpoint chứa meta data)
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

    # Trích xuất Gallery (từ tập Train)
    print("\n" + "="*50)
    print(" BASELINE 2: TRÍCH XUẤT GALLERY (RESNET FEATURES) ")
    print("="*50)
    gallery_features = {}
    for person_id, indices in tqdm(train_indices.items(), desc="Gallery"):
        if dataset is not None and hasattr(dataset, 'samples'):
            images = [dataset[idx][0] for idx in indices]
            batch_images = torch.stack(images).to(device)
            with torch.no_grad():
                features = model(batch_images) # (N, 512)
                # Normalize before average
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                avg_feat = features.mean(dim=0, keepdim=True)
                avg_feat = torch.nn.functional.normalize(avg_feat, p=2, dim=1)
            gallery_features[person_id] = avg_feat.cpu()

    # Trích xuất Probe (từ tập Val)
    print("\n" + "="*50)
    print(" KẾT QUẢ SO SÁNH VAL VÀ TRAIN (RESNET MATCHING) ")
    print("="*50)
    correct_matches = 0
    total_probes = 0
    
    y_true = []
    y_scores = []
    
    for probe_label, indices in tqdm(val_indices.items(), desc="Probe Matching"):
        if len(indices) == 0:
            continue
            
        # Mỗi ảnh trong tập Val là một lần chấm công độc lập (Independent Probe)
        for idx in indices:
            img = dataset[idx][0].unsqueeze(0).to(device)
            
            with torch.no_grad():
                probe_feat = model(img)
                probe_feat = torch.nn.functional.normalize(probe_feat, p=2, dim=1).cpu()
                
            best_match = None
            best_sim = -float('inf')
            
            for gal_label, gal_feat in gallery_features.items():
                # Cosine similarity
                sim = torch.mm(probe_feat, gal_feat.t()).item()
                
                y_true.append(1 if gal_label == probe_label else 0)
                y_scores.append(sim)
                
                if sim > best_sim:
                    best_sim = sim
                    best_match = gal_label
                    
            is_correct = best_match == probe_label
            if is_correct:
                correct_matches += 1
            total_probes += 1
        
    accuracy = (correct_matches / total_probes) * 100 if total_probes > 0 else 0
    print(f"\n=> Matching Accuracy (Rank-1) using ResNet+ArcFace: {accuracy:.2f}% ({correct_matches}/{total_probes})")
    
    # Tính EER
    if len(y_true) > 0 and len(set(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    else:
        eer = 0.0
        
    print(f"=> Equal Error Rate (EER): {eer*100:.2f}%")
    
    # Ghi chú kết quả ra file txt
    from datetime import datetime
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_config_str = OmegaConf.to_yaml(cfg.get('model', {}))
    
    result_txt_path = os.path.join(save_dir, "eval_results.txt")
    with open(result_txt_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{current_time}] === KẾT QUẢ ĐÁNH GIÁ (BASELINE 2 - ResNet+ArcFace) ===\n")
        f.write(f"Dataset: {dataset_name} ({data_dir})\n")
        f.write(f"Total Persons: {len(person_indices)}\n")
        f.write(f"Total Probes: {total_probes}\n")
        f.write(f"Matching Accuracy (Rank-1): {accuracy:.2f}%\n")
        f.write(f"Equal Error Rate (EER): {eer*100:.2f}%\n")
        f.write("--- Model Configuration ---\n")
        f.write(f"{model_config_str}\n")
        f.write("="*70 + "\n")
    print(f"Đã ghi thêm (append) kết quả đánh giá tại: {result_txt_path}")

if __name__ == '__main__':
    main()
