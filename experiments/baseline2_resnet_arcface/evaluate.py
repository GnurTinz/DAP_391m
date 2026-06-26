import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

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
    weight_path = os.path.join(save_dir, "resnet_arcface.pth")
    
    if not os.path.exists(meta_path) or not os.path.exists(weight_path):
        print("Lỗi: Không tìm thấy model hoặc meta data. Vui lòng chạy train.py trước!")
        sys.exit(1)
        
    meta = torch.load(meta_path)
    num_classes = meta['num_classes']
    
    # Init model
    model = ResNetArcFace(num_classes=num_classes, feature_dim=512, pretrained=False).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
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
        idx_to_use = indices[:2]
        if dataset is not None and hasattr(dataset, 'samples'):
            images = [dataset[idx][0] for idx in idx_to_use]
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
    
    for probe_label, indices in tqdm(val_indices.items(), desc="Probe Matching"):
        if len(indices) == 0:
            continue
            
        idx_to_use = indices[:2]
        images = [dataset[idx][0] for idx in idx_to_use]
        batch_images = torch.stack(images).to(device)
            
        with torch.no_grad():
            probe_feats = model(batch_images)
            probe_feats = torch.nn.functional.normalize(probe_feats, p=2, dim=1)
            probe_avg = probe_feats.mean(dim=0, keepdim=True).cpu()
            probe_avg = torch.nn.functional.normalize(probe_avg, p=2, dim=1)
            
        best_match = None
        best_sim = -float('inf')
        
        for gal_label, gal_feat in gallery_features.items():
            # Cosine similarity
            sim = torch.mm(probe_avg, gal_feat.t()).item()
            if sim > best_sim:
                best_sim = sim
                best_match = gal_label
                
        is_correct = best_match == probe_label
        if is_correct:
            correct_matches += 1
        total_probes += 1
        
    accuracy = (correct_matches / total_probes) * 100 if total_probes > 0 else 0
    print(f"\n=> Matching Accuracy (Rank-1) using ResNet+ArcFace: {accuracy:.2f}% ({correct_matches}/{total_probes})")

if __name__ == '__main__':
    main()
