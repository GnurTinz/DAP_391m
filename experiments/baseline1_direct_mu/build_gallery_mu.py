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

from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.datasets import DatasetFactory

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    checkpoint_path = config.get('checkpoint', '')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    # 1. Tải checkpoint nếu có
    checkpoint_data = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Đang tải checkpoint từ {checkpoint_path}...")
        checkpoint_data = torch.load(checkpoint_path, map_location=device)

    # 2. Khởi tạo mô hình
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
    
    model.eval()

    # 3. Nạp trọng số từ checkpoint
    if checkpoint_data:
        state_dict = checkpoint_data.get('model_state_dict', checkpoint_data.get('state_dict', checkpoint_data))
        clean_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
        print("Tải trọng số checkpoint thành công.")
    else:
        print("Cảnh báo: Không có checkpoint hợp lệ.")

    # 4. Khởi tạo Dataset
    dataset_config = config.get('dataset', {})
    data_dir = dataset_config.get('data_dir', 'data/collect')
    dataset_name = dataset_config.get('name', 'OwnOriginalDataset')
    
    print(f"\nĐang tải dữ liệu từ {dataset_name} ({data_dir})...")
    dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=dataset_config, is_train=False)
    
    train_person_indices = {}
    if hasattr(dataset, 'samples') and len(dataset.samples) > 0:
        person_indices = {}
        for i, item in enumerate(dataset.samples):
            label = item[1]
            if label not in person_indices:
                person_indices[label] = []
            person_indices[label].append(i)
            
        for label, indices in person_indices.items():
            split_idx = max(1, len(indices) // 2)
            train_person_indices[label] = indices[:split_idx]
    else:
        train_person_indices = {0: [0]}
        
    print(f"Tổng số Person trên tập Train: {len(train_person_indices)}")

    # ==========================================
    # 5. Trích xuất mu trên tập Train (Gallery)
    # ==========================================
    print("\n" + "="*50)
    print(" BASELINE 1: XÂY DỰNG GALLERY TỪ MU_TRAIN ")
    print("="*50)
    
    gallery_mu = {}
    
    pbar = tqdm(train_person_indices.items(), total=len(train_person_indices), desc="Trích xuất mu")
    for i, (person_id, indices) in enumerate(pbar):
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
            # Chế độ 'average' (lấy trung bình các mu của ảnh)
            mu_avg = mu_val.mean(dim=0, keepdim=True)
        
        gallery_mu[person_id] = mu_avg.cpu().detach()

    # ==========================================
    # 6. Lưu Gallery mu
    # ==========================================
    if checkpoint_path:
        version_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    else:
        version_dir = "logs/unversioned_results"
        os.makedirs(version_dir, exist_ok=True)
        
    save_path = os.path.join(version_dir, 'gallery_mu.pt')
    torch.save({
        'gallery_mu': gallery_mu
    }, save_path)
    
    print(f"\nHoàn tất! Đã lưu Gallery mu (Baseline) vào: {save_path}")

if __name__ == '__main__':
    main()
