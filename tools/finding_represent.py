import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.datasets import DatasetFactory
from src.engine.represent import optimize_representation

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    checkpoint_path = config.get('checkpoint', '')
    samples = config.get('samples', 512)
    steps = config.get('steps', 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    # 1. Khởi tạo mô hình dựa vào config
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'default')
    
    if 'decoder' not in model_config:
        model_config['decoder'] = {}
    if 'image_size' not in model_config['decoder']:
        model_config['decoder']['image_size'] = config.get('dataset', {}).get('image_size', [128, 128])
    
    if model_type == 'unet':
        print("Sử dụng kiến trúc U-Net (UNetPalmModel)")
        model = UNetPalmModel(model_config).to(device)
    else:
        print("Sử dụng kiến trúc VAE (ProbabilisticPalmModel)")
        model = ProbabilisticPalmModel(model_config).to(device)
    
    # 2. Load checkpoint nếu có
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Đang tải checkpoint từ {checkpoint_path}...")
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint_data.get('model_state_dict', checkpoint_data.get('state_dict', checkpoint_data))
        
        # Loại bỏ prefix 'model.' của PyTorch Lightning nếu có
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_state_dict[k[6:]] = v
            else:
                clean_state_dict[k] = v
                
        model.load_state_dict(clean_state_dict, strict=False)
        print("Tải checkpoint thành công.")
    else:
        print("Cảnh báo: Không có checkpoint hợp lệ. Chạy với mô hình khởi tạo ngẫu nhiên!")

    # 3. Khởi tạo Dataset
    dataset_config = config.get('dataset', {})
    data_dir = dataset_config.get('data_dir', 'data/collect')
    dataset_name = dataset_config.get('name', 'OwnOriginalDataset')
    
    print(f"\nĐang tải dữ liệu từ {dataset_name} ({data_dir})...")
    try:
        dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=dataset_config, is_train=False)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        # Lấy 2 ảnh thật từ tập dữ liệu (để dự phòng cho mode average)
        real_image, label = next(iter(dataloader))
        real_image = real_image.to(device)
        print(f"Đã lấy batch ảnh thật (shape: {real_image.shape}) để tìm kiếm r.")
    except Exception as e:
        print(f"Lỗi khi load dataset: {e}. Fallback về ảnh nhiễu (Dummy Data).")
        img_size = dataset_config.get('image_size', [128, 128])
        real_image = torch.randn(2, 3, img_size[0], img_size[1]).to(device)
    
    # 4. Tìm kiếm r (Optimize Representation)
    print("\nBắt đầu quy trình giả lập Inference (FINDING r)...")
    optimized_r = optimize_representation(
        model=model,
        image=real_image,
        config=config,
        device=device,
        num_samples=samples,
        steps=steps,
        lr=0.01,
        alpha=0.1
    )
    
    print(f"\nVector r cuối cùng có shape: {optimized_r.shape}")

if __name__ == '__main__':
    main()
