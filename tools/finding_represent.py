import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.models.palm_model import ProbabilisticPalmModel
from src.engine.represent import optimize_representation

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    checkpoint_path = config.get('checkpoint', '')
    samples = config.get('samples', 512)
    steps = config.get('steps', 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    # Khởi tạo mô hình
    model = ProbabilisticPalmModel(config).to(device)
    
    # Load checkpoint nếu có
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Đang tải checkpoint từ {checkpoint_path}...")
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        else:
            model.load_state_dict(checkpoint_data)
        print("Tải checkpoint thành công.")
    else:
        print("Cảnh báo: Không có checkpoint nào được cung cấp. Chạy với mô hình khởi tạo ngẫu nhiên để Test!")

    # Tạo 1 ảnh Dummy Data (1, C, H, W)
    img_size = config.get('dataset', {}).get('image_size', [128, 128])
    dummy_image = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
    
    print("\nBắt đầu quy trình giả lập Inference (FINDING r)...")
    optimized_r = optimize_representation(
        model=model,
        image=dummy_image,
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
