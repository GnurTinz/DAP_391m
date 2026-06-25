import os
import sys
import argparse
import yaml
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

def main():
    parser = argparse.ArgumentParser(description="Find optimal representation r (Test-Time Optimization)")
    parser.add_argument('--config', type=str, default='config/mnist.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (optional for dummy test)')
    parser.add_argument('--samples', type=int, default=512, help='Number of pos/neg samples (N)')
    parser.add_argument('--steps', type=int, default=50, help='Number of optimization steps (T)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    # Khởi tạo mô hình
    model = ProbabilisticPalmModel(config).to(device)
    
    # Load checkpoint nếu có
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Đang tải checkpoint từ {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
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
        num_samples=args.samples,
        steps=args.steps,
        lr=0.01,
        alpha=0.1
    )
    
    print(f"\nVector r cuối cùng có shape: {optimized_r.shape}")

if __name__ == '__main__':
    main()
