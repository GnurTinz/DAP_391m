import torch
import argparse
import os
import yaml
from torch.utils.data import DataLoader

from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.datasets.mnist_dataset import MNISTDataset
from src.utils.generator import ImageGenerator
# Thêm import cho PalmDataset nếu dùng

def main():
    parser = argparse.ArgumentParser(description="Generate or Sample (Reconstruct) images from trained model")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='logs/generated_samples.png', help='Output image path')
    parser.add_argument('--num_images', type=int, default=8, help='Number of images to generate/visualize')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature to scale the latent noise (for variations mode)')
    
    # Mode sinh ảnh: unconditional, reconstruct, variations
    parser.add_argument('--mode', type=str, choices=['unconditional', 'reconstruct', 'variations'], default='variations',
                        help='Mode sinh ảnh: unconditional (từ nhiễu z), reconstruct (nén và giải nén) hoặc variations (sinh biến thể 1 ảnh)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")
    
    # 1. Khởi tạo mô hình dựa vào loại kiến trúc trong config
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'default')
    
    # Xử lý đồng bộ image_size (nếu có)
    if 'decoder' not in model_config:
        model_config['decoder'] = {}
    if 'image_size' not in model_config['decoder']:
        model_config['decoder']['image_size'] = config.get('dataset', {}).get('image_size', [128, 128])
    
    if model_type == 'unet':
        model = UNetPalmModel(model_config).to(device)
        # U-Net bắt buộc dùng reconstruct hoặc variations
        if args.mode == 'unconditional':
            print("CẢNH BÁO: Mô hình U-Net không hỗ trợ mode 'unconditional'. Tự động chuyển sang 'variations'.")
            args.mode = 'variations'
    else:
        model = ProbabilisticPalmModel(model_config).to(device)
        
    latent_dim = model_config.get('encoder', {}).get('latent_dim', 128)
    
    # 2. Tải trọng số
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Đã nạp trọng số từ: {args.checkpoint}")
    else:
        print("CẢNH BÁO: Chưa cung cấp checkpoint hợp lệ. Ảnh xuất ra sẽ là nhiễu ngẫu nhiên chưa qua huấn luyện.")
    
    # 3. Tải dataloader nếu cần
    dataloader = None
    if args.mode in ['reconstruct', 'variations']:
        print(f"Đang tải dữ liệu để lấy mẫu gốc cho mode '{args.mode}'...")
        data_dir = config.get('dataset', {}).get('data_dir', 'data/MNIST')
        dataset = MNISTDataset(data_dir=data_dir, config=config.get('dataset', {}), is_train=False)
        dataloader = DataLoader(dataset, batch_size=args.num_images if args.mode == 'reconstruct' else 1, shuffle=True)
        
    # 4. Sử dụng bộ khung chung (ImageGenerator) để sinh ảnh
    generator = ImageGenerator(model, dataloader, device)
    
    if args.mode == 'reconstruct':
        if args.output == 'logs/generated_samples.png':
            args.output = 'logs/reconstructed.png'
        generator.generate_reconstruction(num_images=args.num_images, output_path=args.output)
    elif args.mode == 'unconditional':
        generator.generate_unconditional(num_images=args.num_images, latent_dim=latent_dim, output_path=args.output)
    elif args.mode == 'variations':
        if args.output == 'logs/generated_samples.png':
            args.output = 'logs/variations.png'
        generator.generate_variations(num_variations=args.num_images, temperature=args.temperature, output_path=args.output)

if __name__ == '__main__':
    main()
