import torch
import argparse
import os
import yaml
from torch.utils.data import DataLoader

from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.datasets import DatasetFactory
from src.utils.generator import ImageGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate or Sample (Reconstruct) images from trained model")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to model checkpoint')
    
    # Cấu hình qua tham số CLI sẽ ghi đè (overwrite) cấu hình trong YAML
    parser.add_argument('--output', type=str, default=None, help='Output image path (ghi đè YAML)')
    parser.add_argument('--num_images', type=int, default=None, help='Number of images to generate (ghi đè YAML)')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for variations mode (ghi đè YAML)')
    parser.add_argument('--mode', type=str, choices=['unconditional', 'reconstruct', 'variations', 'contrastive', 'latent_sampling'], default=None,
                        help='Mode sinh ảnh: unconditional, reconstruct, variations, contrastive, hoặc latent_sampling (ghi đè YAML)')
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Đọc cấu hình 'generation' từ YAML
    gen_config = config.get('generation', {})
    mode = args.mode if args.mode is not None else gen_config.get('mode', 'reconstruct')
    num_images = args.num_images if args.num_images is not None else gen_config.get('num_images', 8)
    temperature = args.temperature if args.temperature is not None else gen_config.get('temperature', 1.0)
    output_path = args.output if args.output is not None else gen_config.get('output_path', 'logs/generated_samples.png')

        
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
        if mode == 'unconditional':
            print("CẢNH BÁO: Mô hình U-Net không hỗ trợ mode 'unconditional'. Tự động chuyển sang 'variations'.")
            mode = 'variations'
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
    if mode in ['reconstruct', 'variations', 'contrastive', 'latent_sampling']:
        print(f"Đang tải dữ liệu để lấy mẫu gốc cho mode '{mode}'...")
        data_dir = config.get('dataset', {}).get('data_dir', 'data/MNIST')
        dataset_name = config.get('dataset', {}).get('name', 'MNISTDataset')
        
        # For backward compatibility with old configs
        if dataset_name.upper() == 'POLYU':
            dataset_name = 'PalmPrintDataset'
        elif dataset_name.upper() == 'MNIST':
            dataset_name = 'MNISTDataset'
            
        dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=config.get('dataset', {}), is_train=False)
        
        if mode == 'contrastive':
            bs = 64
        elif mode == 'reconstruct':
            bs = num_images
        else:
            bs = 1
            
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
        
    # 4. Sử dụng bộ khung chung (ImageGenerator) để sinh ảnh
    generator = ImageGenerator(model, dataloader, device)
    
    if mode == 'reconstruct':
        if args.output is None and 'output_path' not in gen_config:
            output_path = 'logs/reconstructed.png'
        generator.generate_reconstruction(num_images=num_images, output_path=output_path)
    elif mode == 'unconditional':
        generator.generate_unconditional(num_images=num_images, latent_dim=latent_dim, output_path=output_path)
    elif mode == 'variations':
        if args.output is None and 'output_path' not in gen_config:
            output_path = 'logs/variations.png'
        generator.generate_variations(num_variations=num_images, temperature=temperature, output_path=output_path)
    elif mode == 'contrastive':
        if args.output is None and 'output_path' not in gen_config:
            output_path = 'logs/contrastive.png'
        generator.generate_contrastive(output_path=output_path)
    elif mode == 'latent_sampling':
        if args.output is None and 'output_path' not in gen_config:
            output_path = 'logs/latent_sampling.png'
        generator.generate_from_latent(num_images=num_images, output_path=output_path)

if __name__ == '__main__':
    main()
