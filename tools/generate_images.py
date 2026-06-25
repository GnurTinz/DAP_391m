import torch
import torchvision.utils as vutils
import argparse
import os
import yaml
from src.models.palm_model import ProbabilisticPalmModel
import matplotlib.pyplot as plt

def generate_and_save_images(model, num_images=16, latent_dim=128, device='cpu', output_path='logs/generated_samples.png'):
    model.eval()
    with torch.no_grad():
        # Sinh ngẫu nhiên vector z từ phân phối chuẩn
        z = torch.randn(num_images, latent_dim).to(device)
        
        # Qua bộ giải mã (Decoder) để sinh ảnh
        # Lưu ý: model.decoder nhận z và trả về x_hat
        generated_images = model.decoder(z)
        
        # Đưa giá trị pixel từ [-1, 1] về [0, 1] để lưu
        generated_images = (generated_images + 1) / 2.0
        
        # Tạo lưới ảnh và lưu
        vutils.save_image(generated_images, output_path, nrow=int(num_images**0.5), padding=2, normalize=False)
        print(f"Đã lưu {num_images} ảnh sinh ngẫu nhiên tại: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate detailed images from trained VAE")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='logs/generated_samples.png', help='Output image path')
    parser.add_argument('--num_images', type=int, default=16, help='Number of images to generate')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")
    
    # Khởi tạo mô hình
    model = ProbabilisticPalmModel(config['model']).to(device)
    latent_dim = config['model']['encoder']['latent_dim']
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Đã nạp trọng số từ: {args.checkpoint}")
    else:
        print("CẢNH BÁO: Không có checkpoint được cung cấp hoặc tìm thấy. Ảnh sinh ra sẽ là nhiễu ngẫu nhiên chưa qua huấn luyện.")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_and_save_images(model, args.num_images, latent_dim, device, args.output)

if __name__ == '__main__':
    main()
