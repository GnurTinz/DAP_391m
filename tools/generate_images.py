import torch
from torch.utils.data import DataLoader
import hydra
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import DictConfig, OmegaConf

from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.datasets import DatasetFactory
from src.utils.generator import ImageGenerator

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
        
    # Đọc cấu hình 'generation' từ YAML
    gen_config = config.get('generation', {})
    mode = gen_config.get('mode', 'reconstruct')
    num_images = gen_config.get('num_images', 8)
    temperature = gen_config.get('temperature', 1.0)
    checkpoint_path = config.get('checkpoint', '')
    
    # Trích xuất version_dir từ checkpoint (nếu có)
    import re
    version_dir = "logs/unversioned_results"
    if checkpoint_path:
        match = re.search(r'(.*[\\/]version_\d+)', checkpoint_path.replace('\\', '/'))
        if match:
            version_dir = match.group(1)
            
    # Cấu hình output directory
    output_dir = os.path.join(version_dir, 'generated')
    os.makedirs(output_dir, exist_ok=True)
    
    # Tự động nạp cấu hình kiến trúc từ config_backup.yaml nếu có
    backup_path = os.path.join(version_dir, "config_backup.yaml")
    if os.path.exists(backup_path):
        print(f"Tìm thấy file {backup_path}, đang nạp để đồng bộ kiến trúc model...")
        import yaml
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_config = yaml.safe_load(f)
            
        if 'model' in backup_config:
            config['model'] = backup_config['model']
            
        if 'dataset' in backup_config and 'image_size' in backup_config['dataset']:
            if 'dataset' not in config:
                config['dataset'] = {}
            config['dataset']['image_size'] = backup_config['dataset']['image_size']
            
    output_path = config.get('output', "")
    if not output_path or output_path == "logs/results":
        output_path = os.path.join(output_dir, f"{mode}.png")
    else:
        # Nếu người dùng chỉ truyền tên file (không có đường dẫn), đưa vào thư mục output_dir
        if not os.path.dirname(output_path):
            output_path = os.path.join(output_dir, output_path)
            # Đảm bảo có đuôi file ảnh
            if not any(output_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                output_path += '.png'
        
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
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint_data.get('model_state_dict', checkpoint_data.get('state_dict', checkpoint_data))
        
        # PyTorch Lightning lưu checkpoint kèm prefix "model." do class GenerativeLightningModule bọc model bên trong
        # Ta cần loại bỏ prefix này trước khi load vào UNetPalmModel / ProbabilisticPalmModel
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_state_dict[k[6:]] = v
            else:
                clean_state_dict[k] = v
                
        model.load_state_dict(clean_state_dict, strict=False)
        print(f"Đã nạp trọng số từ: {checkpoint_path}")
    else:
        print("CẢNH BÁO: Chưa cung cấp checkpoint hợp lệ. Ảnh xuất ra sẽ là nhiễu ngẫu nhiên chưa qua huấn luyện.")
    
    # 3. Tải dataloader nếu cần
    dataloader = None
    if mode in ['reconstruct', 'variations', 'contrastive', 'latent_sampling', 'average', 'interpolate']:
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
            bs = 6
        elif mode == 'reconstruct':
            bs = num_images
        elif mode in ['average', 'interpolate']:
            bs = 2
        else:
            bs = 1
            
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
        
    # 4. Sử dụng bộ khung chung (ImageGenerator) để sinh ảnh
    generator = ImageGenerator(model, dataloader, device)
    
    if mode == 'reconstruct':
        generator.generate_reconstruction(num_images=num_images, output_path=output_path)
    elif mode == 'unconditional':
        generator.generate_unconditional(num_images=num_images, latent_dim=latent_dim, output_path=output_path)
    elif mode == 'variations':
        generator.generate_variations(num_variations=num_images, temperature=temperature, output_path=output_path)
    elif mode == 'contrastive':
        generator.generate_contrastive(output_path=output_path)
    elif mode == 'latent_sampling':
        generator.generate_from_latent(num_images=num_images, output_path=output_path)
    elif mode == 'average':
        generator.generate_average(num_images=num_images, temperature=temperature, output_path=output_path)
    elif mode == 'interpolate':
        generator.generate_interpolate(num_images=num_images, output_path=output_path)

if __name__ == '__main__':
    main()
