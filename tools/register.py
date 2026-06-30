import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import DatasetFactory
from src.engine.represent import optimize_r_in_projected_space
from tools.utils import set_seed, setup_logger, load_models

class Registerer:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.use_raw_mu = config.get('testing', {}).get('use_raw_mu', True)
        self.loss_type = config.get('testing', {}).get('gallery_loss', 'bce')
        self.num_samples = config.get('testing', {}).get('gallery_samples', 256)
        self.max_steps = config.get('testing', {}).get('gallery_max_steps', 100)
        self.lr = config.get('testing', {}).get('gallery_lr', 0.01)

    def extract_features(self, images):
        """Trích xuất mu và logvar từ tập các ảnh của cùng 1 người."""
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self.model(images, decode=False)
        return outputs['mu'], outputs['logvar']

    def register_person(self, person_images, all_other_mu=None, all_other_logvar=None):
        """
        Đăng ký 1 người:
        Nếu dùng raw mu, trả về trung bình các vector mu (hoặc tập hợp các mu).
        Nếu dùng optimize_r, chạy gradient descent.
        """
        mu_c, logvar_c = self.extract_features(person_images)
        
        optimize_r = self.config.get('testing', {}).get('optimize_r', True)
        
        if self.use_raw_mu:
            # Lưu lại tất cả mẫu (hoặc trung bình, ở đây lưu tất cả cho độ chính xác cao)
            return mu_c.cpu()
        elif optimize_r:
            # Optimize r
            r_cs = []
            for i in range(mu_c.size(0)):
                r_c = optimize_r_in_projected_space(
                    mu_c[i:i+1], logvar_c[i:i+1], all_other_mu, all_other_logvar,
                    model=self.model, device=self.device, config=self.config,
                    num_samples=self.num_samples, loss_type=self.loss_type, 
                    max_steps=self.max_steps, lr=self.lr, verbose=False
                )
                r_cs.append(r_c)
            return torch.stack(r_cs).cpu()
        else:
            # Dùng trực tiếp kết quả từ projector (không tối ưu)
            with torch.no_grad():
                proj_mu = self.model.projector(mu_c)
                proj_mu = torch.nn.functional.normalize(proj_mu, p=2, dim=1)
            return proj_mu.cpu()


def process_registration(palm_model, train_dataset, device, logger, config, output_dir, dataset_name):
    # Lấy tối đa 5 ảnh / 1 người theo user request
    max_images_per_person = config.get('testing', {}).get('max_images_per_person', 5)
    
    person_images = {}
    logger.info("Grouping images by person...")
    for img, label in tqdm(train_dataset, desc="Reading dataset"):
        label_val = label.item() if isinstance(label, torch.Tensor) else label
        if label_val not in person_images:
            person_images[label_val] = []
        if len(person_images[label_val]) < max_images_per_person:
            person_images[label_val].append(img)
            
    logger.info(f"Found {len(person_images)} unique identities.")
    
    registerer = Registerer(palm_model, device, config)
    gallery = {}
    
    use_raw_mu = config.get('testing', {}).get('use_raw_mu', True)
    optimize_r = config.get('testing', {}).get('optimize_r', True)
    
    if not use_raw_mu and optimize_r:
        # Nếu cần optimize, trích xuất trước toàn bộ mu, logvar để làm negative samples
        logger.info("Extracting global features for negative sampling...")
        all_mu = []
        all_logvar = []
        all_labels = []
        loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        palm_model.eval()
        with torch.no_grad():
            for imgs, lbls in tqdm(loader):
                imgs = imgs.to(device)
                outputs = palm_model(imgs, decode=False)
                all_mu.append(outputs['mu'])
                all_logvar.append(outputs['logvar'])
                all_labels.append(lbls.to(device))
                
        global_mu = torch.cat(all_mu, dim=0)
        global_logvar = torch.cat(all_logvar, dim=0)
        global_labels = torch.cat(all_labels, dim=0)
    
    logger.info("Registering persons one by one...")
    for person_id, img_list in tqdm(person_images.items(), desc="Registering"):
        imgs_tensor = torch.stack(img_list)
        
        if not use_raw_mu and optimize_r:
            idx_others = (global_labels != person_id).nonzero(as_tuple=True)[0]
            other_mu = global_mu[idx_others]
            other_logvar = global_logvar[idx_others]
            r_vectors = registerer.register_person(imgs_tensor, other_mu, other_logvar)
        else:
            r_vectors = registerer.register_person(imgs_tensor)
            
        gallery[person_id] = r_vectors
        
    gallery_save_path = os.path.join(output_dir, f'gallery_{dataset_name.lower()}.pt')
    torch.save(gallery, gallery_save_path)
    logger.info(f"Saved gallery to {gallery_save_path}")
    return gallery_save_path

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    seed = config.get('seed', 42)
    checkpoint_path = config.get('checkpoint', '')
    
    version_dir = "logs/unversioned_results"
    if checkpoint_path:
        match = re.search(r'(.*[\\/]version_\d+)', checkpoint_path.replace('\\', '/'))
        if match:
            version_dir = match.group(1)
            # Load run_config.yaml if it exists to keep previous settings (e.g. model settings)
            run_cfg_path = os.path.join(version_dir, 'run_config.yaml')
            if os.path.isfile(run_cfg_path):
                try:
                    run_cfg = OmegaConf.load(run_cfg_path)
                    for k, v in run_cfg.items():
                        config[k] = OmegaConf.to_container(v, resolve=True) if isinstance(v, DictConfig) else v
                    print(f"Loaded existing configuration from {run_cfg_path}")
                except Exception as e:
                    print(f"Warning: failed to load {run_cfg_path}: {e}")
            
            # Load config_backup.yaml from training to get model definition
            backup_cfg_path = os.path.join(version_dir, 'config_backup.yaml')
            if os.path.isfile(backup_cfg_path):
                try:
                    backup_cfg = OmegaConf.load(backup_cfg_path)
                    if 'model' in backup_cfg:
                        config['model'] = OmegaConf.to_container(backup_cfg['model'], resolve=True)
                except: pass

    # Cập nhật từ lệnh người dùng lên trên cùng (Ghi đè lại nếu có sửa từ CLI)
    cli_cfg = OmegaConf.to_container(cfg, resolve=True)
    if 'testing' not in config:
        config['testing'] = {}
    for k, v in cli_cfg.get('testing', {}).items():
        config['testing'][k] = v

    os.makedirs(version_dir, exist_ok=True)
    set_seed(seed)

    logger, log_file, timestamp = setup_logger(version_dir, 'register_pipeline')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    palm_model = load_models(config, device, logger, checkpoint_path)
    
    data_dir = config.get('dataset', {}).get('data_dir', 'data/PolyU')
    dataset_name = config.get('dataset', {}).get('name', 'PolyU')
    if dataset_name.upper() == 'POLYU':
        dataset_name = 'PalmPrintDataset'
    elif dataset_name.upper() == 'MNIST':
        dataset_name = 'MNISTDataset'
    
    # Save the updated configuration for attendance to use
    run_cfg_path = os.path.join(version_dir, 'run_config.yaml')
    with open(run_cfg_path, 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(config, f, allow_unicode=True)
    
    logger.info(f"Dataset Name: {dataset_name}")
    try:
        train_dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=config.get('dataset', {}), is_train=True)
    except ValueError as e:
        logger.error(str(e))
        return

    process_registration(palm_model, train_dataset, device, logger, config, version_dir, dataset_name)
    logger.info("Registration complete.")

if __name__ == '__main__':
    main()
