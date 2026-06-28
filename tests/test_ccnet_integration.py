import os
import sys
import unittest
import torch
from omegaconf import OmegaConf
import hydra

# Thêm đường dẫn gốc để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.data_module import DatasetFactory
from src.models.unet_model import UNetPalmModel

class TestCCNetIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Thiết lập Hydra để giả lập lệnh chạy từ terminal
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../config")
        # Load cấu hình với dataset own_original và model unet_ccnet
        cls.cfg = hydra.compose(config_name="config", overrides=["dataset=own_original", "model=unet_ccnet"])
        
    def test_forward_pass_with_ccnet(self):
        config = OmegaConf.to_container(self.cfg, resolve=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Khởi tạo Dataset
        data_dir = config.get('dataset', {}).get('data_dir', 'data/own_original')
        dataset_name = config.get('dataset', {}).get('name', 'OwnOriginal')
        
        try:
            train_dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=config.get('dataset', {}), is_train=True)
        except Exception as e:
            self.skipTest(f"Skipping because dataset couldn't be loaded (maybe missing data folder): {e}")
            
        self.assertGreater(len(train_dataset), 0, "Dataset rỗng!")
        
        # Lấy 1 batch nhỏ để test
        batch_size = 2
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        images, labels = next(iter(dataloader))
        images = images.to(device)
        
        # 2. Khởi tạo Model (UNetPalmModel sẽ tự gọi PalmEncoder -> CCNetBackbone)
        model_config = config.get('model', {})
        if 'decoder' not in model_config:
            model_config['decoder'] = {}
        model_config['decoder']['image_size'] = config.get('dataset', {}).get('image_size', [128, 128])
        
        print("\nKhởi tạo mô hình UNet với CCNet Backbone...")
        model = UNetPalmModel(model_config).to(device)
        
        # 3. Chạy Forward Pass
        print("Đang chạy forward pass (ảnh đầu vào shape: {})...".format(images.shape))
        model.eval()
        with torch.no_grad():
            outputs = model(images, decode=True)
            
        # 4. Kiểm tra đầu ra
        self.assertIn('mu', outputs)
        self.assertIn('logvar', outputs)
        self.assertIn('x_hat', outputs)
        
        mu = outputs['mu']
        logvar = outputs['logvar']
        recon = outputs['x_hat']
        
        print(f"Hoàn thành! Shape của mu: {mu.shape}, logvar: {logvar.shape}")
        
        expected_latent_dim = model_config.get('encoder', {}).get('latent_dim', 128)
        self.assertEqual(mu.shape, (batch_size, expected_latent_dim))
        self.assertEqual(logvar.shape, (batch_size, expected_latent_dim))
        self.assertEqual(recon.shape, images.shape, "Reconstruction phải cùng kích thước ảnh đầu vào")

if __name__ == '__main__':
    unittest.main()
