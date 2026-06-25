import unittest
import torch
import os
import shutil
from src.datasets.palm_dataset import PalmPrintDataset

class TestPalmPrintDataset(unittest.TestCase):
    def setUp(self):
        self.config = {
            'image_size': [128, 128]
        }
        self.dummy_data_dir = 'dummy_test_data_dir'
        
        # Tạo thư mục tạm để test nếu muốn test có load data (tùy chọn)
        # Ở đây dùng dry_run như trong code đã hỗ trợ (tự generate torch.randn)
        
    def tearDown(self):
        if os.path.exists(self.dummy_data_dir):
            shutil.rmtree(self.dummy_data_dir)

    def test_dataset_initialization_dry_run(self):
        # Khi thư mục không tồn tại, code có cơ chế trả về dummy size=100
        dataset = PalmPrintDataset(self.dummy_data_dir, self.config, is_train=True)
        self.assertEqual(len(dataset), 100, "Dummy dataset phải có độ dài 100")
        
    def test_dataset_getitem_dry_run(self):
        dataset = PalmPrintDataset(self.dummy_data_dir, self.config, is_train=False)
        
        # Test lấy 1 mẫu
        img, label = dataset[0]
        
        # Shape phải là [3, 128, 128] do config
        self.assertEqual(img.shape, (3, 128, 128), "Image shape không khớp với config")
        self.assertIsInstance(label, int, "Label phải là kiểu int")
        
    def test_transforms_exist(self):
        dataset_train = PalmPrintDataset(self.dummy_data_dir, self.config, is_train=True)
        dataset_test = PalmPrintDataset(self.dummy_data_dir, self.config, is_train=False)
        
        # Kiểm tra object có transform
        self.assertIsNotNone(dataset_train.transform)
        self.assertIsNotNone(dataset_test.transform)
        
        # Số lượng transform train thường nhiều hơn test do có Augmentation
        self.assertTrue(len(dataset_train.transform.transforms) > len(dataset_test.transform.transforms))

if __name__ == '__main__':
    unittest.main()
