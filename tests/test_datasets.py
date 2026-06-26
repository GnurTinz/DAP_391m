import sys
import os
import unittest
import tempfile
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.datasets.tongji_dataset import TongjiDataset
from src.datasets.iitd_dataset import IITDDataset

class TestNewDatasets(unittest.TestCase):
    def setUp(self):
        # Tạo thư mục tạm để giả lập cấu trúc file
        self.test_dir = tempfile.mkdtemp()
        
        # --- Tạo cấu trúc ảo cho Tongji ---
        self.tongji_dir = os.path.join(self.test_dir, 'Tongji')
        os.makedirs(os.path.join(self.tongji_dir, 'session1'))
        os.makedirs(os.path.join(self.tongji_dir, 'session2'))
        
        # Giả lập 20 ảnh (2 người) cho session 1
        for i in range(20):
            open(os.path.join(self.tongji_dir, 'session1', f"{i:03d}.bmp"), 'w').close()
            
        # Giả lập 20 ảnh (2 người) cho session 2
        for i in range(20):
            open(os.path.join(self.tongji_dir, 'session2', f"{i:03d}.bmp"), 'w').close()
            
            
        # --- Tạo cấu trúc ảo cho IITD ---
        self.iitd_dir = os.path.join(self.test_dir, 'IITD')
        os.makedirs(os.path.join(self.iitd_dir, 'Left'))
        os.makedirs(os.path.join(self.iitd_dir, 'Right'))
        
        # Người 001 có 5 ảnh bên Left, 5 ảnh bên Right
        for i in range(5):
            open(os.path.join(self.iitd_dir, 'Left', f"001_{i:02d}.bmp"), 'w').close()
            open(os.path.join(self.iitd_dir, 'Right', f"001_{i+5:02d}.bmp"), 'w').close()
            
        # Người 002 có 5 ảnh bên Left, 5 ảnh bên Right
        for i in range(5):
            open(os.path.join(self.iitd_dir, 'Left', f"002_{i:02d}.bmp"), 'w').close()
            open(os.path.join(self.iitd_dir, 'Right', f"002_{i+5:02d}.bmp"), 'w').close()

    def tearDown(self):
        # Dọn dẹp thư mục tạm
        shutil.rmtree(self.test_dir)

    def test_tongji_session_mode(self):
        config = {
            'split_mode': 'session',
            'image_size': [128, 128]
        }
        train_dataset = TongjiDataset(data_dir=self.tongji_dir, config=config, is_train=True)
        val_dataset = TongjiDataset(data_dir=self.tongji_dir, config=config, is_train=False)
        
        self.assertEqual(len(train_dataset.samples), 20)
        self.assertEqual(len(val_dataset.samples), 20)
        
        # Ảnh 0-9 là label 0, 10-19 là label 1
        self.assertEqual(train_dataset.samples[0][1], 0)
        self.assertEqual(train_dataset.samples[9][1], 0)
        self.assertEqual(train_dataset.samples[10][1], 1)
        self.assertEqual(train_dataset.samples[19][1], 1)

    def test_tongji_mixed_mode(self):
        config = {
            'split_mode': 'mixed',
            'train_ratio': 0.8,
            'image_size': [128, 128],
            'seed': 42
        }
        train_dataset = TongjiDataset(data_dir=self.tongji_dir, config=config, is_train=True)
        val_dataset = TongjiDataset(data_dir=self.tongji_dir, config=config, is_train=False)
        
        # Mỗi người (label) có tổng cộng 20 ảnh (10 session1 + 10 session2)
        # Train 80% = 16 ảnh / người. Có 2 người => 32 ảnh train, 8 ảnh val
        self.assertEqual(len(train_dataset.samples), 32)
        self.assertEqual(len(val_dataset.samples), 8)

    def test_iitd_hand_mode(self):
        config = {
            'split_mode': 'hand',
            'image_size': [128, 128]
        }
        train_dataset = IITDDataset(data_dir=self.iitd_dir, config=config, is_train=True)
        val_dataset = IITDDataset(data_dir=self.iitd_dir, config=config, is_train=False)
        
        # Left làm train (10 ảnh), Right làm val (10 ảnh)
        self.assertEqual(len(train_dataset.samples), 10)
        self.assertEqual(len(val_dataset.samples), 10)
        
        # Đảm bảo parser ID hoạt động tốt ("001" và "002" chuyển thành 0 và 1)
        labels = train_dataset.get_labels()
        self.assertIn(0, labels)
        self.assertIn(1, labels)

    def test_iitd_ratio_mode(self):
        config = {
            'split_mode': 'ratio',
            'train_ratio': 0.8,
            'image_size': [128, 128],
            'seed': 42
        }
        train_dataset = IITDDataset(data_dir=self.iitd_dir, config=config, is_train=True)
        val_dataset = IITDDataset(data_dir=self.iitd_dir, config=config, is_train=False)
        
        # Tổng 20 ảnh (10 ảnh/người). Train 80% = 8 ảnh/người => 16 train, 4 val
        self.assertEqual(len(train_dataset.samples), 16)
        self.assertEqual(len(val_dataset.samples), 4)

if __name__ == '__main__':
    unittest.main()
