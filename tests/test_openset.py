import unittest
import torch
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path

# Thêm đường dẫn thư mục gốc vào sys.path để import
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.iitd_dataset import IITDDataset
from src.datasets.tongji_dataset import TongjiDataset
from src.datasets.own_original_dataset import OwnOriginalDataset
from tools.eval_attendance import evaluate_open_set

class TestOpenSetDataset(unittest.TestCase):
    def setUp(self):
        # Tạo cấu trúc thư mục giả lập cho IITD
        self.test_dir = tempfile.mkdtemp()
        self.iitd_dir = os.path.join(self.test_dir, "IITD")
        os.makedirs(self.iitd_dir)
        
        left_dir = os.path.join(self.iitd_dir, "Left")
        right_dir = os.path.join(self.iitd_dir, "Right")
        os.makedirs(left_dir)
        os.makedirs(right_dir)
        
        # Giả lập 20 người, mỗi người có 10 ảnh (cả left/right = 5 ảnh mỗi bên)
        self.num_persons = 20
        self.imgs_per_person = 10
        for pid in range(1, self.num_persons + 1):
            for i in range(1, 6):
                # Tạo file rỗng làm ảnh
                open(os.path.join(left_dir, f"{pid:03d}_l_{i}.bmp"), "w").close()
                open(os.path.join(right_dir, f"{pid:03d}_r_{i}.bmp"), "w").close()
                
        # Tạo cấu trúc thư mục giả lập cho OwnOriginalDataset
        self.own_dir = os.path.join(self.test_dir, "Own")
        os.makedirs(self.own_dir)
        for pid in range(1, self.num_persons + 1):
            cls_dir = os.path.join(self.own_dir, f"person_{pid:03d}")
            left_own = os.path.join(cls_dir, "left")
            right_own = os.path.join(cls_dir, "right")
            os.makedirs(left_own)
            os.makedirs(right_own)
            for i in range(1, 6):
                open(os.path.join(left_own, f"img_{i}.png"), "w").close()
                open(os.path.join(right_own, f"img_{i}.png"), "w").close()
                
        # Tạo cấu trúc thư mục giả lập cho TongjiDataset (300 người)
        self.tongji_dir = os.path.join(self.test_dir, "Tongji")
        self.tongji_s1 = os.path.join(self.tongji_dir, "session1")
        self.tongji_s2 = os.path.join(self.tongji_dir, "session2")
        os.makedirs(self.tongji_s1)
        os.makedirs(self.tongji_s2)
        
        self.num_tongji_persons = 300
        for pid in range(self.num_tongji_persons):
            for i in range(10):
                idx = pid * 10 + i
                open(os.path.join(self.tongji_s1, f"{idx:05d}.bmp"), "w").close()
                open(os.path.join(self.tongji_s2, f"{idx:05d}.bmp"), "w").close()
                
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_iitd_person_split(self):
        # Config: Train 10 người, Known 5 người, Stranger 5 người
        config = {
            "split_mode": "person",
            "num_train_persons": 10,
            "num_known_persons": 5,
            "num_stranger_persons": 5,
            "register_ratio": 0.4, # 4 ảnh register, 6 ảnh probe (do tổng 10 ảnh)
            "seed": 42
        }
        
        # Test Train split
        config['split'] = 'train'
        train_ds = IITDDataset(self.iitd_dir, config, is_train=True)
        self.assertEqual(len(np.unique(train_ds.labels)), 10)
        self.assertEqual(len(train_ds), 10 * 10) # 10 người * 10 ảnh
        
        # Test Register split
        config['split'] = 'register'
        reg_ds = IITDDataset(self.iitd_dir, config, is_train=False)
        self.assertEqual(len(np.unique(reg_ds.labels)), 5)
        self.assertEqual(len(reg_ds), 5 * 4) # 5 người * 4 ảnh (ratio 0.4 * 10 = 4)
        
        # Test Probe split
        config['split'] = 'probe'
        probe_ds = IITDDataset(self.iitd_dir, config, is_train=False)
        self.assertEqual(len(np.unique(probe_ds.labels)), 5)
        self.assertEqual(len(probe_ds), 5 * 6) # 5 người * 6 ảnh
        
        # Test Stranger split
        config['split'] = 'stranger'
        str_ds = IITDDataset(self.iitd_dir, config, is_train=False)
        self.assertEqual(len(np.unique(str_ds.labels)), 5)
        self.assertEqual(len(str_ds), 5 * 10) # 5 người * 10 ảnh
        
        # Đảm bảo các tập người không bị trùng lặp
        train_ids = set(train_ds.labels)
        known_ids = set(reg_ds.labels) # register và probe dùng chung id
        str_ids = set(str_ds.labels)
        
        self.assertTrue(train_ids.isdisjoint(known_ids))
        self.assertTrue(train_ids.isdisjoint(str_ids))
        self.assertTrue(known_ids.isdisjoint(str_ids))

    def test_own_original_person_split(self):
        # Config: Train 10 người, Known 5 người, Stranger 5 người
        config = {
            "split_mode": "person",
            "num_train_persons": 10,
            "num_known_persons": 5,
            "num_stranger_persons": 5,
            "register_ratio": 0.4,
            "seed": 42
        }
        
        # Test Train split
        config['split'] = 'train'
        train_ds = OwnOriginalDataset(self.own_dir, config, is_train=True)
        self.assertEqual(len(np.unique(train_ds.labels)), 10)
        self.assertEqual(len(train_ds), 10 * 10) # 10 người * 10 ảnh
        
        # Test Register split
        config['split'] = 'register'
        reg_ds = OwnOriginalDataset(self.own_dir, config, is_train=False)
        self.assertEqual(len(np.unique(reg_ds.labels)), 5)
        self.assertEqual(len(reg_ds), 5 * 4) # 5 người * 4 ảnh (ratio 0.4 * 10 = 4)
        
        # Test Probe split
        config['split'] = 'probe'
        probe_ds = OwnOriginalDataset(self.own_dir, config, is_train=False)
        self.assertEqual(len(np.unique(probe_ds.labels)), 5)
        self.assertEqual(len(probe_ds), 5 * 6) # 5 người * 6 ảnh
        
        # Test Stranger split
        config['split'] = 'stranger'
        str_ds = OwnOriginalDataset(self.own_dir, config, is_train=False)
        self.assertEqual(len(np.unique(str_ds.labels)), 5)
        self.assertEqual(len(str_ds), 5 * 10) # 5 người * 10 ảnh
        
        # Đảm bảo các tập người không bị trùng lặp
        train_ids = set(train_ds.labels)
        known_ids = set(reg_ds.labels)
        str_ids = set(str_ds.labels)
        
        self.assertTrue(train_ids.isdisjoint(known_ids))
        self.assertTrue(train_ids.isdisjoint(str_ids))
        self.assertTrue(known_ids.isdisjoint(str_ids))

    def test_tongji_person_split(self):
        # Config: Train 200, Known 50, Stranger 50 -> Tổng = 300
        config = {
            "split_mode": "person",
            "num_train_persons": 200,
            "num_known_persons": 50,
            "num_stranger_persons": 50,
            "register_ratio": 0.5, # ratio 0.5 của 20 ảnh/người = 10 ảnh register
            "seed": 42
        }
        
        # Train split
        config['split'] = 'train'
        train_ds = TongjiDataset(self.tongji_dir, config, is_train=True)
        self.assertEqual(len(np.unique(train_ds.labels)), 200)
        self.assertEqual(len(train_ds), 200 * 20) # 200 người * 20 ảnh (10 từ s1 + 10 từ s2)
        
        # Register split
        config['split'] = 'register'
        reg_ds = TongjiDataset(self.tongji_dir, config, is_train=False)
        self.assertEqual(len(np.unique(reg_ds.labels)), 50)
        self.assertEqual(len(reg_ds), 50 * 10) # 50 người * 10 ảnh
        
        # Probe split
        config['split'] = 'probe'
        probe_ds = TongjiDataset(self.tongji_dir, config, is_train=False)
        self.assertEqual(len(np.unique(probe_ds.labels)), 50)
        self.assertEqual(len(probe_ds), 50 * 10) # 50 người * 10 ảnh
        
        # Stranger split
        config['split'] = 'stranger'
        str_ds = TongjiDataset(self.tongji_dir, config, is_train=False)
        self.assertEqual(len(np.unique(str_ds.labels)), 50)
        self.assertEqual(len(str_ds), 50 * 20) # 50 người * 20 ảnh
        
        # Disjoint check
        train_ids = set(train_ds.labels)
        known_ids = set(reg_ds.labels)
        str_ids = set(str_ds.labels)
        
        self.assertTrue(train_ids.isdisjoint(known_ids))
        self.assertTrue(train_ids.isdisjoint(str_ids))
        self.assertTrue(known_ids.isdisjoint(str_ids))


class TestOpenSetEvaluation(unittest.TestCase):
    def test_evaluate_open_set(self):
        device = torch.device('cpu')
        
        # Gallery: 3 người known (id: 0, 1, 2)
        # Mô phỏng proj vectors chiều dài 4
        gallery = {
            0: torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]),
            1: torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            2: torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        }
        
        # Known probes: 3 mẫu
        # Probe 0 -> khớp gallery 0
        # Probe 1 -> khớp gallery 1
        # Probe 2 -> cố ý làm kém để test FRR
        known_proj = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # id 0 (Sim = 1.0)
            [0.0, 1.0, 0.0, 0.0],  # id 1 (Sim = 1.0)
            [0.1, 0.0, 0.0, 0.9],  # id 2 (kém, Sim = 0.0) -> FRR
        ])
        known_labels = torch.tensor([0, 1, 2])
        
        # Stranger probes: 2 mẫu
        # Str 1 -> ngẫu nhiên
        # Str 2 -> vô tình cực kỳ giống id 1 -> sẽ gây ra FAR
        stranger_proj = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],  # stranger (Sim max = 0.0 với gallery)
            [0.0, 0.9, 0.0, 0.1],  # stranger (nhưng giống id 1, Sim ~0.9) -> FAR
        ])
        stranger_labels = torch.tensor([99, 100])
        
        # Chạy evaluation với threshold cứng = 0.5 để dễ test
        results = evaluate_open_set(
            gallery, known_proj, known_labels, stranger_proj, stranger_labels,
            device=device, eer_thresh=0.5
        )
        
        # Kiểm tra Rank-1 của known probes
        # [1.0, 1.0, 0.0] -> 2 đúng, 1 sai. Rank-1 = 2/3 = 66.67%
        self.assertAlmostEqual(results["open_set_rank1"], 66.67, places=1)
        
        # FAR: stranger nào có sim >= 0.5? Mẫu 2 có sim 0.9 -> FAR = 1/2 = 50%
        self.assertAlmostEqual(results["far"], 50.0)
        
        # FRR: known probe nào có sim đúng < 0.5? Mẫu 3 (id 2) có sim 0.0 -> FRR = 1/3 = 33.33%
        self.assertAlmostEqual(results["frr"], 33.33, places=1)
        
        # Kiểm tra log scores
        self.assertEqual(len(results["genuine_scores"]), 3)
        self.assertEqual(len(results["stranger_max_sims"]), 2)
        
        # Genuine scores thực tế
        expected_genuines = [1.0, 1.0, 0.0]
        for actual, expected in zip(results["genuine_scores"], expected_genuines):
            self.assertAlmostEqual(actual, expected, places=4)

if __name__ == '__main__':
    unittest.main()
