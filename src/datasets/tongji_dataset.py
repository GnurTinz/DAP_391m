import os
import glob
from collections import defaultdict
from .base import BaseDataset
import random

class TongjiDataset(BaseDataset):
    """
    Tongji Dataset.
    Cấu trúc thư mục:
    - data/Tongji/session1
    - data/Tongji/session2
    
    Quy luật: Sắp xếp theo tên file alphabet, cứ 10 file liên tiếp thuộc về 1 người.
    Mỗi session có 6000 ảnh => 600 người.
    """
    def __init__(self, data_dir: str, config: dict, is_train: bool = True):
        self.split_mode = config.get('split_mode', 'session') # 'session' hoặc 'mixed'
        self.train_ratio = config.get('train_ratio', 0.8)
        self.seed = config.get('seed', 42)
        
        self.session1_dir = os.path.join(data_dir, 'session1')
        self.session2_dir = os.path.join(data_dir, 'session2')
        
        super().__init__(data_dir, config, is_train)
        
    def _get_session_files(self, session_dir):
        if not os.path.exists(session_dir):
            return []
        # Lọc các file ảnh và sắp xếp alphabet để đảm bảo thứ tự
        valid_exts = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
        files = [f for f in os.listdir(session_dir) if f.lower().endswith(valid_exts)]
        files.sort()
        return files
        
    def _load_data(self):
        s1_files = self._get_session_files(self.session1_dir)
        s2_files = self._get_session_files(self.session2_dir)
        
        if self.split_mode == 'session':
            # Kịch bản 1: session1 = train, session2 = val
            if self.is_train:
                for i, f in enumerate(s1_files):
                    label = i // 10
                    self.samples.append((os.path.join(self.session1_dir, f), label))
            else:
                for i, f in enumerate(s2_files):
                    label = i // 10
                    self.samples.append((os.path.join(self.session2_dir, f), label))
                    
        elif self.split_mode == 'mixed':
            # Kịch bản 2: Trộn session1 và session2 lại, chia tỷ lệ
            person_dict = defaultdict(list)
            
            for i, f in enumerate(s1_files):
                label = i // 10
                person_dict[label].append(os.path.join(self.session1_dir, f))
                
            for i, f in enumerate(s2_files):
                label = i // 10
                person_dict[label].append(os.path.join(self.session2_dir, f))
                
            rng = random.Random(self.seed)
            for label, paths in person_dict.items():
                rng.shuffle(paths)
                split_idx = int(len(paths) * self.train_ratio)
                
                if self.is_train:
                    selected_paths = paths[:split_idx]
                else:
                    selected_paths = paths[split_idx:]
                    
                for p in selected_paths:
                    self.samples.append((p, label))
        else:
            raise ValueError(f"Không hỗ trợ split_mode: {self.split_mode} cho TongjiDataset")
            
        print(f"Loaded TongjiDataset ({self.split_mode} mode) - {'Train' if self.is_train else 'Val'}: {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples) if len(self.samples) > 0 else 100 # Mock length for dry run
        
    def __getitem__(self, idx):
        if len(self.samples) == 0:
            import torch
            return torch.zeros(3, self.image_size[0], self.image_size[1]), 0
            
        img_path, label = self.samples[idx]
        image = self._load_image(img_path)
        return image, label

    def get_labels(self):
        if not self.samples:
            return [0] * 100
        return [label for _, label in self.samples]
