import os
from collections import defaultdict
from .base import BaseDataset
import random

class IITDDataset(BaseDataset):
    """
    IITD Palmprint V1 Dataset.
    Cấu trúc thư mục dự kiến:
    - data/IITD Palmprint V1/Segmented/Left
    - data/IITD Palmprint V1/Segmented/Right
    
    Quy luật: Tên file dạng <id>_<num>.bmp (vd: 001_01.bmp)
    ID của người là phần trước dấu gạch dưới `_`.
    """
    def __init__(self, data_dir: str, config: dict, is_train: bool = True):
        self.split_mode = config.get('split_mode', 'hand') # 'hand' hoặc 'ratio'
        self.train_ratio = config.get('train_ratio', 0.8)
        self.seed = config.get('seed', 42)
        
        self.left_dir = os.path.join(data_dir, 'Left')
        self.right_dir = os.path.join(data_dir, 'Right')
        
        # Dictionary map string ID (vd '001') -> int ID (vd 0)
        self.id_to_label = {}
        
        super().__init__(data_dir, config, is_train)
        
    def _parse_id(self, filename):
        # Tách tên file "001_01.bmp" -> "001"
        basename = os.path.splitext(filename)[0]
        person_id_str = basename.split('_')[0]
        
        if person_id_str not in self.id_to_label:
            self.id_to_label[person_id_str] = len(self.id_to_label)
            
        return self.id_to_label[person_id_str]
        
    def _get_files(self, folder):
        if not os.path.exists(folder):
            return []
        valid_exts = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
        return [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
        
    def _load_data(self):
        left_files = self._get_files(self.left_dir)
        right_files = self._get_files(self.right_dir)
        
        if self.split_mode == 'hand':
            # Kịch bản 1: Left = train, Right = val
            target_dir = self.left_dir if self.is_train else self.right_dir
            files = left_files if self.is_train else right_files
            
            for f in files:
                label = self._parse_id(f)
                self.samples.append((os.path.join(target_dir, f), label))
                
        elif self.split_mode == 'ratio':
            # Kịch bản 2: Gộp chung Left và Right, chia theo tỷ lệ
            person_dict = defaultdict(list)
            
            for f in left_files:
                label = self._parse_id(f)
                person_dict[label].append(os.path.join(self.left_dir, f))
                
            for f in right_files:
                label = self._parse_id(f)
                person_dict[label].append(os.path.join(self.right_dir, f))
                
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
            raise ValueError(f"Không hỗ trợ split_mode: {self.split_mode} cho IITDDataset")
            
        print(f"Loaded IITDDataset ({self.split_mode} mode) - {'Train' if self.is_train else 'Val'}: {len(self.samples)} samples. Total IDs: {len(self.id_to_label)}")

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
