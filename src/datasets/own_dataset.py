import os
import torch
from torchvision import transforms
from PIL import Image
from typing import Any, Tuple, Dict
from .base import BaseDataset

class OwnDataset(BaseDataset):
    """
    Custom Dataset loader (OwnDataset).
    Reads data from a specified directory (e.g., data/scripts/script1).
    Expects subdirectories for each class.
    """
    
    def __init__(self, data_dir: str, config: Dict[str, Any], is_train: bool = True):
        super().__init__(data_dir, config, is_train)
    def _load_data(self) -> None:
        """
        Load dataset from directory structure.
        Expects: data_dir/class_id/image.png
        """
        import random
        from collections import defaultdict
        
        if not os.path.exists(self.data_dir):
            print(f"Warning: Directory {self.data_dir} does not exist. (Skipping data load for dry run)")
            return

        self.classes = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        split_mode = self.config.get('split_mode', 'hand')
        
        if split_mode == 'hand':
            train_hand = self.config.get('train_hand', 'left')
            val_hand = self.config.get('val_hand', 'right')
            target_hand = train_hand if self.is_train else val_hand
            
            for cls_name in self.classes:
                cls_dir = os.path.join(self.data_dir, cls_name)
                hand_dir = os.path.join(cls_dir, target_hand)
                if not os.path.isdir(hand_dir):
                    continue
                for img_name in os.listdir(hand_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(hand_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[cls_name]))
                        
        elif split_mode == 'ratio':
            hand_filter = self.config.get('hand_filter', 'both')
            train_ratio = self.config.get('train_ratio', 0.8)
            seed = self.config.get('seed', 42)
            
            person_dict = defaultdict(list)
            for cls_name in self.classes:
                cls_dir = os.path.join(self.data_dir, cls_name)
                subdirs = ['left', 'right'] if hand_filter == 'both' else [hand_filter]
                for subdir in subdirs:
                    hand_dir = os.path.join(cls_dir, subdir)
                    if not os.path.isdir(hand_dir):
                        continue
                    for img_name in os.listdir(hand_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            img_path = os.path.join(hand_dir, img_name)
                            person_dict[self.class_to_idx[cls_name]].append(img_path)
            
            rng = random.Random(seed)
            for label, paths in person_dict.items():
                rng.shuffle(paths)
                split_idx = int(len(paths) * train_ratio)
                
                selected_paths = paths[:split_idx] if self.is_train else paths[split_idx:]
                for p in selected_paths:
                    self.samples.append((p, label))
                    
        elif split_mode == 'person':
            self._load_person_split()
        else:
            raise ValueError(f"Không hỗ trợ split_mode: {split_mode} cho OwnDataset")
            
        current_split = self.config.get('split', 'train' if self.is_train else 'val')
        print(f"Loaded OwnDataset ({split_mode}) split='{current_split}': {len(self.samples)} samples, {len(set(lbl for _, lbl in self.samples))} identities")

    def _load_person_split(self):
        import random
        from collections import defaultdict
        
        num_train    = self.config.get('num_train_persons', 10)
        num_known    = self.config.get('num_known_persons', 5)
        num_stranger = self.config.get('num_stranger_persons', None)
        reg_ratio    = self.config.get('register_ratio', 0.5)
        cur_split    = self.config.get('split', 'train' if self.is_train else 'val')
        seed         = self.config.get('seed', 42)
        hand_filter  = self.config.get('hand_filter', 'both')
        
        person_dict = defaultdict(list)
        for cls_name in self.classes:
            pid = self.class_to_idx[cls_name]
            cls_dir = os.path.join(self.data_dir, cls_name)
            subdirs = ['left', 'right'] if hand_filter == 'both' else [hand_filter]
            for subdir in subdirs:
                hand_dir = os.path.join(cls_dir, subdir)
                if not os.path.isdir(hand_dir):
                    continue
                for img_name in os.listdir(hand_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        person_dict[pid].append(os.path.join(hand_dir, img_name))
                        
        all_persons = sorted(person_dict.keys())
        rng = random.Random(seed)
        rng.shuffle(all_persons)
        
        total = len(all_persons)
        if num_stranger is None:
            num_stranger = total - num_train - num_known
            
        used = num_train + num_known + num_stranger
        assert used <= total, (
            f"X({num_train}) + Y({num_known}) + Z({num_stranger}) = {used} "
            f"vượt quá tổng số người ({total}) trong dataset."
        )
        
        train_persons    = all_persons[:num_train]
        known_persons    = all_persons[num_train : num_train + num_known]
        stranger_persons = all_persons[num_train + num_known : num_train + num_known + num_stranger]
        
        print(f"  [Person Split] Total={total} | "
              f"Train={len(train_persons)} | "
              f"Known={len(known_persons)} | "
              f"Stranger={len(stranger_persons)} | "
              f"register_ratio={reg_ratio}")
              
        pid_to_label = {}
        for pid in sorted(train_persons) + sorted(known_persons) + sorted(stranger_persons):
            if pid not in pid_to_label:
                pid_to_label[pid] = len(pid_to_label)
                
        if cur_split == 'train':
            for pid in sorted(train_persons):
                new_label = pid_to_label[pid]
                for path in sorted(person_dict[pid]):
                    self.samples.append((path, new_label))
        elif cur_split == 'register':
            for pid in sorted(known_persons):
                new_label = pid_to_label[pid]
                paths = sorted(person_dict[pid])
                split_idx = max(1, int(len(paths) * reg_ratio))
                for path in paths[:split_idx]:
                    self.samples.append((path, new_label))
        elif cur_split == 'probe':
            for pid in sorted(known_persons):
                new_label = pid_to_label[pid]
                paths = sorted(person_dict[pid])
                split_idx = max(1, int(len(paths) * reg_ratio))
                for path in paths[split_idx:]:
                    self.samples.append((path, new_label))
        elif cur_split == 'stranger':
            for pid in sorted(stranger_persons):
                new_label = pid_to_label[pid]
                for path in sorted(person_dict[pid]):
                    self.samples.append((path, new_label))
        elif cur_split == 'val':
            for pid in sorted(known_persons) + sorted(stranger_persons):
                new_label = pid_to_label[pid]
                for path in sorted(person_dict[pid]):
                    self.samples.append((path, new_label))
        else:
            raise ValueError(f"Unknown split='{cur_split}' cho person mode.")

    def __len__(self) -> int:
        return len(self.samples) if len(self.samples) > 0 else 100 # Mock length for dry run

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if len(self.samples) == 0:
            # Return dummy data for skeleton verification
            img_size = self.config.get('image_size', [128, 128])
            return torch.randn(3, img_size[0], img_size[1]), 0

        img_path, label = self.samples[idx]
        try:
            image = self._load_image(img_path)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img_size = self.config.get('image_size', [128, 128])
            return torch.randn(self.channels, img_size[0], img_size[1]), label

    def get_labels(self):
        if not self.samples:
            return [0] * 100
        return [label for _, label in self.samples]
