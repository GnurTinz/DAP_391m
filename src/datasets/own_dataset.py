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
        
        # Define default transforms
        img_size = tuple(self.config.get('image_size', (128, 128)))
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                # Các phép biến đổi cơ bản
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def _load_data(self) -> None:
        """
        Load dataset from directory structure.
        Expects: data_dir/class_id/image.png
        """
        if not os.path.exists(self.data_dir):
            print(f"Warning: Directory {self.data_dir} does not exist. (Skipping data load for dry run)")
            return

        self.classes = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        hand_filter = self.config.get('hand_filter', 'both')
        
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
                        self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples) if len(self.samples) > 0 else 100 # Mock length for dry run

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if len(self.samples) == 0:
            # Return dummy data for skeleton verification
            img_size = self.config.get('image_size', [128, 128])
            return torch.randn(3, img_size[0], img_size[1]), 0

        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img_size = self.config.get('image_size', [128, 128])
            return torch.randn(3, img_size[0], img_size[1]), label

    def get_labels(self):
        if not self.samples:
            return [0] * 100
        return [label for _, label in self.samples]
