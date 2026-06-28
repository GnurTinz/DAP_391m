from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    """
    Abstract Base Class for PalmPrint Datasets.
    Enforces implementation of core dataset methods.
    """
    
    def __init__(self, data_dir: str, config: Dict[str, Any], is_train: bool = True):
        self.data_dir = data_dir
        self.config = config
        self.is_train = is_train
        self.samples = []
        self.image_size = tuple(self.config.get('image_size', (128, 128)))
        self.channels = self.config.get('channels', 3)
        self.transform = self._build_transforms()
        self._load_data()

    def _build_transforms(self):
        from torchvision import transforms
        transform_cfg = self.config.get('transforms', {})
        
        t_list = [transforms.Resize(self.image_size)]
        
        if self.is_train:
            if 'random_horizontal_flip' in transform_cfg:
                p = transform_cfg.get('random_horizontal_flip', 0.5)
                t_list.append(transforms.RandomHorizontalFlip(p=p))
                
            if 'random_rotation' in transform_cfg:
                degrees = transform_cfg.get('random_rotation', 15)
                t_list.append(transforms.RandomRotation(degrees))
                
            if 'random_grayscale' in transform_cfg:
                p = transform_cfg.get('random_grayscale', 0.5)
                # RandomGrayscale automatically keeps 3 channels but sets R=G=B
                t_list.append(transforms.RandomGrayscale(p=p))
                
            if 'color_jitter' in transform_cfg:
                cj = transform_cfg['color_jitter']
                t_list.append(transforms.ColorJitter(
                    brightness=cj.get('brightness', 0.1),
                    contrast=cj.get('contrast', 0.1),
                    saturation=cj.get('saturation', 0),
                    hue=cj.get('hue', 0)
                ))
                
            if 'random_affine' in transform_cfg:
                ra = transform_cfg['random_affine']
                t_list.append(transforms.RandomAffine(
                    degrees=ra.get('degrees', 0),
                    translate=tuple(ra.get('translate', (0.1, 0.1))) if 'translate' in ra else None,
                    scale=tuple(ra.get('scale', (0.9, 1.1))) if 'scale' in ra else None
                ))
                
        t_list.append(transforms.ToTensor())
        
        if self.channels == 1:
            t_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        else:
            t_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            
        return transforms.Compose(t_list)

    def _load_image(self, img_path: str) -> torch.Tensor:
        from PIL import Image
        import os
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = Image.open(img_path)
        if self.channels == 1:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
            
        if self.transform:
            img = self.transform(img)
            
        return img

    @abstractmethod
    def _load_data(self) -> None:
        """
        Load dataset metadata into self.samples.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Return a tuple of (image_tensor, label).
        Can be extended to return (image_tensor, label, mask) if needed.
        """
        pass

    def get_labels(self):
        """
        Return a list of labels for all samples, useful for samplers.
        """
        raise NotImplementedError
