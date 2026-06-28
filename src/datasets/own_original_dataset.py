import os
from typing import Any, Dict
from .own_dataset import OwnDataset

class OwnOriginalDataset(OwnDataset):
    """
    Custom Dataset loader (OwnOriginalDataset).
    Designed specifically to read from the 'data/collect' directory.
    Inherits the structure loading logic from OwnDataset.
    """
    
    def __init__(self, data_dir: str = 'data/collect', config: Dict[str, Any] = None, is_train: bool = True):
        if config is None:
            config = {}
        # Nếu config có truyền data_dir khác thì ưu tiên, nếu không thì dùng mặc định 'data/collect'
        actual_dir = data_dir if data_dir and data_dir != "" else 'data/collect'
        
        # Bắt buộc thêm Augmentation RandomGrayscale (3 channel như nhau) riêng cho tập OwnOriginal
        if is_train:
            if 'transforms' not in config:
                config['transforms'] = {}
            # Áp dụng xác suất 50% ảnh sẽ bị chuyển thành xám (R=G=B)
            config['transforms']['random_grayscale'] = 0.5
            
        super().__init__(actual_dir, config, is_train)
