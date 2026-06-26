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
        super().__init__(actual_dir, config, is_train)
