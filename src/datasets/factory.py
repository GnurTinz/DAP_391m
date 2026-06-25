from typing import Dict, Any, Optional
from .base import BaseDataset
from .palm_dataset import PalmPrintDataset
from .mnist_dataset import MNISTDataset
from .own_dataset import OwnDataset

class DatasetFactory:
    """
    Factory class to instantiate datasets based on the dataset name.
    """
    
    _datasets = {
        'PalmPrintDataset': PalmPrintDataset,
        'MNISTDataset': MNISTDataset,
        'OwnDataset': OwnDataset
    }
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class: type):
        """Register a new dataset class."""
        cls._datasets[name] = dataset_class
        
    @classmethod
    def create(cls, name: str, data_dir: str, config: Dict[str, Any], is_train: bool = True) -> BaseDataset:
        """
        Create and return a Dataset instance.
        """
        if name not in cls._datasets:
            raise ValueError(f"Dataset '{name}' is not supported. Supported datasets are: {list(cls._datasets.keys())}")
            
        dataset_class = cls._datasets[name]
        return dataset_class(data_dir=data_dir, config=config, is_train=is_train)
