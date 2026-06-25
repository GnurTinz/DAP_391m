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
        self._load_data()

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
