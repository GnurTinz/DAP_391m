from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """
    Abstract Base Class for all models.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass.
        """
        pass
