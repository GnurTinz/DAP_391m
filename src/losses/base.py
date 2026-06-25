from abc import ABC, abstractmethod
import torch.nn as nn

class BaseLoss(nn.Module, ABC):
    """
    Abstract Base Class for custom losses.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
