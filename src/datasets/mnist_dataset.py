import torch
from torchvision import datasets, transforms
from typing import Any, Tuple, Dict
from .base import BaseDataset

class MNISTDataset(BaseDataset):
    """
    Dataset wrapper cho MNIST sử dụng kiến trúc BaseDataset.
    Tự động tải dữ liệu từ torchvision.datasets.
    """
    def __init__(self, data_dir: str, config: Dict[str, Any], is_train: bool = True):
        super().__init__(data_dir, config, is_train)
        # Load MNIST dataset qua torchvision (không áp transform ở đây, sẽ áp ở __getitem__)
        self.mnist_data = datasets.MNIST(
            root=self.data_dir, 
            train=self.is_train, 
            download=True, 
            transform=None
        )

    def _load_data(self) -> None:
        # Bỏ qua vì torchvision.datasets.MNIST đã tự động quản lý samples
        pass

    def __len__(self) -> int:
        return len(self.mnist_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.mnist_data[idx]
        if self.channels == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_labels(self):
        return self.mnist_data.targets.tolist()
