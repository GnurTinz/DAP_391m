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
        # Các transform mặc định cho MNIST
        img_size = tuple(self.config.get('image_size', (32, 32)))
        
        # Vì mô hình PalmPrint hiện tại nhận đầu vào 3 channels, 
        # ta có thể convert Grayscale sang RGB bằng Grayscale(num_output_channels=3)
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load MNIST dataset qua torchvision
        self.mnist_data = datasets.MNIST(
            root=self.data_dir, 
            train=self.is_train, 
            download=True, 
            transform=self.transform
        )

    def _load_data(self) -> None:
        # Bỏ qua vì torchvision.datasets.MNIST đã tự động quản lý samples
        pass

    def __len__(self) -> int:
        return len(self.mnist_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.mnist_data[idx]
        return image, label

    def get_labels(self):
        return self.mnist_data.targets.tolist()
