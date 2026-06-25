import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch
from src.datasets.factory import DatasetFactory
from src.datasets.sampler import PKSampler, WeightedClassSampler

class PalmDataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.dataset_cfg = config.get('dataset', {})
        self.train_cfg = config.get('training', {})
        
        self.data_dir = self.dataset_cfg.get('data_dir', 'data/MNIST')
        self.dataset_name = self.dataset_cfg.get('name', 'MNISTDataset')
        
        self.batch_size = self.train_cfg.get('batch_size', 32)
        self.num_workers = self.train_cfg.get('num_workers', 2)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            split_mode = self.dataset_cfg.get('split_mode', 'ratio')
            
            if split_mode == 'ratio':
                full_dataset = DatasetFactory.create(
                    self.dataset_name, 
                    data_dir=self.data_dir, 
                    config=self.dataset_cfg, 
                    is_train=True
                )
                
                train_ratio = self.dataset_cfg.get('train_ratio', 0.9)
                train_size = int(train_ratio * len(full_dataset))
                val_size = len(full_dataset) - train_size
                
                print("-" * 50)
                print(f"[DATASET INFO] Tên Dataset: {self.dataset_name}")
                print(f"[DATASET INFO] Thư mục: {self.data_dir}")
                print(f"[DATASET INFO] Tổng số mẫu: {len(full_dataset)}")
                
                if hasattr(full_dataset, 'classes'):
                    print(f"[DATASET INFO] Số lượng nhãn: {len(full_dataset.classes)}")
                    # print(f"[DATASET INFO] Danh sách nhãn: {full_dataset.classes}")
                elif hasattr(full_dataset, 'class_to_idx'):
                    print(f"[DATASET INFO] Số lượng nhãn: {len(full_dataset.class_to_idx)}")
                    
                print(f"[DATASET INFO] Split Mode: Ratio")
                print(f"[DATASET INFO] Tập Train: {train_size} samples")
                print(f"[DATASET INFO] Tập Val: {val_size} samples")
                print("-" * 50)
                
                # Using fixed generator for reproducible splits
                self.train_dataset, self.val_dataset = random_split(
                    full_dataset, 
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                
            elif split_mode == 'hand':
                train_hand = self.dataset_cfg.get('train_hand', 'left')
                val_hand = self.dataset_cfg.get('val_hand', 'right')
                
                train_cfg = self.dataset_cfg.copy()
                train_cfg['hand_filter'] = train_hand
                
                val_cfg = self.dataset_cfg.copy()
                val_cfg['hand_filter'] = val_hand
                
                self.train_dataset = DatasetFactory.create(
                    self.dataset_name, 
                    data_dir=self.data_dir, 
                    config=train_cfg, 
                    is_train=True
                )
                
                self.val_dataset = DatasetFactory.create(
                    self.dataset_name, 
                    data_dir=self.data_dir, 
                    config=val_cfg, 
                    is_train=False
                )
                
                print("-" * 50)
                print(f"[DATASET INFO] Tên Dataset: {self.dataset_name}")
                print(f"[DATASET INFO] Thư mục: {self.data_dir}")
                if hasattr(self.train_dataset, 'classes'):
                    print(f"[DATASET INFO] Số lượng nhãn: {len(self.train_dataset.classes)}")
                    
                print(f"[DATASET INFO] Split Mode: Hand (Train: {train_hand}, Val: {val_hand})")
                print(f"[DATASET INFO] Tập Train: {len(self.train_dataset)} samples")
                print(f"[DATASET INFO] Tập Val: {len(self.val_dataset)} samples")
                print("-" * 50)

    def train_dataloader(self):
        use_sampler = self.train_cfg.get('use_sampler', False)
        
        if use_sampler:
            sampler_type = self.train_cfg.get('sampler_type', 'pk_sampler')
            # Extract underlying targets for the sampler
            if hasattr(self.train_dataset.dataset, 'get_labels'):
                all_targets = self.train_dataset.dataset.get_labels()
            elif hasattr(self.train_dataset.dataset, 'targets'):
                all_targets = self.train_dataset.dataset.targets
            else:
                raise AttributeError("Dataset must implement 'get_labels()' or 'targets' for sampler.")
            targets = [all_targets[i] for i in self.train_dataset.indices]
            
            if sampler_type == 'pk_sampler':
                p = self.train_cfg.get('sampler_p', 8)
                k = self.train_cfg.get('sampler_k', 4)
                sampler = PKSampler(targets, p=p, k=k)
            elif sampler_type == 'weighted':
                sampler = WeightedClassSampler(targets, batch_size=self.batch_size)
                
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                drop_last=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )
