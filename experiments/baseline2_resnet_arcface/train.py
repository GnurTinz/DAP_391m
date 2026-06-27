import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.datasets import DatasetFactory
from experiments.baseline2_resnet_arcface.model import ResNetArcFace

class LabelRemapDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.samples = []
        self.label_map = {}
        self.num_classes = 0
        
        # Build label map
        for i, item in enumerate(original_dataset.samples):
            orig_label = item[1]
            if orig_label not in self.label_map:
                self.label_map[orig_label] = self.num_classes
                self.num_classes += 1
            self.samples.append((i, self.label_map[orig_label]))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        orig_idx, new_label = self.samples[idx]
        img = self.dataset[orig_idx][0]
        return img, new_label
        
@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    # Load dataset
    dataset_config = config.get('dataset', {})
    data_dir = dataset_config.get('data_dir', 'data/collect')
    dataset_name = dataset_config.get('name', 'OwnOriginalDataset')
    
    print(f"\nĐang tải dữ liệu Train từ {dataset_name} ({data_dir})...")
    orig_dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=dataset_config, is_train=True)
    
    train_dataset = LabelRemapDataset(orig_dataset)
    num_classes = train_dataset.num_classes
    print(f"Tổng số Classes (Identities): {num_classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # Init model
    model = ResNetArcFace(num_classes=num_classes, feature_dim=512, pretrained=True).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    epochs = 30
    save_dir = "logs/baseline2_resnet"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "checkpoint_resnet_arcface.pth")
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    torch.save({'label_map': train_dataset.label_map, 'num_classes': num_classes}, os.path.join(save_dir, "meta.pt"))
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            features, logits = model(imgs, labels)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
            
        scheduler.step()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'label_map': train_dataset.label_map,
            'num_classes': num_classes
        }, checkpoint_path)
        
    torch.save(model.state_dict(), os.path.join(save_dir, "resnet_arcface.pth"))
    print(f"\nHoàn tất huấn luyện Baseline 2! Trọng số lưu tại {save_dir}")

if __name__ == '__main__':
    main()
