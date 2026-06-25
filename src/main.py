import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.palm_dataset import PalmPrintDataset
from src.models.palm_model import ProbabilisticPalmModel
from src.engine.trainer import Trainer

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config_path = os.path.join('config', 'default.yaml')
    if not os.path.exists(config_path):
        # Fallback to absolute if run from src
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'default.yaml')
        
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)
    
    print("Initializing Dataset...")
    train_dataset = PalmPrintDataset(
        data_dir=config['dataset']['data_dir'],
        config=config['dataset'],
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=0 # Set to 0 for mock run
    )
    
    print("Initializing Model...")
    model = ProbabilisticPalmModel(config['model'])
    
    print("Initializing Trainer...")
    trainer = Trainer(model, train_loader, config)
    
    print("Dry run completed successfully! All modules initialized.")
    # uncomment to run 1 epoch test
    # trainer.epochs = 1
    # trainer.train()

if __name__ == "__main__":
    main()
