import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import DatasetFactory
from src.models.palm_model import ProbabilisticPalmModel
from src.engine.trainer import Trainer
from src.utils.logger import BaseLogger

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
    
    # Initialize Logger
    logger = BaseLogger(config.get('logging', {}))
    logger.info("Configuration loaded.")
    
    logger.info("Initializing Dataset...")
    dataset_name = config.get('dataset', {}).get('name', 'PalmPrintDataset')
    # For backward compatibility with old configs
    if dataset_name.upper() == 'POLYU':
        dataset_name = 'PalmPrintDataset'
    elif dataset_name.upper() == 'MNIST':
        dataset_name = 'MNISTDataset'
        
    try:
        train_dataset = DatasetFactory.create(
            dataset_name,
            data_dir=config['dataset']['data_dir'],
            config=config['dataset'],
            is_train=True
        )
    except ValueError as e:
        logger.error(str(e))
        return
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=0 # Set to 0 for mock run
    )
    
    logger.info("Initializing Model...")
    model = ProbabilisticPalmModel(config['model'])
    
    logger.info("Initializing Trainer...")
    trainer = Trainer(model, train_loader, config, logger=logger)
    
    logger.info("Dry run completed successfully! All modules initialized.")
    # uncomment to run 1 epoch test
    # trainer.epochs = 1
    # trainer.train()
    
    # Close logger
    logger.close()

if __name__ == "__main__":
    main()
