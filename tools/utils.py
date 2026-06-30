import os
import random
import numpy as np
import torch
import logging
from datetime import datetime
from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_dir, experiment_name):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logger = logging.getLogger(f'Logger_{experiment_name}')
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger, log_file, timestamp

def load_models(config, device, logger, checkpoint_path=None):
    logger.info("Initializing models...")
    
    model_config = config.get('model', {})
    if 'decoder' not in model_config:
        model_config['decoder'] = {}
    model_config['decoder']['image_size'] = config.get('dataset', {}).get('image_size', [128, 128])
    
    model_type = model_config.get('type', 'probabilistic')
    if model_type == 'unet':
        palm_model = UNetPalmModel(model_config).to(device)
    else:
        palm_model = ProbabilisticPalmModel(model_config).to(device)
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'palm_model_state_dict' in checkpoint:
            state_dict = checkpoint['palm_model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            if len(state_dict) == 0:
                 state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        try:
            palm_model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded weights.")
        except Exception as e:
            logger.warning(f"Could not load palm_model weights: {e}")
    else:
        logger.warning("No valid checkpoint provided. Models are initialized randomly.")
        
    palm_model.eval()
    return palm_model
