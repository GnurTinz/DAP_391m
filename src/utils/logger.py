import os
import logging
from torch.utils.tensorboard import SummaryWriter

class BaseLogger:
    """
    Handles logging to file and TensorBoard.
    """
    def __init__(self, config: dict):
        self.log_dir = config.get('log_dir', 'logs/experiments')
        self.enable_tensorboard = config.get('enable_tensorboard', True)
        self.experiment_name = config.get('experiment_name', 'palmprint_run')
        
        self.exp_dir = os.path.join(self.log_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Setup Python Logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent adding handlers multiple times
        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler(os.path.join(self.exp_dir, 'training.log'))
            fh.setLevel(logging.INFO)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            
        # Setup TensorBoard
        self.writer = None
        if self.enable_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tb_logs'))
            self.logger.info("TensorBoard logging enabled.")
            
    def info(self, msg):
        self.logger.info(msg)
        
    def error(self, msg):
        self.logger.error(msg)
        
    def log_scalar(self, tag, value, step):
        """
        Log a scalar value to TensorBoard.
        """
        if self.writer:
            self.writer.add_scalar(tag, value, step)
            
    def close(self):
        if self.writer:
            self.writer.close()
