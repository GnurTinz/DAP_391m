import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.lightning_module import GenerativeLightningModule
from src.datasets.data_module import PalmDataModule

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print("="*50)
    print("CẤU HÌNH HUẤN LUYỆN (HYDRA OMEGACONF)")
    print(OmegaConf.to_yaml(cfg))
    print("="*50)

    # Convert OmegaConf DictConfig to native python dict since internal code expects it
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Setup seed
    seed = cfg.training.get('seed', 42)
    pl.seed_everything(seed, workers=True)

    # Initialize DataModule
    print("Khởi tạo DataModule...")
    datamodule = PalmDataModule(config_dict)

    # Initialize LightningModule
    print("Khởi tạo LightningModule...")
    model = GenerativeLightningModule(config_dict)

    # Setup Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.logging.log_dir,
        filename='best',
        save_top_k=1,
        save_last=True,
        monitor='val/Total_Loss',
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Setup Logger
    logger = False
    if cfg.logging.enable_tensorboard:
        logger = TensorBoardLogger(
            save_dir=cfg.logging.log_dir,
            name=cfg.logging.experiment_name,
            log_graph=True
        )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.get('accelerator', 'auto'),
        devices=cfg.training.get('devices', 1),
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=cfg.training.log_interval
    )

    # Start Training
    print("Bắt đầu huấn luyện...")
    
    # Lấy log_dir thật sự của version (ví dụ logs/experiments/lightning_run/version_0)
    version_dir = logger.log_dir if logger else cfg.logging.log_dir
    os.makedirs(version_dir, exist_ok=True)
    
    # Lưu file config_backup.yaml
    with open(os.path.join(version_dir, "config_backup.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
        
    # Cập nhật đường dẫn lưu của ModelCheckpoint thành bên trong version_X/checkpoints
    checkpoint_callback.dirpath = os.path.join(version_dir, "checkpoints")
    
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
