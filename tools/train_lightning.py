import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.lightning_module import GenerativeLightningModule
from src.datasets.data_module import PalmDataModule

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # 1. KIỂM TRA RESUME ĐỂ LOAD LẠI CONFIG GỐC CỦA VERSION
    ckpt_path = cfg.training.get('resume_from_checkpoint', None)
    if ckpt_path and os.path.exists(ckpt_path):
        version_dir = os.path.dirname(os.path.dirname(ckpt_path))
        backup_config_path = os.path.join(version_dir, "config_backup.yaml")
        if os.path.exists(backup_config_path):
            print(f"🔄 Đang tải lại cấu hình gốc từ: {backup_config_path}")
            backup_cfg = OmegaConf.load(backup_config_path)
            # Phải giữ lại thông tin resume_from_checkpoint
            if 'training' not in backup_cfg:
                backup_cfg.training = {}
            backup_cfg.training.resume_from_checkpoint = ckpt_path
            cfg = backup_cfg

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
    
    # Kích hoạt setup để lấy số lượng class thật của dataset (rất quan trọng cho ArcFace)
    datamodule.setup(stage='fit')
    
    # Tính toán num_classes từ dataset
    if hasattr(datamodule.train_dataset, 'dataset') and hasattr(datamodule.train_dataset.dataset, 'samples'):
        # Trường hợp random_split (dùng Subset)
        labels = [datamodule.train_dataset.dataset.samples[i][1] for i in datamodule.train_dataset.indices]
        num_classes = max(labels) + 1 if len(labels) > 0 else 100
    elif hasattr(datamodule.train_dataset, 'samples'):
        # Trường hợp chia sẵn Train/Val
        labels = [item[1] for item in datamodule.train_dataset.samples]
        num_classes = max(labels) + 1 if len(labels) > 0 else 100
    else:
        num_classes = config_dict.get('dataset', {}).get('num_classes', 100)
        
    print(f"Tự động nhận diện số lượng classes: {num_classes}")
    if 'dataset' not in config_dict:
        config_dict['dataset'] = {}
    config_dict['dataset']['num_classes'] = num_classes

    # Initialize LightningModule
    print("Khởi tạo LightningModule...")
    model = GenerativeLightningModule(config_dict)

    # Setup Logger
    logger = False
    if cfg.logging.enable_tensorboard:
        kwargs = {
            "save_dir": cfg.logging.log_dir,
            "name": cfg.logging.experiment_name,
            "log_graph": True
        }
        # Nếu chạy tiếp (resume), trỏ thẳng logger vào đúng thư mục version cũ
        ckpt_path = cfg.training.get('resume_from_checkpoint', None)
        if ckpt_path and os.path.exists(ckpt_path):
            version_dir = os.path.dirname(os.path.dirname(ckpt_path))
            kwargs["version"] = os.path.basename(version_dir)
            
        logger = TensorBoardLogger(**kwargs)

    # Lấy log_dir thật sự của version (ví dụ logs/experiments/lightning_run/version_0)
    version_dir = logger.log_dir if logger else cfg.logging.log_dir
    os.makedirs(version_dir, exist_ok=True)
    
    # Lưu file config_backup.yaml
    with open(os.path.join(version_dir, "config_backup.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # Setup Callbacks
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(version_dir, "checkpoints"),
        filename='best',
        save_top_k=1,
        monitor='val/Total_Loss_epoch',
        mode='min'
    )
    
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(version_dir, "checkpoints"),
        filename='last',
        save_top_k=1,
        every_n_epochs=1,
        monitor='step',
        mode='max'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Cấu hình thanh tiến trình (TQDM) phù hợp cho Colab/Terminal
    refresh_rate = cfg.logging.get('progress_bar_refresh_rate', 10)
    progress_bar = TQDMProgressBar(refresh_rate=refresh_rate)

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.get('accelerator', 'auto'),
        devices=cfg.training.get('devices', 1),
        callbacks=[best_checkpoint_callback, last_checkpoint_callback, lr_monitor, progress_bar],
        logger=logger,
        log_every_n_steps=cfg.training.log_interval,
        check_val_every_n_epoch=1
    )

    # Start Training
    print("Bắt đầu huấn luyện...")
    
    ckpt_path = cfg.training.get('resume_from_checkpoint', None)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Tiếp tục huấn luyện từ checkpoint: {ckpt_path}")
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
