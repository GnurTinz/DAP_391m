import os
import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Thêm đường dẫn tới thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.train_lightning import main as train_main

class TestTrainDeterministic(unittest.TestCase):
    def test_train_deterministic_config(self):
        """
        Kiểm tra config_deterministic.yaml có thiết lập đúng:
        - Seed = 42
        - use_mlp = False
        - loss_schedules là contrastive_only (beta_kl=0, lambda_rec=0, lambda_con=1.0)
        Và mô phỏng vòng chạy train_lightning để xem có crash không.
        """
        # Đọc trực tiếp config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config_deterministic.yaml")
        self.assertTrue(os.path.exists(config_path), "File config_deterministic.yaml không tồn tại")
        
        cfg = OmegaConf.load(config_path)
        
        # 1. Kiểm tra seed
        self.assertEqual(cfg.seed, 42, "Seed phải bằng 42")
        
        # 2. Kiểm tra không dùng MLP
        self.assertFalse(cfg.model.projector.use_mlp, "Phải tắt use_mlp")

        from hydra import initialize, compose
        from hydra.core.global_hydra import GlobalHydra
        
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize(version_base=None, config_path="../config"):
            # Compose config_deterministic
            composed_cfg = compose(config_name="config_deterministic", overrides=[
                "dataset.num_train_persons=2",
                "dataset.num_known_persons=2",
                "dataset.num_stranger_persons=1",
                "training.batch_size=4",
                "training.sampler_k=2",
                "training.sampler_p=2"
            ])
            OmegaConf.set_struct(composed_cfg, False)
            composed_cfg.training.fast_dev_run = True

            # Kiểm tra loss schedule
            self.assertEqual(composed_cfg.loss_schedules.beta_kl.value, 0.0)
            self.assertEqual(composed_cfg.loss_schedules.lambda_rec.value, 0.0)
            self.assertEqual(composed_cfg.loss_schedules.lambda_con.value, 1.0)
            self.assertEqual(composed_cfg.loss_schedules.lambda_unc.value, 0.0)

            # Thực thi thử hàm train_main
            try:
                # Đổi thư mục log để không bị rác
                composed_cfg.logging.log_dir = "tests/temp_logs"
                train_main(composed_cfg)
                self.assertTrue(True, "Hàm train chạy thành công")
            except Exception as e:
                self.fail(f"Hàm train_lightning gặp lỗi: {e}")

if __name__ == "__main__":
    unittest.main()
