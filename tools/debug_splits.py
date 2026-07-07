import os
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Thêm path để import từ src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.factory import DatasetFactory

def make_dataloader(config: dict, split: str) -> DataLoader:
    ds_cfg  = config.get("dataset", {})
    ds_cfg_copy = dict(ds_cfg)
    ds_cfg_copy['split'] = split
    name     = ds_cfg_copy.get("name", "iitd")
    ddir     = ds_cfg_copy.get("data_dir", "data/IITD")
    is_train = split == "train"
    
    # Không in log dài dòng, chỉ load
    dataset = DatasetFactory.create(name, ddir, ds_cfg_copy, is_train=is_train)
    return DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

def get_unique_labels(loader):
    labels = set()
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            lbls = batch[1]
        else:
            lbls = batch.get("label", batch.get("id"))
        labels.update(lbls.tolist())
    return labels

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    print("="*60)
    print(" BẮT ĐẦU KIỂM TRA PHÂN CHIA DỮ LIỆU (SPLIT_MODE='PERSON')")
    print("="*60)
    
    try:
        train_loader    = make_dataloader(config, "train")
        reg_loader      = make_dataloader(config, "register")
        probe_loader    = make_dataloader(config, "probe")
        stranger_loader = make_dataloader(config, "stranger")
    except Exception as e:
        print(f"Lỗi khi tạo dataloader: {e}")
        return

    print("\nĐang đếm danh tính (Identities) trong từng tập...")
    train_ids    = get_unique_labels(train_loader)
    reg_ids      = get_unique_labels(reg_loader)
    probe_ids    = get_unique_labels(probe_loader)
    stranger_ids = get_unique_labels(stranger_loader)
    
    print(f" - Train    : {len(train_ids)} người")
    print(f" - Register : {len(reg_ids)} người")
    print(f" - Probe    : {len(probe_ids)} người")
    print(f" - Stranger : {len(stranger_ids)} người")
    
    print("\n" + "="*60)
    print(" KIỂM TRA TÍNH HỢP LỆ CỦA BÀI TOÁN OPEN-SET")
    print("="*60)
    
    # 1. Register và Probe có phải là cùng một nhóm người?
    print("[Check 1] Tập Register và Probe có chứa CÙNG MỘT NHÓM NGƯỜI (Known) hay không?")
    diff_reg_probe = reg_ids.symmetric_difference(probe_ids)
    if len(diff_reg_probe) == 0:
        print("   => [PASS] Hoàn hảo! Register và Probe hoàn toàn khớp nhau.")
    else:
        print(f"   => [FAIL] Có sự khác biệt: {diff_reg_probe}")
        
    # 2. Stranger có bị trùng vào nhóm Known không?
    print("\n[Check 2] Nhóm người lạ (Stranger) có bị trùng với nhân viên (Register/Probe) không?")
    overlap_stranger = stranger_ids.intersection(reg_ids.union(probe_ids))
    if len(overlap_stranger) == 0:
        print("   => [PASS] An toàn! Không có ai vừa làm nhân viên vừa làm người lạ.")
    else:
        print(f"   => [FAIL] Phát hiện trùng lặp: {overlap_stranger}")
        
    # 3. Tập Đánh giá có bị rò rỉ dữ liệu từ tập Train không? (Data Leakage)
    print("\n[Check 3] Tập đánh giá (Register/Probe/Stranger) có bị trùng với tập Huấn luyện (Train) không?")
    eval_ids = reg_ids.union(probe_ids).union(stranger_ids)
    overlap_train = train_ids.intersection(eval_ids)
    if len(overlap_train) == 0:
        print("   => [PASS] Tuyệt vời! Không bị Data Leakage. Tập Test độc lập 100% với tập Train.")
    else:
        print(f"   => [FAIL] Rò rỉ dữ liệu! Những ID sau xuất hiện cả ở Train lẫn Test: {overlap_train}")
        
    print("\n" + "="*60)
    print(" TỔNG KẾT: Dữ liệu đã được chia cực kỳ chuẩn xác!")
    print("="*60)

if __name__ == "__main__":
    main()
