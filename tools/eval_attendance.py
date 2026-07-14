"""
eval_attendance.py
==================
Chương trình đánh giá toàn diện khả năng điểm danh (Attendance Evaluation).
Hỗ trợ nhiều chế độ (Mode):
    - Mode 0: Baseline (1-NN, không tối ưu r)
    - Mode 1: Optimize r trong Projected Space (real / spherical / decoder_loop)
    - Mode 2: Optimize r trong Latent Space (real / spherical / decoder_loop)

Thang đo chính: EER (Equal Error Rate), Rank-1 Accuracy
Visualization  : t-SNE (Projected Space và Mu Space)

Mỗi bước được lưu cache (.pt) để tái sử dụng (bỏ qua extract lại khi đã có).

Cách chạy:
    python tests/eval_attendance.py \\
        checkpoint="logs/Unet_Palmnet/version_2/checkpoints/last.ckpt" \\
        dataset=iitd_hand \\
        +eval.mode=0 \\
        +eval.output_dir=""

    # Hoặc Mode 1 với hard negative:
    python tests/eval_attendance.py \\
        checkpoint="..." dataset=iitd_hand \\
        +eval.mode=1 +eval.neg_strategy=real

    # Hoặc Mode 2 với decoder loop:
    python tests/eval_attendance.py \\
        checkpoint="..." dataset=iitd_hand \\
        +eval.mode=2 +eval.neg_strategy=decoder_loop
"""

import os
import sys
import re
import json
import yaml
import math
import time
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, auc

from src.models import UNetPalmModel, ProbabilisticPalmModel
from src.datasets.factory import DatasetFactory
from src.engine.represent import (
    optimize_r_from_latent,
    optimize_r_in_projected_space,
)

# ==============================================================================
# CONFIG DEFAULTS
# ==============================================================================
DEFAULT_EVAL_CFG = {
    "mode": 0,                   # 0=Baseline, 1=Optimize Proj, 2=Optimize Latent
    "neg_strategy": "real",      # real | spherical | decoder_loop
    "num_samples_opt": 512,      # số mẫu positive/negative khi tối ưu r
    "max_steps_opt": 200,        # bước tối ưu
    "lr_opt": 0.01,              # learning rate tối ưu
    "decoder_T_min": 2.0,        # nhiệt độ decoder_loop
    "decoder_T_max": 5.0,
    "gallery_split": "val",      # split dùng để build gallery
    "probe_split": "test",       # split dùng để probe
    "output_dir": "",            # tự động nếu để trống
    "force_reextract": False,    # bỏ qua cache và extract lại
    "tsne_top_k": 20,            # số identity hiển thị trong t-SNE
    "tsne_perplexity": 30,
    "batch_size": 32,
}

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
def make_logger(log_dir: str, run_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_name}_{ts}.log")

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        
        # Sửa lỗi UnicodeEncodeError trên console Windows
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
            
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
    return logger, log_path

# ==============================================================================
# MODEL LOADING (tái sử dụng pattern từ pca_latent.py)
# ==============================================================================
def load_model(config: dict, device: torch.device, logger: logging.Logger):
    """Load model từ checkpoint, ưu tiên đọc config_backup.yaml."""
    checkpoint_path = config.get("checkpoint", "")
    version_dir = ""

    if checkpoint_path:
        m = re.search(r"(.*[\\/]version_\d+)", checkpoint_path.replace("\\", "/"))
        if m:
            version_dir = m.group(1)

    # 1. Đọc config_backup nếu có
    backup_path = os.path.join(version_dir, "config_backup.yaml") if version_dir else ""
    if backup_path and os.path.exists(backup_path):
        logger.info(f"Reading model config from {backup_path}")
        with open(backup_path, "r", encoding="utf-8") as f:
            backup_cfg = yaml.safe_load(f)
        if "model" in backup_cfg:
            config["model"] = backup_cfg["model"]
        if "dataset" in backup_cfg and "image_size" in backup_cfg["dataset"]:
            config.setdefault("dataset", {})["image_size"] = backup_cfg["dataset"]["image_size"]
    else:
        logger.warning(f"config_backup.yaml not found at {version_dir}. Using current config.")

    # 2. Build model
    model_cfg = config.get("model", {})
    model_cfg.setdefault("decoder", {})["image_size"] = config.get("dataset", {}).get("image_size", [128, 128])
    model_type = model_cfg.get("type", "unet")

    if model_type == "unet":
        model = UNetPalmModel(model_cfg).to(device)
    else:
        model = ProbabilisticPalmModel(model_cfg).to(device)

    # 3. Load weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning("No checkpoint found. Using random weights.")

    # 4. Freeze BN
    model.eval()
    bn_frozen = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()
            m.requires_grad_(False)
            bn_frozen += 1
    if bn_frozen:
        logger.info(f"Frozen {bn_frozen} BatchNorm layer(s) into eval mode.")

    return model, version_dir

# ==============================================================================
# DATALOADER
# ==============================================================================
def make_dataloader(config: dict, split: str, batch_size: int = 32) -> DataLoader:
    """
    Tạo DataLoader cho một split cụ thể.

    split có thể là:
      - 'train'    : tập huấn luyện
      - 'test'/'val': tập kiểm tra (closed-set)
      - 'register' : gallery (open-set, person mode)
      - 'probe'    : known probe (open-set, person mode)
      - 'stranger' : người lạ (open-set, person mode)
    """
    ds_cfg  = config.get("dataset", {})
    # Sao chép và inject 'split' vào config để dataset class biết cần load phần nào
    ds_cfg_copy = dict(ds_cfg)
    ds_cfg_copy['split'] = split

    name     = ds_cfg_copy.get("name", "iitd")
    ddir     = ds_cfg_copy.get("data_dir", "data/IITD")
    is_train = split == "train"

    print(f"========================Chế độ lấy DataLoader: split='{split}', is_train={is_train}")
    dataset = DatasetFactory.create(name, ddir, ds_cfg_copy, is_train=is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True, drop_last=False)

# ==============================================================================
# FEATURE EXTRACTION (với caching)
# ==============================================================================
@torch.no_grad()
def extract_features(model, dataloader: DataLoader, device: torch.device,
                     cache_path: str = None, force: bool = False,
                     logger: logging.Logger = None):
    """
    Trích xuất mu, proj, logvar, labels từ dataloader.
    Đồng thời lưu một ảnh đại diện per-identity (ref_images) để dùng
    cho decoder_loop negative sampling (cần x_ref làm skip-connection).
    Nếu cache_path tồn tại và force=False, load từ cache.
    """
    if cache_path and os.path.exists(cache_path) and not force:
        if logger:
            logger.info(f"Loading feature cache: {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        return data["mu"], data["proj"], data["logvar"], data["labels"], data.get("ref_images", {})

    all_mu, all_proj, all_logvar, all_labels = [], [], [], []
    # Lưu một ảnh đại diện per-identity (CPU, tiết kiệm bộ nhớ)
    ref_images: dict = {}    # label (int) -> Tensor (1, C, H, W)

    for batch in tqdm(dataloader, desc="Extracting features"):
        if isinstance(batch, (tuple, list)):
            imgs, labels = batch[0], batch[1]
        else:
            imgs   = batch.get("image", batch.get("img"))
            labels = batch.get("label", batch.get("id"))

        imgs_dev = imgs.to(device)
        out = model(imgs_dev, decode=False)

        all_mu.append(out["mu"].cpu())
        all_proj.append(out["proj"].cpu())
        all_logvar.append(out["logvar"].cpu())

        lbl_list = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)
        if isinstance(labels, torch.Tensor):
            all_labels.append(labels.cpu())
        else:
            all_labels.append(torch.tensor(lbl_list))

        # Lưu ảnh đầu tiên thấy của mỗi identity
        for j, lbl in enumerate(lbl_list):
            lbl = int(lbl)
            if lbl not in ref_images:
                ref_images[lbl] = imgs[j:j+1].cpu()  # (1, C, H, W) trên CPU

    mu     = torch.cat(all_mu,     dim=0)
    proj   = torch.cat(all_proj,   dim=0)
    logvar = torch.cat(all_logvar, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"mu": mu, "proj": proj, "logvar": logvar,
                    "labels": labels, "ref_images": ref_images}, cache_path)
        if logger:
            logger.info(f"Saved feature cache (+ {len(ref_images)} ref images): {cache_path}")

    return mu, proj, logvar, labels, ref_images

# ==============================================================================
# GALLERY BUILDING
# ==============================================================================
def build_gallery(model, mu, proj, logvar, labels, eval_cfg: dict,
                  device: torch.device, cache_path: str = None,
                  force: bool = False, logger: logging.Logger = None,
                  ref_images: dict = None):
    """
    Xây dựng gallery (dict: label -> normalized_vectors).
    Hỗ trợ Mode 0 (1-NN proj), Mode 1 (optimize proj space), Mode 2 (optimize latent space).
    ref_images: dict {label(int) -> Tensor(1,C,H,W)} — dùng khi neg_strategy='decoder_loop'.
    Cache gallery để tái sử dụng.
    """
    mode = eval_cfg.get("mode", 0)
    neg_strategy = eval_cfg.get("neg_strategy", "real")
    if ref_images is None:
        ref_images = {}

    if cache_path and os.path.exists(cache_path) and not force:
        cached_gallery = torch.load(cache_path, map_location="cpu")
        if len(cached_gallery) > 0:
            first_val = next(iter(cached_gallery.values()))
            if isinstance(first_val, dict) and "feat" in first_val:
                logger.info(f"Loading gallery cache: {cache_path}")
                return cached_gallery
            else:
                if logger:
                    logger.warning("Old gallery cache format detected! Ignoring cache and rebuilding.")

    unique_labels = labels.unique().tolist()
    gallery = {}

    all_labels_list = labels.tolist()

    logger.info(f"Building gallery: Mode={mode}, neg_strategy='{neg_strategy}', {len(unique_labels)} identities")

    # Chuẩn bị mu_others và logvar_others (tất cả sample — sẽ filter per-identity bên trong)
    mu_all      = mu.to(device)
    logvar_all  = logvar.to(device)
    proj_all    = proj.to(device)
    labels_dev  = labels.to(device)

    config_repr = {
        "represent": {
            "neg_strategy":    neg_strategy,
            "pos_temperature": 0.5,
            "neg_temperature": 1.0,
            "decoder_T_min":   eval_cfg.get("decoder_T_min", 2.0),
            "decoder_T_max":   eval_cfg.get("decoder_T_max", 5.0),
            "x_ref":           None,  # sẽ override per-sample nếu dùng decoder_loop
        }
    }

    for lbl in tqdm(unique_labels, desc="Registering identities"):
        idx_list  = (labels == lbl).nonzero(as_tuple=True)[0]
        other_idx = (labels != lbl).nonzero(as_tuple=True)[0]

        mu_group     = mu_all[idx_list]
        logvar_group = logvar_all[idx_list]
        proj_group   = proj_all[idx_list]
        mu_others    = mu_all[other_idx]
        logvar_others = logvar_all[other_idx]

        # Cập nhật x_ref per-identity cho decoder_loop
        if neg_strategy == "decoder_loop":
            lbl_int = int(lbl)
            if lbl_int in ref_images:
                config_repr["represent"]["x_ref"] = ref_images[lbl_int].to(device)
            else:
                # Fallback: dùng mu trực tiếp decode (không skip connection)
                config_repr["represent"]["x_ref"] = None

        if mode == 0:
            # Baseline: 1-NN dùng proj trực tiếp
            gallery_r = F.normalize(proj_group, p=2, dim=1)

        elif mode == 1:
            # Optimize r trong Projected Space (per-image)
            gallery_r_list = []
            for i in range(mu_group.size(0)):
                # Cập nhật x_ref per-image nếu dùng decoder_loop
                if neg_strategy == "decoder_loop" and lbl_int in ref_images:
                    config_repr["represent"]["x_ref"] = ref_images[lbl_int].to(device)
                r_i = optimize_r_in_projected_space(
                    mu_group[i:i+1], logvar_group[i:i+1],
                    mu_others, logvar_others,
                    model, device,
                    config=config_repr,
                    num_samples=eval_cfg.get("num_samples_opt", 512),
                    max_steps=eval_cfg.get("max_steps_opt", 200),
                    lr=eval_cfg.get("lr_opt", 0.01),
                    loss_type="bce",
                    verbose=False,
                )
                gallery_r_list.append(r_i.squeeze(0))
            gallery_r = torch.stack(gallery_r_list, dim=0)  # (N, proj_dim)

        elif mode == 2 or mode == 4:
            # Optimize r trong Latent Space (per-image)
            gallery_r_list = []
            for i in range(mu_group.size(0)):
                mu_c     = mu_group[i:i+1]
                logvar_c = logvar_group[i:i+1]
                # Cập nhật x_ref per-image nếu dùng decoder_loop
                if neg_strategy == "decoder_loop" and int(lbl) in ref_images:
                    config_repr["represent"]["x_ref"] = ref_images[int(lbl)].to(device)
                r_latent, _, _ = optimize_r_from_latent(
                    mu_c, logvar_c, mu_others, logvar_others,
                    model, device,
                    config=config_repr,
                    num_samples=eval_cfg.get("num_samples_opt", 512),
                    max_steps=eval_cfg.get("max_steps_opt", 200),
                    lr=eval_cfg.get("lr_opt", 0.01),
                    verbose=False,
                )
                if mode == 2:
                    with torch.no_grad():
                        proj_r = model.projector(mu_c + r_latent)
                        proj_r = F.normalize(proj_r, p=2, dim=1)
                    gallery_r_list.append(proj_r.squeeze(0))
                else: # mode == 4 (Đánh giá trực tiếp trên mu)
                    mu_r = F.normalize(mu_c + r_latent, p=2, dim=1)
                    gallery_r_list.append(mu_r.squeeze(0))
            gallery_r = torch.stack(gallery_r_list, dim=0)  # (N, dim)

        elif mode == 3:
            # Baseline: 1-NN dùng mu trực tiếp (Latent Space)
            gallery_r = F.normalize(mu_group, p=2, dim=1)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 0, 1, 2, 3, or 4.")

        gallery[int(lbl)] = {
            "feat": gallery_r.cpu(),
            "mu": mu_group.cpu(),
            "logvar": logvar_group.cpu()
        }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(gallery, cache_path)
        logger.info(f"Saved gallery cache: {cache_path}")

    return gallery

# ==============================================================================
# METRICS: EER + Rank-1 ACC
# ==============================================================================
def calculate_eer(genuine_scores: list, impostor_scores: list):
    """Tính EER và threshold tương ứng."""
    if not genuine_scores or not impostor_scores:
        return 0.0, 0.0
    from sklearn.metrics import roc_curve
    y_true  = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    y_score = genuine_scores + impostor_scores
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer  = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    thresh = thresholds[eer_idx]
    return float(eer * 100), float(thresh)


def compute_pairwise_kl(mu_p, logvar_p, mu_g, logvar_g):
    """
    Tính Pairwise KL Divergence giữa Probe và Gallery.
    """
    var_p = torch.exp(logvar_p)
    var_g = torch.exp(logvar_g)
    D = mu_p.size(1)
    
    inv_var_g = 1.0 / var_g
    term1 = torch.mm(var_p, inv_var_g.t())
    t2_a = torch.mm(mu_p**2, inv_var_g.t())
    t2_b = -2.0 * torch.mm(mu_p, (mu_g * inv_var_g).t())
    t2_c = (mu_g**2 * inv_var_g).sum(dim=1).unsqueeze(0)
    term2 = t2_a + t2_b + t2_c
    
    term3 = logvar_g.sum(dim=1).unsqueeze(0) - logvar_p.sum(dim=1).unsqueeze(1)
    
    kl = 0.5 * (term1 + term2 - D + term3)
    return kl

def evaluate_gallery(gallery: dict, probe_proj: torch.Tensor,
                     probe_labels: torch.Tensor, device: torch.device):
    """
    Đánh giá gallery bằng cosine similarity.
    Trả về: rank1_acc (%), EER (%), genuine_scores, impostor_scores, sim_matrix.
    """
    # Flatten gallery -> (Total_G, proj_dim) + gallery_labels
    gallery_tensors_list, flat_gallery_labels = [], []
    gallery_labels_list = sorted(gallery.keys())

    for lbl in gallery_labels_list:
        t = gallery[lbl]["feat"].to(device)
        gallery_tensors_list.append(t)
        flat_gallery_labels.extend([lbl] * t.size(0))

    gallery_tensors = torch.cat(gallery_tensors_list, dim=0)  # (Total_G, proj_dim)
    flat_labels_t   = torch.tensor(flat_gallery_labels, device=device)

    probe_proj_dev  = F.normalize(probe_proj.to(device), p=2, dim=1)
    probe_labels_dev = probe_labels.to(device)

    # Raw cosine similarity (N_probe, Total_G)
    raw_sim = torch.mm(probe_proj_dev, gallery_tensors.t())

    # Aggregate per identity: max similarity (1-NN)
    gid_tensor  = torch.tensor(gallery_labels_list, device=device)
    sim_per_id_list = []
    for gid in gallery_labels_list:
        cols = (flat_labels_t == gid).nonzero(as_tuple=True)[0]
        sim_per_id_list.append(raw_sim[:, cols].max(dim=1)[0])
    sim_matrix = torch.stack(sim_per_id_list, dim=1)  # (N_probe, N_ids)

    # Match matrix
    match_matrix = (probe_labels_dev.unsqueeze(1) == gid_tensor.unsqueeze(0))

    # Rank-1
    best_idx      = sim_matrix.argmax(dim=1)
    correct_rank1 = match_matrix[torch.arange(len(probe_labels_dev)), best_idx].sum().item()
    rank1_acc     = correct_rank1 / len(probe_labels_dev) * 100.0

    # Genuine / Impostor scores
    genuine_scores  = sim_matrix[match_matrix].cpu().tolist()
    impostor_scores = sim_matrix[~match_matrix].cpu().tolist()

    eer, eer_thresh = calculate_eer(genuine_scores, impostor_scores)

    return {
        "rank1_acc":       rank1_acc,
        "eer":             eer,
        "eer_threshold":   eer_thresh,
        "genuine_scores":  genuine_scores,
        "impostor_scores": impostor_scores,
        "mean_genuine":    float(np.mean(genuine_scores)),
        "mean_impostor":   float(np.mean(impostor_scores)),
        "n_probe":         len(probe_labels_dev),
        "n_gallery_ids":   len(gallery_labels_list),
    }


def evaluate_open_set(gallery: dict,
                      known_proj: torch.Tensor, known_labels: torch.Tensor,
                      stranger_proj: torch.Tensor, stranger_labels: torch.Tensor,
                      device: torch.device,
                      eer_thresh: float = None,
                      known_mu: torch.Tensor = None,
                      known_logvar: torch.Tensor = None,
                      stranger_mu: torch.Tensor = None,
                      stranger_logvar: torch.Tensor = None):
    """
    Đánh giá Open-Set Recognition.

    Args:
        gallery       : dict label -> Tensor(N_reg, proj_dim), chỉ chứa Y người known
        known_proj    : probe features của Y người known (Tensor N_known, proj_dim)
        known_labels  : labels tương ứng (0..Y-1), khớp với gallery keys
        stranger_proj : probe features của Z người lạ (Tensor N_stranger, proj_dim)
        stranger_labels: labels người lạ (chỉ dùng để ghi log)
        eer_thresh    : ngưỡng sđ (từ closed-set EER); None → tự tính từ genuine vs stranger

    Returns dict:
        open_set_rank1  : Rank-1 accuracy trên Y known probe (%)
        far             : False Acceptance Rate của Z stranger (%)
        frr             : False Rejection Rate của Y known probe (%)
        eer             : EER giữa genuine scores và stranger max-sim (%)
        eer_threshold   : ngưỡng sđ được dùng
        genuine_scores  : max sim đúng person của mỗi known probe
        stranger_max_sims: max sim tới bất kỳ gallery entry của mỗi stranger probe
        n_known_probe   : số lượng known probe samples
        n_stranger      : số lượng stranger samples
        n_gallery_ids   : số identity trong gallery
    """
    # ── Flat gallery ──────────────────────────────────────────────────────
    gallery_tensors_list, flat_gallery_labels = [], []
    gallery_mu_list, gallery_logvar_list = [], []
    gallery_labels_list = sorted(gallery.keys())
    for lbl in gallery_labels_list:
        gdata = gallery[lbl]
        t = gdata["feat"].to(device)
        m = gdata["mu"].to(device)
        lv = gdata["logvar"].to(device)
        gallery_tensors_list.append(t)
        gallery_mu_list.append(m)
        gallery_logvar_list.append(lv)
        flat_gallery_labels.extend([lbl] * t.size(0))
    gallery_tensors  = torch.cat(gallery_tensors_list, dim=0)   # (Total_G, proj_dim)
    gallery_mu       = torch.cat(gallery_mu_list, dim=0)
    gallery_logvar   = torch.cat(gallery_logvar_list, dim=0)
    flat_labels_t    = torch.tensor(flat_gallery_labels, device=device)
    gid_tensor       = torch.tensor(gallery_labels_list, device=device)

    # ── Known probe similarity matrix ─────────────────────────────────────
    known_proj_dev  = F.normalize(known_proj.to(device), p=2, dim=1)
    known_labels_dev = known_labels.to(device)
    raw_known_sim   = torch.mm(known_proj_dev, gallery_tensors.t())  # (N_known, Total_G)

    # ── Áp dụng màng lọc KL Divergence (nếu có mu, logvar) ────────────────
    kl_thresh = None
    kl_eer = 0.0
    if known_mu is not None and known_logvar is not None:
        known_mu_dev = known_mu.to(device)
        known_lv_dev = known_logvar.to(device)
        raw_known_kl = compute_pairwise_kl(known_mu_dev, known_lv_dev, gallery_mu, gallery_logvar)
        
        # Tìm ngưỡng KL từ tập Known (Genuine vs Impostor)
        match_matrix_kl = (known_labels_dev.unsqueeze(1) == flat_labels_t.unsqueeze(0))
        genuine_kl = raw_known_kl[match_matrix_kl].cpu().tolist()
        impostor_kl = raw_known_kl[~match_matrix_kl].cpu().tolist()
        
        # calculate_eer cần score lớn là Genuine. Mà KL nhỏ là Genuine, nên đảo dấu
        kl_eer, neg_kl_thresh = calculate_eer([-x for x in genuine_kl], [-x for x in impostor_kl])
        kl_thresh = -neg_kl_thresh
        
        # KL Gating: Nếu khoảng cách KL > ngưỡng -> Cosine = -1
        raw_known_sim[raw_known_kl > kl_thresh] = -1.0

    # Aggregate per identity (max-sim 1-NN) → (N_known, N_ids)
    sim_per_id_list = []
    for gid in gallery_labels_list:
        cols = (flat_labels_t == gid).nonzero(as_tuple=True)[0]
        sim_per_id_list.append(raw_known_sim[:, cols].max(dim=1)[0])
    sim_matrix_known = torch.stack(sim_per_id_list, dim=1)   # (N_known, N_ids)

    match_matrix = (known_labels_dev.unsqueeze(1) == gid_tensor.unsqueeze(0))

    # Rank-1 accuracy (closed-set, known only)
    best_idx       = sim_matrix_known.argmax(dim=1)
    correct        = match_matrix[torch.arange(len(known_labels_dev)), best_idx].sum().item()
    open_set_rank1 = correct / len(known_labels_dev) * 100.0

    # Genuine scores: max-sim đúng identity của mỗi known probe
    genuine_scores = sim_matrix_known[match_matrix].cpu().tolist()

    # ── Stranger probe → max-sim tới bất kỳ gallery entry ─────────────────
    stranger_proj_dev = F.normalize(stranger_proj.to(device), p=2, dim=1)
    raw_str_sim       = torch.mm(stranger_proj_dev, gallery_tensors.t())  # (N_str, Total_G)

    if kl_thresh is not None and stranger_mu is not None and stranger_logvar is not None:
        str_mu_dev = stranger_mu.to(device)
        str_lv_dev = stranger_logvar.to(device)
        raw_str_kl = compute_pairwise_kl(str_mu_dev, str_lv_dev, gallery_mu, gallery_logvar)
        raw_str_sim[raw_str_kl > kl_thresh] = -1.0

    stranger_max_sims = raw_str_sim.max(dim=1)[0].cpu().tolist()

    # ── EER (genuine vs stranger) ─────────────────────────────────────────
    eer_auto, eer_thresh_auto = calculate_eer(genuine_scores, stranger_max_sims)
    if eer_thresh is None:
        eer_thresh = eer_thresh_auto
        eer_val    = eer_auto
    else:
        # Tính lại EER tại ngưỡng của closed-set
        eer_val    = eer_auto   # vẫn dùng EER tính từ genuine vs stranger

    # ── STAGE 1: Uncertainty Rejection (nếu có logvar) ────────────────────
    stage1_far = 0.0
    stage1_frr = 0.0
    unc_eer = 0.0
    unc_thresh = 0.0
    known_unc_list = []
    str_unc_list = []
    known_sigma_list = []
    str_sigma_list = []

    if known_logvar is not None and stranger_logvar is not None:
        # Uncertainty = mean(exp(logvar))
        known_unc = torch.exp(known_logvar).mean(dim=1).cpu().tolist()
        str_unc   = torch.exp(stranger_logvar).mean(dim=1).cpu().tolist()
        known_unc_list = known_unc
        str_unc_list   = str_unc

        known_sigma = torch.exp(0.5 * known_logvar).mean(dim=1).cpu().tolist()
        str_sigma   = torch.exp(0.5 * stranger_logvar).mean(dim=1).cpu().tolist()
        known_sigma_list = known_sigma
        str_sigma_list   = str_sigma

        # Tìm ngưỡng uncertainty (bằng cách xem known là genuine, str là impostor)
        # Đổi dấu vì calculate_eer kỳ vọng genuine CAO HƠN impostor.
        unc_eer, neg_unc_thresh = calculate_eer([-u for u in known_unc], [-u for u in str_unc])
        unc_thresh = -neg_unc_thresh

        # Lọc Stage 1 (những mẫu có unc <= unc_thresh mới được đi tiếp vào Stage 2)
        passed_str_sims = [s for s, u in zip(stranger_max_sims, str_unc) if u <= unc_thresh]
        passed_gen_sims = [s for s, u in zip(genuine_scores, known_unc) if u <= unc_thresh]

        # Tính toán FAR/FRR kết hợp cả 2 Stage
        far_count = sum(s >= eer_thresh for s in passed_str_sims)
        far = (far_count / len(stranger_max_sims)) * 100.0 if len(stranger_max_sims) > 0 else 0.0

        rejected_stage1 = sum(u > unc_thresh for u in known_unc)
        rejected_stage2 = sum(s < eer_thresh for s in passed_gen_sims)
        frr = ((rejected_stage1 + rejected_stage2) / len(genuine_scores)) * 100.0 if len(genuine_scores) > 0 else 0.0

        stage1_far = (sum(u <= unc_thresh for u in str_unc) / len(str_unc)) * 100.0 if len(str_unc) > 0 else 0.0
        stage1_frr = (rejected_stage1 / len(known_unc)) * 100.0 if len(known_unc) > 0 else 0.0
    else:
        # ── FAR: tỷ lệ stranger vượt ngưỡng (bị nhận nhầm) ────────────────────
        far = sum(s >= eer_thresh for s in stranger_max_sims) / len(stranger_max_sims) * 100.0

        # ── FRR: tỷ lệ known probe bị reject (genuine sim < ngưỡng) ─────────────
        frr = sum(s < eer_thresh  for s in genuine_scores)   / len(genuine_scores)   * 100.0

    # ── AUROC known-vs-unknown ────────────────────────────────────────────
    y_true_auroc = [1] * len(genuine_scores) + [0] * len(stranger_max_sims)
    y_score_auroc = genuine_scores + stranger_max_sims
    if len(np.unique(y_true_auroc)) > 1:
        auroc_known_unknown = roc_auc_score(y_true_auroc, y_score_auroc) * 100.0
    else:
        auroc_known_unknown = 0.0

    # ── DIR/TPIR @ FPIR & OSCR & Risk-Coverage (Definition B) ──────────────
    dir_at_fpir_1 = 0.0
    dir_at_fpir_01 = 0.0
    oscr_area = 0.0
    fpr_list, ccr_list = [], []
    coverage_list, risk_list = [], []

    if len(stranger_max_sims) > 0 and len(genuine_scores) > 0:
        # 1. DIR @ FPIR
        sorted_stranger_sims = sorted(stranger_max_sims, reverse=True)
        n_str = len(sorted_stranger_sims)
        
        idx_1 = max(0, int(np.ceil(0.01 * n_str)) - 1)
        thresh_1 = sorted_stranger_sims[idx_1]
        dir_at_fpir_1 = sum(s >= thresh_1 for s in genuine_scores) / len(genuine_scores) * 100.0
        
        idx_01 = max(0, int(np.ceil(0.001 * n_str)) - 1)
        thresh_01 = sorted_stranger_sims[idx_01]
        dir_at_fpir_01 = sum(s >= thresh_01 for s in genuine_scores) / len(genuine_scores) * 100.0

        # 2. OSCR & Risk-Coverage
        known_max_sims = sim_matrix_known.max(dim=1)[0].cpu().numpy()
        is_correct_rank1 = match_matrix[torch.arange(len(known_labels_dev)), best_idx].cpu().numpy()
        stranger_arr = np.array(stranger_max_sims)
        
        all_scores = np.unique(np.concatenate([known_max_sims, stranger_arr]))
        all_scores = np.sort(all_scores)[::-1]  # descending
        
        n_known = len(known_max_sims)
        n_total = n_known + n_str
        
        for tau in all_scores:
            ccr = np.sum((known_max_sims >= tau) & is_correct_rank1) / n_known
            fpr = np.sum(stranger_arr >= tau) / n_str
            ccr_list.append(float(ccr))
            fpr_list.append(float(fpr))
            
            accepted_knowns = np.sum(known_max_sims >= tau)
            accepted_strangers = np.sum(stranger_arr >= tau)
            total_acc = accepted_knowns + accepted_strangers
            
            coverage = total_acc / n_total
            misclassified_knowns = np.sum((known_max_sims >= tau) & ~is_correct_rank1)
            
            risk = (misclassified_knowns + accepted_strangers) / total_acc if total_acc > 0 else 0.0
            
            coverage_list.append(float(coverage))
            risk_list.append(float(risk))
            
        if len(fpr_list) > 1:
            oscr_area = auc(fpr_list, ccr_list) * 100.0

    return {
        "open_set_rank1":   open_set_rank1,
        "far":              far,
        "frr":              frr,
        "auroc":            auroc_known_unknown,
        "dir_at_fpir_1%":   dir_at_fpir_1,
        "dir_at_fpir_0.1%": dir_at_fpir_01,
        "oscr_area":        oscr_area,
        "oscr_fpr_list":    fpr_list,
        "oscr_ccr_list":    ccr_list,
        "risk_list":        risk_list,
        "coverage_list":    coverage_list,
        "eer":              eer_val,
        "eer_threshold":    eer_thresh,
        "unc_eer":          unc_eer,
        "unc_threshold":    unc_thresh,
        "kl_eer":           kl_eer,
        "kl_threshold":     kl_thresh if kl_thresh else 0.0,
        "stage1_far":       stage1_far,
        "stage1_frr":       stage1_frr,
        "genuine_scores":   genuine_scores,
        "stranger_max_sims": stranger_max_sims,
        "known_unc":        known_unc_list,
        "stranger_unc":     str_unc_list,
        "known_sigma":      known_sigma_list,
        "stranger_sigma":   str_sigma_list,
        "mean_genuine":     float(np.mean(genuine_scores)),
        "mean_stranger":    float(np.mean(stranger_max_sims)),
        "n_known_probe":    len(known_labels_dev),
        "n_stranger":       len(stranger_proj),
        "n_gallery_ids":    len(gallery_labels_list),
    }

# ==============================================================================
# VISUALIZATION: t-SNE
# ==============================================================================
def _make_palette(n: int):
    base = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#66c2a5","#fc8d62","#8da0cb","#e78ac3",
        "#a6d854","#ffd92f","#e5c494","#b3b3b3",
        "#1b9e77","#d95f02",
    ]
    return base[:n]

def plot_tsne(feats: np.ndarray, labels: np.ndarray,
              title: str, save_path: str,
              top_k: int = 20, perplexity: int = 30, n_iter: int = 1000,
              logger: logging.Logger = None):
    """Vẽ t-SNE 2D scatter, tô màu top-K identities."""
    if logger:
        logger.info(f"Running t-SNE ({feats.shape[0]} points, perplexity={perplexity})...")

    # Pre-reduce nếu chiều lớn
    if feats.shape[1] > 50:
        pre = PCA(n_components=50).fit_transform(feats)
    else:
        pre = feats

    emb = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
               random_state=42, init="pca").fit_transform(pre)

    unique_ids = np.unique(labels)
    top_ids    = unique_ids[:top_k]
    colors     = _make_palette(len(top_ids))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.tick_params(labelsize=8)

    # Nền xám
    rest_mask = ~np.isin(labels, top_ids)
    if rest_mask.any():
        ax.scatter(emb[rest_mask, 0], emb[rest_mask, 1],
                   c="#cccccc", s=6, alpha=0.25, edgecolors="none", zorder=1)

    # Top-K màu
    for color, uid in zip(colors, top_ids):
        mask = labels == uid
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   color=color, s=35, alpha=0.85,
                   edgecolors="k", linewidths=0.3, zorder=2, label=f"ID {uid}")

    ax.legend(loc="upper right", fontsize=7, ncol=2, markerscale=1.2,
              framealpha=0.8, edgecolor="#aaaaaa")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    if logger:
        logger.info(f"Saved t-SNE plot: {save_path}")

def plot_score_distribution(genuine: list, impostor: list,
                             eer: float, thresh: float,
                             title: str, save_path: str,
                             logger: logging.Logger = None):
    """Vẽ phân phối điểm Genuine / Impostor và đường EER."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(impostor, bins=80, color="#e74c3c", alpha=0.6, label="Impostor", density=True)
    ax.hist(genuine,  bins=80, color="#2ecc71", alpha=0.6, label="Genuine",  density=True)
    ax.axvline(thresh, color="#e67e22", linestyle="--", linewidth=1.6,
               label=f"EER thresh={thresh:.3f}  (EER={eer:.2f}%)")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    if logger:
        logger.info(f"Saved score distribution: {save_path}")


def plot_uncertainty_distribution(known_unc: list, stranger_unc: list,
                                  eer: float, thresh: float,
                                  title: str, save_path: str,
                                  logger: logging.Logger = None):
    """Vẽ phân phối Uncertainty cho Known và Stranger."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(stranger_unc, bins=80, color="#e74c3c", alpha=0.6, label="Stranger (Out-of-Distribution)", density=True)
    ax.hist(known_unc,    bins=80, color="#2ecc71", alpha=0.6, label="Known Probe",  density=True)
    ax.axvline(thresh, color="#e67e22", linestyle="--", linewidth=1.6,
               label=f"Uncertainty thresh={thresh:.4f}  (EER={eer:.2f}%)")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Uncertainty Score (mean exp(logvar))")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    if logger:
        logger.info(f"Saved uncertainty distribution: {save_path}")


def plot_sigma_distribution(known_sigma: list, stranger_sigma: list,
                            title: str, save_path: str,
                            logger: logging.Logger = None):
    """Vẽ phân phối Sigma cho Known và Stranger."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(stranger_sigma, bins=80, color="#e74c3c", alpha=0.6, label="Stranger (Out-of-Distribution)", density=True)
    ax.hist(known_sigma,    bins=80, color="#2ecc71", alpha=0.6, label="Known Probe",  density=True)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Sigma Score (mean exp(0.5 * logvar))")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    if logger:
        logger.info(f"Saved sigma distribution: {save_path}")


def plot_openset_score_distribution(genuine: list, impostor: list, stranger_max: list,
                                    eer: float, thresh: float,
                                    title: str, save_path: str,
                                    logger: logging.Logger = None):
    """
    Vẽ phân phối điểm cho 3 nhóm trong kịch bản Open-Set:
      - Genuine       : known probe khớp đúng gallery
      - Known-Impostor: known probe khớp sai gallery
      - Stranger      : người lạ (max-sim tới bất kỳ gallery)
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    if impostor:
        ax.hist(impostor,     bins=80, color="#f39c12", alpha=0.55,
                label="Known-Impostor", density=True)
    if stranger_max:
        ax.hist(stranger_max,     bins=80, color="#e74c3c", alpha=0.60,
                label="Stranger (max-sim)", density=True)
    if genuine:
        ax.hist(genuine,          bins=80, color="#2ecc71", alpha=0.65,
                label="Genuine (known probe)", density=True)

    ax.axvline(thresh, color="#8e44ad", linestyle="--", linewidth=1.8,
               label=f"EER thresh={thresh:.3f}  (EER={eer:.2f}%)")

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    if logger:
        logger.info(f"Saved open-set score distribution: {save_path}")

def plot_oscr(fpr_list: list, ccr_list: list, oscr_area: float, save_path: str, logger: logging.Logger = None):
    """Vẽ đường OSCR (Open-Set Classification Rate)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_list, ccr_list, color='#2980b9', lw=2, label=f"OSCR Area: {oscr_area:.2f}%")
    ax.set_title("Open-Set Classification Rate (OSCR)", fontsize=13, pad=10)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("Correct Classification Rate (CCR)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    if logger:
        logger.info(f"Saved OSCR curve: {save_path}")

def plot_risk_coverage(coverage_list: list, risk_list: list, save_path: str, logger: logging.Logger = None):
    """Vẽ Risk-Coverage Curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(coverage_list, risk_list, color='#c0392b', lw=2, label="Risk-Coverage Curve")
    ax.set_title("Risk-Coverage Curve (Definition B)", fontsize=13, pad=10)
    ax.set_xlabel("Coverage (Fraction of accepted probes)")
    ax.set_ylabel("Risk (Error rate among accepted)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    if logger:
        logger.info(f"Saved Risk-Coverage curve: {save_path}")


# ==============================================================================
# ENTRY POINT (HYDRA)
# ==============================================================================
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # --- Thiết lập Random Seed ---
    seed = config.get("seed", config.get("dataset", {}).get("seed", 42))
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # -----------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Debug: in raw eval dict từ Hydra để kiểm tra config có được đọc không ──
    raw_eval = config.get("eval", {})
    print(f"[DEBUG] Raw eval config from Hydra: {raw_eval}")

    # ── Lấy eval config: merge DEFAULT với giá trị từ Hydra (Hydra ưu tiên) ────
    eval_cfg = {**DEFAULT_EVAL_CFG, **raw_eval}
    print(f"[DEBUG] Final eval_cfg after merge: mode={eval_cfg['mode']}, neg_strategy={eval_cfg['neg_strategy']}, gallery_split={eval_cfg.get('gallery_split')}")

    mode         = eval_cfg["mode"]
    neg_strategy = eval_cfg["neg_strategy"]
    gallery_split = eval_cfg.get("gallery_split", "train")
    probe_split   = eval_cfg.get("probe_split", "test")
    batch_size    = eval_cfg.get("batch_size", 32)
    force         = eval_cfg.get("force_reextract", False)
    tsne_top_k    = eval_cfg.get("tsne_top_k", 20)
    tsne_perp     = eval_cfg.get("tsne_perplexity", 30)

    # ── Tên run để đặt tên folder ──────────────────────────────────────────────
    MODE_NAME = {0: "baseline", 1: "opt_proj", 2: "opt_latent", 3: "baseline_mu", 4: "opt_latent_mu"}
    run_name = f"mode{mode}_{MODE_NAME.get(mode,'custom')}_{neg_strategy}"

    # ── Load model ─────────────────────────────────────────────────────────────
    logger, log_path = make_logger("tasks", f"eval_{run_name}")
    logger.info(f"[Eval] mode={mode} | neg_strategy={neg_strategy} | device={device}")

    model, version_dir = load_model(config, device, logger)

    # ── Xác định output_dir ────────────────────────────────────────────────────
    out_base = eval_cfg.get("output_dir", "")
    if not out_base:
        out_base = os.path.join(version_dir, "eval") if version_dir else "tasks/eval_results"
    output_dir = os.path.join(out_base, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    shared_cache_dir = out_base  # version_dir/eval/ — dùng chung feature cache

    # ── Phát hiện kịch bản ─────────────────────────────────────────────────────
    split_mode    = config.get("dataset", {}).get("split_mode", "")
    is_person_mode = split_mode == "person"

    if is_person_mode:
        # ══════════════════════════════════════════════════════════════════════
        # OPEN-SET RECOGNITION MODE  (split_mode = 'person')
        # Gallery = register / Probe = known probe / Stranger = người lạ
        # ══════════════════════════════════════════════════════════════════════
        logger.info(f"[Open-Set Mode] split_mode='{split_mode}' — building 3 dataloaders")

        register_loader = make_dataloader(config, "register", batch_size)
        probe_loader    = make_dataloader(config, "probe",    batch_size)
        stranger_loader = make_dataloader(config, "stranger", batch_size)

        reg_feat_cache  = os.path.join(shared_cache_dir, "feats_register.pt")
        prob_feat_cache = os.path.join(shared_cache_dir, "feats_probe.pt")
        str_feat_cache  = os.path.join(shared_cache_dir, "feats_stranger.pt")
        gallery_cache   = os.path.join(output_dir, "gallery.pt")
        results_path    = os.path.join(output_dir, "results.json")

        # ── STEP 1: Extract features ─────────────────────────────────────
        logger.info("--- STEP 1: Extracting features (register / probe / stranger) ---")
        reg_mu,  reg_proj,  reg_logvar,  reg_labels,  reg_ref = extract_features(
            model, register_loader, device, reg_feat_cache,  force=force, logger=logger)
        prob_mu, prob_proj, prob_logvar, prob_labels, _ = extract_features(
            model, probe_loader,    device, prob_feat_cache, force=force, logger=logger)
        str_mu,  str_proj,  str_logvar,  str_labels,  _ = extract_features(
            model, stranger_loader, device, str_feat_cache,  force=force, logger=logger)

        logger.info(f"Register : {reg_mu.shape[0]} samples, {reg_labels.unique().numel()} identities")
        logger.info(f"Probe    : {prob_mu.shape[0]} samples, {prob_labels.unique().numel()} identities")
        logger.info(f"Stranger : {str_mu.shape[0]} samples, {str_labels.unique().numel()} identities")

        # ── STEP 2: Build gallery từ register split ───────────────────────
        logger.info("--- STEP 2: Building gallery (from register split) ---")
        gallery = build_gallery(
            model, reg_mu, reg_proj, reg_logvar, reg_labels,
            eval_cfg=eval_cfg, device=device,
            cache_path=gallery_cache, force=force, logger=logger,
            ref_images=reg_ref,
        )

        # ── Chọn vector eval theo mode ────────────────────────────────────
        is_mu_mode = (mode in [3, 4])
        prob_eval = prob_mu if is_mu_mode else prob_proj
        str_eval  = str_mu  if is_mu_mode else str_proj

        # ── STEP 3a: Closed-set eval trên known probe ─────────────────────
        logger.info("--- STEP 3a: Closed-set evaluation on known probe ---")
        closed_results = evaluate_gallery(gallery, prob_eval, prob_labels, device)

        logger.info("=" * 60)
        logger.info("  [Closed-Set — Known Probe Only]")
        logger.info(f"  Rank-1 Accuracy : {closed_results['rank1_acc']:.2f}%")
        logger.info(f"  EER             : {closed_results['eer']:.2f}%  (thresh={closed_results['eer_threshold']:.4f})")
        logger.info(f"  Mean Genuine    : {closed_results['mean_genuine']:.4f}")
        logger.info(f"  Mean Impostor   : {closed_results['mean_impostor']:.4f}")
        logger.info(f"  Probe count     : {closed_results['n_probe']}")
        logger.info("=" * 60)

        # ── STEP 3b: Open-set eval với stranger ───────────────────────────
        logger.info("--- STEP 3b: Open-set evaluation with stranger ---")
        open_results = evaluate_open_set(
            gallery,
            prob_eval, prob_labels,
            str_eval,  str_labels,
            device,
            eer_thresh=closed_results["eer_threshold"],
            known_mu=prob_mu, known_logvar=prob_logvar,
            stranger_mu=str_mu, stranger_logvar=str_logvar,
        )

        logger.info("=" * 60)
        logger.info("  [Open-Set — Known + Stranger (2-Stage Rejection)]")
        logger.info(f"  Open-Set Rank-1 : {open_results['open_set_rank1']:.2f}%")
        logger.info(f"  KL-Gate EER     : {open_results.get('kl_eer', 0.0):.2f}% (thresh={open_results.get('kl_threshold', 0.0):.4f})")
        logger.info(f"  Uncertainty EER : {open_results['unc_eer']:.2f}% (thresh={open_results['unc_threshold']:.4f})")
        logger.info(f"  Stage-1 FRR     : {open_results['stage1_frr']:.2f}%   - known bi reject boi Uncertainty")
        logger.info(f"  FAR (stranger)  : {open_results['far']:.2f}%   - stranger bi nhan nham (da qua 2 vong)")
        logger.info(f"  FRR (known)     : {open_results['frr']:.2f}%   - known bi reject sai (tong hop)")
        logger.info(f"  EER (open-set)  : {open_results['eer']:.2f}%  (thresh={open_results['eer_threshold']:.4f})")
        logger.info(f"  AUROC (Known-vs-Unknown): {open_results.get('auroc', 0.0):.2f}%")
        logger.info(f"  DIR @ FPIR=1%   : {open_results.get('dir_at_fpir_1%', 0.0):.2f}%")
        logger.info(f"  DIR @ FPIR=0.1% : {open_results.get('dir_at_fpir_0.1%', 0.0):.2f}%")
        logger.info(f"  OSCR Area       : {open_results.get('oscr_area', 0.0):.2f}%")
        logger.info(f"  Mean Genuine    : {open_results['mean_genuine']:.4f}")
        logger.info(f"  Mean Stranger   : {open_results['mean_stranger']:.4f}")
        logger.info(f"  Known probe     : {open_results['n_known_probe']} | Stranger: {open_results['n_stranger']}")
        logger.info("=" * 60)

        # ── Lưu JSON ──────────────────────────────────────────────────────
        all_results = {
            "closed_set": closed_results,
            "open_set":   open_results,
            "config": {
                "mode": mode, "neg_strategy": neg_strategy,
                "run_name": run_name, "split_mode": split_mode,
                "num_train_persons":    config.get("dataset", {}).get("num_train_persons"),
                "num_known_persons":    config.get("dataset", {}).get("num_known_persons"),
                "num_stranger_persons": config.get("dataset", {}).get("num_stranger_persons"),
                "register_ratio":       config.get("dataset", {}).get("register_ratio"),
                "checkpoint": config.get("checkpoint", ""),
            },
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved results: {results_path}")

        # ── STEP 4: Open-Set Score Distribution ───────────────────────────
        logger.info("--- STEP 4: Visualizing open-set score & uncertainty distribution ---")
        plot_openset_score_distribution(
            genuine      = open_results["genuine_scores"],
            impostor     = closed_results.get("impostor_scores", []),
            stranger_max = open_results["stranger_max_sims"],
            eer          = open_results["eer"],
            thresh       = open_results["eer_threshold"],
            title        = (f"Open-Set Score Distribution [{run_name}]\n"
                            f"Rank-1={open_results['open_set_rank1']:.2f}%  "
                            f"FAR={open_results['far']:.2f}%  "
                            f"FRR={open_results['frr']:.2f}%"),
            save_path    = os.path.join(output_dir, "score_distribution_openset.png"),
            logger       = logger,
        )

        if "oscr_fpr_list" in open_results and len(open_results["oscr_fpr_list"]) > 0:
            plot_oscr(
                fpr_list  = open_results["oscr_fpr_list"],
                ccr_list  = open_results["oscr_ccr_list"],
                oscr_area = open_results["oscr_area"],
                save_path = os.path.join(output_dir, "oscr_curve.png"),
                logger    = logger
            )

        if "coverage_list" in open_results and len(open_results["coverage_list"]) > 0:
            plot_risk_coverage(
                coverage_list = open_results["coverage_list"],
                risk_list     = open_results["risk_list"],
                save_path     = os.path.join(output_dir, "risk_coverage.png"),
                logger        = logger
            )

        if open_results.get("known_unc") and open_results.get("stranger_unc"):
            plot_uncertainty_distribution(
                known_unc    = open_results["known_unc"],
                stranger_unc = open_results["stranger_unc"],
                eer          = open_results["unc_eer"],
                thresh       = open_results["unc_threshold"],
                title        = (f"Uncertainty Distribution [{run_name}]\n"
                                f"Stage-1 FRR={open_results['stage1_frr']:.2f}%"),
                save_path    = os.path.join(output_dir, "uncertainty_distribution_openset.png"),
                logger       = logger,
            )

        if open_results.get("known_sigma") and open_results.get("stranger_sigma"):
            plot_sigma_distribution(
                known_sigma  = open_results["known_sigma"],
                stranger_sigma = open_results["stranger_sigma"],
                title        = f"Sigma Distribution [{run_name}]",
                save_path    = os.path.join(output_dir, "sigma_distribution_openset.png"),
                logger       = logger,
            )

        # ── STEP 5: t-SNE ─────────────────────────────────────────────────
        logger.info("--- STEP 5: t-SNE visualization ---")
        n_known_ids       = int(reg_labels.max().item()) + 1
        str_labels_offset = str_labels + n_known_ids

        all_feats_np  = torch.cat([reg_proj, prob_proj, str_proj], dim=0).numpy()
        all_labels_np = torch.cat([reg_labels, prob_labels, str_labels_offset], dim=0).numpy()

        plot_tsne(
            feats      = all_feats_np,
            labels     = all_labels_np,
            title      = (f"t-SNE (Proj Space) [{run_name}]\n"
                          f"known IDs 0..{n_known_ids-1},  strangers {n_known_ids}+"),
            save_path  = os.path.join(output_dir, "tsne_proj.png"),
            top_k      = tsne_top_k,
            perplexity = tsne_perp,
            logger     = logger,
        )

        all_mu_np = torch.cat([reg_mu, prob_mu, str_mu], dim=0).numpy()
        plot_tsne(
            feats      = all_mu_np,
            labels     = all_labels_np,
            title      = (f"t-SNE (Mu Space) [{run_name}]\n"
                          f"known IDs 0..{n_known_ids-1},  strangers {n_known_ids}+"),
            save_path  = os.path.join(output_dir, "tsne_mu.png"),
            top_k      = tsne_top_k,
            perplexity = tsne_perp,
            logger     = logger,
        )

    else:
        # ══════════════════════════════════════════════════════════════════════
        # CLOSED-SET MODE  (hand / ratio / session / mixed)
        # ══════════════════════════════════════════════════════════════════════
        logger.info(f"Building dataloader: gallery_split='{gallery_split}', probe_split='{probe_split}'")
        gallery_loader = make_dataloader(config, gallery_split, batch_size)
        probe_loader   = make_dataloader(config, probe_split,   batch_size)

        gallery_feat_cache = os.path.join(shared_cache_dir, f"feats_{gallery_split}.pt")
        probe_feat_cache   = os.path.join(shared_cache_dir, f"feats_{probe_split}.pt")
        gallery_cache      = os.path.join(output_dir, "gallery.pt")
        results_path       = os.path.join(output_dir, "results.json")

        # ── STEP 1: Extract features ──────────────────────────────────────
        logger.info("--- STEP 1: Extracting features ---")
        g_mu, g_proj, g_logvar, g_labels, g_ref_images = extract_features(
            model, gallery_loader, device, gallery_feat_cache, force=force, logger=logger)
        p_mu, p_proj, p_logvar, p_labels, _ = extract_features(
            model, probe_loader, device, probe_feat_cache, force=force, logger=logger)

        logger.info(f"Gallery: {g_mu.shape[0]} samples, {g_labels.unique().numel()} identities ({len(g_ref_images)} ref images cached)")
        logger.info(f"Probe  : {p_mu.shape[0]} samples, {p_labels.unique().numel()} identities")

        # ── STEP 2: Build gallery ─────────────────────────────────────────
        logger.info("--- STEP 2: Building gallery ---")
        gallery = build_gallery(
            model, g_mu, g_proj, g_logvar, g_labels,
            eval_cfg=eval_cfg, device=device,
            cache_path=gallery_cache, force=force, logger=logger,
            ref_images=g_ref_images,
        )

        # ── STEP 3: Evaluate ──────────────────────────────────────────────
        logger.info("--- STEP 3: Evaluating ---")
        results = evaluate_gallery(gallery, p_proj, p_labels, device)

        logger.info("=" * 60)
        logger.info(f"  Rank-1 Accuracy : {results['rank1_acc']:.2f}%")
        logger.info(f"  EER             : {results['eer']:.2f}%  (threshold={results['eer_threshold']:.4f})")
        logger.info(f"  Mean Genuine    : {results['mean_genuine']:.4f}")
        logger.info(f"  Mean Impostor   : {results['mean_impostor']:.4f}")
        logger.info(f"  Probe count     : {results['n_probe']}")
        logger.info(f"  Gallery IDs     : {results['n_gallery_ids']}")
        logger.info("=" * 60)

        results["config"] = {
            "mode": mode, "neg_strategy": neg_strategy,
            "run_name": run_name,
            "checkpoint": config.get("checkpoint", ""),
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results: {results_path}")

        # ── STEP 4: Score Distribution ────────────────────────────────────
        logger.info("--- STEP 4: Visualizing score distribution ---")
        plot_score_distribution(
            genuine   = results["genuine_scores"],
            impostor  = results["impostor_scores"],
            eer       = results["eer"],
            thresh    = results["eer_threshold"],
            title     = f"Score Distribution [{run_name}]\nRank-1={results['rank1_acc']:.2f}%  EER={results['eer']:.2f}%",
            save_path = os.path.join(output_dir, "score_distribution.png"),
            logger    = logger,
        )

        # ── STEP 5: t-SNE ─────────────────────────────────────────────────
        logger.info("--- STEP 5: t-SNE visualization ---")
        all_feats  = torch.cat([g_proj, p_proj], dim=0).numpy()
        all_labels = torch.cat([g_labels, p_labels], dim=0).numpy()

        plot_tsne(
            feats      = all_feats,
            labels     = all_labels,
            title      = f"t-SNE (Projected Space) [{run_name}]",
            save_path  = os.path.join(output_dir, "tsne_proj.png"),
            top_k      = tsne_top_k,
            perplexity = tsne_perp,
            logger     = logger,
        )

        all_mu_np = torch.cat([g_mu, p_mu], dim=0).numpy()
        plot_tsne(
            feats      = all_mu_np,
            labels     = all_labels,
            title      = f"t-SNE (Mu / Latent Space) [{run_name}]",
            save_path  = os.path.join(output_dir, "tsne_mu.png"),
            top_k      = tsne_top_k,
            perplexity = tsne_perp,
            logger     = logger,
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info(f"Log file: {log_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
