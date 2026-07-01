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
        ch = logging.StreamHandler()
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
    ds_cfg = config.get("dataset", {})
    name   = ds_cfg.get("name", "iitd")
    ddir   = ds_cfg.get("data_dir", "data/IITD")
    is_train = split in ("train",)
    print("========================Chế độ lấy DataLoader:", is_train)
    dataset = DatasetFactory.create(name, ddir, ds_cfg, is_train=is_train)
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
        logger.info(f"Loading gallery cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

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
                gallery_r_list.append(r_i)
            gallery_r = torch.stack(gallery_r_list, dim=0)  # (N, proj_dim)

        elif mode == 2:
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
                with torch.no_grad():
                    proj_r = model.projector(mu_c + r_latent)
                    proj_r = F.normalize(proj_r, p=2, dim=1)
                gallery_r_list.append(proj_r.squeeze(0))
            gallery_r = torch.stack(gallery_r_list, dim=0)  # (N, proj_dim)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 0, 1, or 2.")

        gallery[int(lbl)] = gallery_r.cpu()

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
    from sklearn.metrics import roc_curve
    y_true  = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    y_score = genuine_scores + impostor_scores
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer  = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    thresh = thresholds[eer_idx]
    return float(eer * 100), float(thresh)

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
        t = gallery[lbl].to(device)
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

# ==============================================================================
# ENTRY POINT (HYDRA)
# ==============================================================================
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
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
    MODE_NAME = {0: "baseline", 1: "opt_proj", 2: "opt_latent"}
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

    # ── Cache paths ────────────────────────────────────────────────────────────
    # Feature cache chia sẻ giữa tất cả mode cùng checkpoint (nằm ngoài run_name)
    shared_cache_dir   = out_base  # version_dir/eval/
    gallery_feat_cache = os.path.join(shared_cache_dir, f"feats_{gallery_split}.pt")
    probe_feat_cache   = os.path.join(shared_cache_dir, f"feats_{probe_split}.pt")
    gallery_cache      = os.path.join(output_dir, "gallery.pt")
    results_path       = os.path.join(output_dir, "results.json")

    # ── Load dataset ───────────────────────────────────────────────────────────
    logger.info(f"Building dataloader: gallery_split='{gallery_split}', probe_split='{probe_split}'")
    gallery_loader = make_dataloader(config, gallery_split, batch_size)
    probe_loader   = make_dataloader(config, probe_split,   batch_size)

    # ── Extract features ───────────────────────────────────────────────────────
    logger.info("--- STEP 1: Extracting features ---")
    g_mu, g_proj, g_logvar, g_labels, g_ref_images = extract_features(
        model, gallery_loader, device, gallery_feat_cache, force=force, logger=logger)
    p_mu, p_proj, p_logvar, p_labels, _ = extract_features(
        model, probe_loader, device, probe_feat_cache, force=force, logger=logger)

    logger.info(f"Gallery: {g_mu.shape[0]} samples, {g_labels.unique().numel()} identities ({len(g_ref_images)} ref images cached)")
    logger.info(f"Probe  : {p_mu.shape[0]} samples, {p_labels.unique().numel()} identities")

    # ── Build gallery ──────────────────────────────────────────────────────────
    logger.info("--- STEP 2: Building gallery ---")
    gallery = build_gallery(
        model, g_mu, g_proj, g_logvar, g_labels,
        eval_cfg=eval_cfg, device=device,
        cache_path=gallery_cache, force=force, logger=logger,
        ref_images=g_ref_images,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
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

    # Lưu kết quả JSON
    results["config"] = {
        "mode": mode, "neg_strategy": neg_strategy,
        "run_name": run_name,
        "checkpoint": config.get("checkpoint", ""),
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results: {results_path}")

    # ── Visualize: Score Distribution ──────────────────────────────────────────
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

    # ── Visualize: t-SNE (Proj Space) ─────────────────────────────────────────
    logger.info("--- STEP 5: t-SNE visualization ---")
    # Dùng gallery features để visualize (tập val có thể nhỏ hơn, dễ nhìn hơn)
    all_feats  = torch.cat([g_proj, p_proj], dim=0).numpy()
    all_labels = torch.cat([g_labels, p_labels], dim=0).numpy()

    plot_tsne(
        feats     = all_feats,
        labels    = all_labels,
        title     = f"t-SNE (Projected Space) [{run_name}]",
        save_path = os.path.join(output_dir, "tsne_proj.png"),
        top_k     = tsne_top_k,
        perplexity= tsne_perp,
        logger    = logger,
    )

    all_mu_np     = torch.cat([g_mu, p_mu], dim=0).numpy()
    plot_tsne(
        feats     = all_mu_np,
        labels    = all_labels,
        title     = f"t-SNE (Mu / Latent Space) [{run_name}]",
        save_path = os.path.join(output_dir, "tsne_mu.png"),
        top_k     = tsne_top_k,
        perplexity= tsne_perp,
        logger    = logger,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info(f"Log file: {log_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
