"""
pca_latent.py
=============
Phân tích PCA trên không gian latent (mu) của ProbabilisticPalmModel.
Nhận đầu vào là train dataset để đánh giá chất lượng của không gian mu mà 
model đã học được sau quá trình huấn luyện.

Cách chạy:
    python tools/pca_latent.py \
        checkpoint="logs/version_X/checkpoints/last.ckpt" \
        dataset.name=PalmPrintDataset \
        dataset.data_dir=data/IITD \
        pca.split=train \
        pca.n_components=2 \
        pca.top_k_ids=30 \
        pca.output_dir=""
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import yaml
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.models import ProbabilisticPalmModel, UNetPalmModel
from src.datasets.factory import DatasetFactory


# ==============================================================================
# 1. HELPER: TẢI MODEL
# ==============================================================================
def load_model(config: dict, device: torch.device):
    """Khởi tạo model và nạp trọng số từ checkpoint."""
    checkpoint_path = config.get('checkpoint', '')
    version_dir = ''

    # ── Bước 1: Đọc config_backup.yaml TRƯỚC khi build model ──────────────────
    if checkpoint_path:
        match = re.search(r'(.*[\\/]version_\d+)', checkpoint_path.replace('\\', '/'))
        if match:
            version_dir = match.group(1)

    backup_path = os.path.join(version_dir, 'config_backup.yaml') if version_dir else ''
    if backup_path and os.path.exists(backup_path):
        print(f"Tìm thấy {backup_path}, đọc cấu hình model từ backup trước khi khởi tạo...")
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_cfg = yaml.safe_load(f)
        if 'model' in backup_cfg:
            config['model'] = backup_cfg['model']
            print(f"  → Dùng model config từ config_backup.yaml (type={backup_cfg['model'].get('type', 'default')})")
        if 'dataset' in backup_cfg and 'image_size' in backup_cfg['dataset']:
            config.setdefault('dataset', {})['image_size'] = backup_cfg['dataset']['image_size']
            print(f"  → image_size: {backup_cfg['dataset']['image_size']}")
    else:
        if version_dir:
            print(f"CẢNH BÁO: Không tìm thấy config_backup.yaml tại {version_dir}. Dùng config hiện tại.")

    # ── Bước 2: Build model từ config đã được cập nhật ────────────────────────
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'default')

    if 'decoder' not in model_config:
        model_config['decoder'] = {}
    if 'image_size' not in model_config['decoder']:
        model_config['decoder']['image_size'] = config.get('dataset', {}).get('image_size', [128, 128])

    if model_type == 'unet':
        model = UNetPalmModel(model_config).to(device)
    else:
        model = ProbabilisticPalmModel(model_config).to(device)

    # ── Bước 3: Nạp trọng số từ checkpoint ────────────────────────────────────
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        # Bỏ prefix "model." của PyTorch Lightning
        clean = {(k[6:] if k.startswith('model.') else k): v for k, v in state_dict.items()}
        model.load_state_dict(clean, strict=False)
        print(f"Đã nạp checkpoint: {checkpoint_path}")
    else:
        print("CẢNH BÁO: Không tìm thấy checkpoint. Đang dùng trọng số ngẫu nhiên.")

    model.eval()
    return model, version_dir


# ==============================================================================
# 2. HELPER: TRÍCH XUẤT MU TỪ DATASET
# ==============================================================================
@torch.no_grad()
def extract_features(model, dataloader: DataLoader, device: torch.device):
    """
    Chạy forward pass qua toàn bộ dataloader, thu thập:
    - all_mu   : (N, latent_dim) — vectơ mu của từng mẫu
    - all_proj : (N, proj_dim)   — vectơ projected (sau MLP projector)
    - all_labels: (N,)           — nhãn identity của từng mẫu
    """
    all_mu    = []
    all_proj  = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Extracting mu & proj"):
        # Hỗ trợ cả tuple (images, labels) và dict {'image':..., 'label':...}
        if isinstance(batch, (tuple, list)):
            images, labels = batch[0], batch[1]
        elif isinstance(batch, dict):
            images = batch.get('image', batch.get('img'))
            labels = batch.get('label', batch.get('id'))
        else:
            raise ValueError(f"Không hỗ trợ định dạng batch: {type(batch)}")

        images = images.to(device)

        if isinstance(model, UNetPalmModel):
            # UNetPalmModel dùng latent_encoder (không phải encoder)
            mu, _ = model.latent_encoder(images)
            proj = model.projector(mu)
        else:
            # ProbabilisticPalmModel
            out = model(images)
            mu   = out['mu']
            proj = out['proj']

        all_mu.append(mu.cpu().numpy())
        all_proj.append(proj.cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

    all_mu     = np.concatenate(all_mu,     axis=0)
    all_proj   = np.concatenate(all_proj,   axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_mu, all_proj, all_labels


# ==============================================================================
# 3. HELPER: TẠO BẢNG MÀU PHÂN BIỆT
# ==============================================================================
def _make_color_palette(n: int):
    """
    Tạo list n màu phân biệt tối đa bằng cách kết hợp nhiều colormap.
    Đảm bảo mỗi màu đều rõ ràng, không bị trùng lặp.
    """
    base_colors = [
        # tab10 (10 màu mạnh, tương phản cao)
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        # Set2 (8 màu pastel phân biệt)
        '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
        '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3',
        # Dark2 (8 màu đậm)
        '#1b9e77', '#d95f02', '#7570b3', '#e7298a',
    ]
    return base_colors[:n]


# ==============================================================================
# HELPER: VẼ 2D SCATTER VỚI TOP-N MÀU + NỀN XÁM
# ==============================================================================
def _scatter_top_ids(ax, feats_2d, all_labels, unique_ids, top_n=20,
                     s_bg=10, s_fg=40, alpha_bg=0.25, alpha_fg=0.85,
                     marker='o'):
    """
    Vẽ scatter: top_n IDs đầu tiên với màu phân biệt, phần còn lại là nền xám.
    Trả về danh sách (uid, color) của các ID được highlight để build legend.
    """
    show_ids    = unique_ids[:top_n]
    rest_mask   = np.isin(all_labels, show_ids, invert=True)
    colors      = _make_color_palette(len(show_ids))

    # Nền xám: tất cả các điểm KHÔNG thuộc top_n
    if rest_mask.any():
        ax.scatter(feats_2d[rest_mask, 0], feats_2d[rest_mask, 1],
                   c='#cccccc', s=s_bg, alpha=alpha_bg,
                   edgecolors='none', zorder=1, label='_nolegend_')

    # Top-N IDs với màu phân biệt
    legend_handles = []
    for color, uid in zip(colors, show_ids):
        mask = all_labels == uid
        pts  = feats_2d[mask]
        sc   = ax.scatter(pts[:, 0], pts[:, 1], color=color, s=s_fg,
                          alpha=alpha_fg, edgecolors='k', linewidths=0.3,
                          zorder=2, marker=marker,
                          label=f'ID {uid}')
        legend_handles.append(sc)
    return legend_handles

def _plot_pca_space(feats: np.ndarray, all_labels: np.ndarray,
                    n_components: int, top_k_ids: int,
                    output_dir: str, prefix: str, space_name: str):
    """
    Vẽ đầy đủ bộ biểu đồ PCA cho một không gian đặc trưng (mu hoặc proj).
      A. PCA 2D toàn cục — tô màu theo identity
      B. Phương sai giải thích (Explained Variance + Cumulative)
      C. Histogram phân phối giá trị theo từng chiều (10 chiều đầu)
      D. Norm L2 của vector đặc trưng
      E. PCA 2D highlight top-K identities
    """
    n_components = min(n_components, feats.shape[1], feats.shape[0])
    unique_ids   = np.unique(all_labels)
    n_ids        = len(unique_ids)
    colormap     = cm.get_cmap('tab20', min(n_ids, 20))

    # Fit PCA
    pca      = PCA(n_components=n_components)
    feats_2d = pca.fit_transform(feats)
    explained = pca.explained_variance_ratio_
    cum_var   = np.cumsum(explained)

def _plot_pca_space(feats: np.ndarray, all_labels: np.ndarray,
                    n_components: int, top_k_ids: int,
                    output_dir: str, prefix: str, space_name: str):
    """
    Vẽ đầy đủ bộ biểu đồ PCA cho một không gian đặc trưng (mu hoặc proj).
      A. PCA 2D — top-K IDs màu phân biệt, phần còn lại nền xám
      B. Phương sai giải thích (Explained Variance + Cumulative)
      C. Histogram phân phối giá trị theo từng chiều (10 chiều đầu)
      D. Norm L2 của vector đặc trưng
    """
    TOP_N        = min(top_k_ids, 20)  # tối đa 20 IDs để đảm bảo màu phân biệt
    n_components = min(n_components, feats.shape[1], feats.shape[0])
    unique_ids   = np.unique(all_labels)

    # Fit PCA
    pca       = PCA(n_components=n_components)
    feats_2d  = pca.fit_transform(feats)
    explained = pca.explained_variance_ratio_
    cum_var   = np.cumsum(explained)

    # ── A: PCA 2D — top-N IDs màu phân biệt ─────────────────────────────────
    print(f"  [A-{prefix}] Vẽ PCA 2D {space_name} (top {TOP_N} IDs)...")
    fig, ax = plt.subplots(figsize=(11, 8))
    handles = _scatter_top_ids(ax, feats_2d[:, :2], all_labels, unique_ids,
                                top_n=TOP_N, s_bg=12, s_fg=50)
    ax.set_title(f'PCA 2D — {space_name}\n'
                 f'(PC1={explained[0]:.1%}, PC2={explained[1]:.1%}  |  '
                 f'Hiển thị top {TOP_N}/{len(unique_ids)} IDs)', fontsize=12)
    ax.set_xlabel(f'PC1 ({explained[0]:.2%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({explained[1]:.2%})', fontsize=11)
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              ncol=max(1, TOP_N // 10), framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.35)
    save_path = os.path.join(output_dir, f'{prefix}_pca_2d.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"     Đã lưu: {save_path}")

    # ── B: Explained Variance ─────────────────────────────────────────────────
    print(f"  [B-{prefix}] Vẽ Explained Variance {space_name}...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, n_components + 1), explained, color='steelblue', alpha=0.8)
    axes[0].set_title(f'{space_name} — Explained Variance per Component', fontsize=12)
    axes[0].set_xlabel('Principal Component'); axes[0].set_ylabel('Ratio')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].plot(range(1, n_components + 1), cum_var, 'o-', color='tomato')
    axes[1].axhline(y=0.95, color='gray', linestyle='--', label='95%')
    axes[1].set_title(f'{space_name} — Cumulative Explained Variance', fontsize=12)
    axes[1].set_xlabel('Number of Components'); axes[1].set_ylabel('Cumulative Ratio')
    axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.5)
    n95 = np.searchsorted(cum_var, 0.95) + 1
    print(f"     Cần {n95} PC để đạt 95% phương sai. PC1+PC2 = {cum_var[1]:.1%}")
    save_path = os.path.join(output_dir, f'{prefix}_pca_explained_variance.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"     Đã lưu: {save_path}")

    # ── C: Histogram phân phối ────────────────────────────────────────────────
    print(f"  [C-{prefix}] Vẽ histogram phân phối {space_name} (10 chiều đầu)...")
    n_show = min(10, feats.shape[1])
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    axes = axes.flatten()
    for i in range(n_show):
        axes[i].hist(feats[:, i], bins=50, color='mediumpurple', alpha=0.8, edgecolor='none')
        axes[i].set_title(f'dim {i}', fontsize=10)
        axes[i].set_xlabel('Value'); axes[i].set_ylabel('Count')
        axes[i].grid(axis='y', linestyle='--', alpha=0.4)
    for i in range(n_show, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(f'Phân phối {space_name} theo từng chiều (10 chiều đầu)', fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}_distribution.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"     Đã lưu: {save_path}")

    # ── D: Norm L2 ────────────────────────────────────────────────────────────
    print(f"  [D-{prefix}] Vẽ phân phối L2-Norm {space_name}...")
    norms = np.linalg.norm(feats, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(norms, bins=60, color='teal', alpha=0.8, edgecolor='none')
    ax.axvline(norms.mean(),    color='red',    linestyle='--', label=f'Mean = {norms.mean():.2f}')
    ax.axvline(np.median(norms), color='orange', linestyle='--', label=f'Median = {np.median(norms):.2f}')
    ax.set_title(f'L2-Norm của {space_name}', fontsize=12)
    ax.set_xlabel('||v||'); ax.set_ylabel('Count')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.4)
    save_path = os.path.join(output_dir, f'{prefix}_norm.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"     Đã lưu: {save_path}")
    print(f"     Thống kê ||v||: mean={norms.mean():.3f}, std={norms.std():.3f}, "
          f"min={norms.min():.3f}, max={norms.max():.3f}")

    # ── E: (đã gộp vào biểu đồ A ở trên — không cần plot riêng nữa)


# ==============================================================================
# 4. HELPER: t-SNE VISUALIZATION CHO MỘT KHÔNG GIAN
# ==============================================================================
def _plot_tsne_space(feats: np.ndarray, all_labels: np.ndarray,
                     top_k_ids: int, perplexity: int, n_iter: int,
                     output_dir: str, prefix: str, space_name: str):
    """
    Vẽ t-SNE 2D cho một không gian đặc trưng:
      F. t-SNE 2D — top-K IDs màu phân biệt, phần còn lại nền xám
    """
    TOP_N      = min(top_k_ids, 20)  # tối đa 20 IDs
    unique_ids = np.unique(all_labels)

    # Nếu số chiều lớn, dùng PCA để giảm xuống 50 trước khi chạy t-SNE (faster)
    feats_input = feats
    if feats.shape[1] > 50:
        pre_pca     = PCA(n_components=50)
        feats_input = pre_pca.fit_transform(feats)
        print(f"  [t-SNE-{prefix}] Pre-reduced {feats.shape[1]}D → 50D bằng PCA trước t-SNE")

    print(f"  [F-{prefix}] Chạy t-SNE (perplexity={perplexity}, max_iter={n_iter}) cho {space_name}...")
    try:
        # sklearn >= 1.5: n_iter đổi thành max_iter
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                    random_state=42, init='pca', learning_rate='auto')
    except TypeError:
        # sklearn < 1.5: dùng n_iter
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                    random_state=42, init='pca', learning_rate='auto')
    feats_2d = tsne.fit_transform(feats_input)
    print(f"  [F-{prefix}] t-SNE hoàn tất. KL divergence = {tsne.kl_divergence_:.4f}")

    # ── F: t-SNE 2D — top-N IDs màu phân biệt ────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    handles = _scatter_top_ids(ax, feats_2d, all_labels, unique_ids,
                                top_n=TOP_N, s_bg=10, s_fg=50)
    ax.set_title(f't-SNE 2D — {space_name}\n'
                 f'(perplexity={perplexity}, KL={tsne.kl_divergence_:.3f}  |  '
                 f'Hiển thị top {TOP_N}/{len(unique_ids)} IDs)', fontsize=12)
    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              ncol=max(1, TOP_N // 10), framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.35)
    save_path = os.path.join(output_dir, f'{prefix}_tsne_2d.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"     Đã lưu: {save_path}")


# ==============================================================================
# 5. PHÂN TÍCH PCA + t-SNE CHO CẢ MU VÀ PROJ
# ==============================================================================
def run_pca_analysis(all_mu: np.ndarray, all_proj: np.ndarray,
                     all_labels: np.ndarray,
                     n_components: int, top_k_ids: int, output_dir: str,
                     tsne_perplexity: int = 30, tsne_n_iter: int = 1000):
    """
    Chạy PCA + t-SNE và lưu biểu đồ phân tích cho cả Mu và Proj.
    Mỗi không gian có:
      - PCA: 5 biểu đồ A-E  (prefix: mu_ / proj_)
      - t-SNE: 2 biểu đồ   (prefix: mu_ / proj_)
    """
    os.makedirs(output_dir, exist_ok=True)
    n_ids = len(np.unique(all_labels))

    print(f"\n{'='*60}")
    print(f"PHÂN TÍCH PCA + t-SNE KHÔNG GIAN LATENT")
    print(f"  Số mẫu:      {all_mu.shape[0]}")
    print(f"  Mu dim:      {all_mu.shape[1]}")
    print(f"  Proj dim:    {all_proj.shape[1]}")
    print(f"  Số ID:       {n_ids}")
    print(f"  PCA dims:    {n_components}")
    print(f"  t-SNE perp:  {tsne_perplexity}")
    print(f"{'='*60}")

    # ── Phân tích không gian MU ───────────────────────────────────────────────
    print("\n[MU] Bắt đầu phân tích không gian Mu...")
    _plot_pca_space(all_mu, all_labels,
                    n_components=n_components, top_k_ids=top_k_ids,
                    output_dir=output_dir, prefix='mu', space_name='Mu (Latent Mean)')
    _plot_tsne_space(all_mu, all_labels,
                     top_k_ids=top_k_ids, perplexity=tsne_perplexity,
                     n_iter=tsne_n_iter, output_dir=output_dir,
                     prefix='mu', space_name='Mu (Latent Mean)')

    # ── Phân tích không gian PROJ ─────────────────────────────────────────────
    print("\n[PROJ] Bắt đầu phân tích không gian Projected...")
    _plot_pca_space(all_proj, all_labels,
                    n_components=n_components, top_k_ids=top_k_ids,
                    output_dir=output_dir, prefix='proj', space_name='Projected (After MLP)')
    _plot_tsne_space(all_proj, all_labels,
                     top_k_ids=top_k_ids, perplexity=tsne_perplexity,
                     n_iter=tsne_n_iter, output_dir=output_dir,
                     prefix='proj', space_name='Projected (After MLP)')

    print(f"\n✓ Hoàn tất! Tất cả biểu đồ đã được lưu vào: {output_dir}")


# ==============================================================================
# 4. ENTRY POINT (HYDRA)
# ==============================================================================
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Đọc cấu hình PCA / t-SNE (override qua CLI hoặc dùng default)
    pca_cfg         = config.get('pca', {})
    split           = pca_cfg.get('split',           'train')
    n_comps         = pca_cfg.get('n_components',    50)
    top_k           = pca_cfg.get('top_k_ids',       30)
    batch_size      = pca_cfg.get('batch_size',      64)
    output_dir      = pca_cfg.get('output_dir',      '')
    tsne_perplexity = pca_cfg.get('tsne_perplexity', 30)
    tsne_n_iter     = pca_cfg.get('tsne_n_iter',     1000)

    # Tải model
    model, version_dir = load_model(config, device)

    # Tự động xác định output_dir
    if not output_dir:
        base = version_dir if version_dir else 'logs/unversioned_results'
        output_dir = os.path.join(base, 'pca_latent')

    # Tải dataset train
    dataset_cfg = config.get('dataset', {})
    dataset_name = dataset_cfg.get('name', 'PalmPrintDataset')
    data_dir = dataset_cfg.get('data_dir', 'data/IITD')
    is_train = (split == 'train')

    print(f"\nĐang tải '{split}' dataset từ: {data_dir}")
    dataset = DatasetFactory.create(dataset_name, data_dir=data_dir,
                                    config=dataset_cfg, is_train=is_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=(device.type == 'cuda'))

    # Trích xuất mu và proj
    all_mu, all_proj, all_labels = extract_features(model, dataloader, device)

    # Chạy phân tích PCA + t-SNE cho cả mu và proj
    run_pca_analysis(all_mu, all_proj, all_labels,
                     n_components=n_comps,
                     top_k_ids=top_k,
                     output_dir=output_dir,
                     tsne_perplexity=tsne_perplexity,
                     tsne_n_iter=tsne_n_iter)


if __name__ == '__main__':
    main()
