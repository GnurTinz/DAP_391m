"""
demos/inference_utils.py
========================
Helper functions for real-time palm attendance inference.
Shared between attendance_demo.py and other pages.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Constants ─────────────────────────────────────────────────────────────────
TASK_PATH  = os.path.join(ROOT, "config", "hand_landmarker.task")
IMAGE_SIZE = 128
PADDING    = 30

# ── Image preprocessing (same normalisation as training) ─────────────────────
_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ==============================================================================
# A. MODEL LOADING
# ==============================================================================

def load_model(ckpt_path: str):
    """
    Load GenerativeLightningModule from checkpoint.
    Returns the inner model (UNetPalmModel / ProbabilisticPalmModel) in eval mode.
    """
    from src.engine.lightning_module import GenerativeLightningModule
    module = GenerativeLightningModule.load_from_checkpoint(
        ckpt_path, map_location="cpu"
    )
    model = module.model
    model.eval()
    return model


# ==============================================================================
# B. PREPROCESSING
# ==============================================================================

def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    """Convert PIL image → normalised model input tensor (1, 3, H, W)."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return _transform(pil_img).unsqueeze(0)  # (1, 3, H, W)


# ==============================================================================
# C. HAND ROI CROPPING  (MediaPipe IMAGE mode)
# ==============================================================================

def crop_palm_roi(pil_img: Image.Image, padding: int = PADDING):
    """
    Detect hand landmarks with MediaPipe and crop palm ROI.

    Returns:
        (cropped_pil, bbox_xyxy) if hand found, else (None, None).
        Falls back to returning original image if task file is missing.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.exists(TASK_PATH):
            # No task file – return original image as ROI (no crop)
            return pil_img, None

        np_img = np.array(pil_img.convert("RGB"))
        h, w   = np_img.shape[:2]

        base_options = mp_python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        with vision.HandLandmarker.create_from_options(options) as detector:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
            result = detector.detect(mp_img)

        if not result.hand_landmarks:
            return None, None

        lms   = result.hand_landmarks[0]
        x_pts = [int(lm.x * w) for lm in lms]
        y_pts = [int(lm.y * h) for lm in lms]

        x_min = max(min(x_pts) - padding, 0)
        y_min = max(min(y_pts) - padding, 0)
        x_max = min(max(x_pts) + padding, w)
        y_max = min(max(y_pts) + padding, h)

        if x_max <= x_min or y_max <= y_min:
            return None, None

        cropped = pil_img.crop((x_min, y_min, x_max, y_max))
        return cropped, (x_min, y_min, x_max, y_max)

    except ImportError:
        # MediaPipe not installed – return original image
        return pil_img, None
    except Exception:
        return None, None


# ==============================================================================
# D. EMBEDDING
# ==============================================================================

@torch.no_grad()
def embed_image(model, tensor: torch.Tensor):
    """
    Run encoder on a preprocessed tensor (1, 3, H, W).
    Returns (mu, logvar), each shape (1, latent_dim).

    Supports two model architectures:
      - PalmModel        : uses ``model.encoder``
      - UNetPalmModel    : uses ``model.latent_encoder``
    """
    # UNetPalmModel stores the probabilistic head under `latent_encoder`
    if hasattr(model, "latent_encoder"):
        mu, logvar = model.latent_encoder(tensor)
    elif hasattr(model, "encoder"):
        mu, logvar = model.encoder(tensor)
    else:
        raise AttributeError(
            "Model has neither 'encoder' nor 'latent_encoder' attribute. "
            f"Available attributes: {[a for a in dir(model) if not a.startswith('_')]}"
        )
    return mu, logvar


# ==============================================================================
# E. UNCERTAINTY & DISTANCE METRICS
# ==============================================================================

def uncertainty_score(logvar: torch.Tensor) -> float:
    """U_p = mean(sigma) = mean(exp(0.5 * logvar))."""
    return torch.exp(0.5 * logvar).mean().item()


def _kl_div(mu_a, lv_a, mu_b, lv_b) -> float:
    """KL( N(mu_a, sigma_a^2) || N(mu_b, sigma_b^2) )."""
    var_a = torch.exp(lv_a)
    var_b = torch.exp(lv_b)
    d = mu_a.size(1)
    return 0.5 * (
        (var_a / var_b).sum()
        + ((mu_b - mu_a) ** 2 / var_b).sum()
        - d
        + (lv_b - lv_a).sum()
    ).item()


def symmetric_kl(mu_p, lv_p, mu_g, lv_g) -> float:
    """D_SKL(p, g) = 0.5 * (KL(p||g) + KL(g||p))."""
    return 0.5 * (_kl_div(mu_p, lv_p, mu_g, lv_g) + _kl_div(mu_g, lv_g, mu_p, lv_p))


# ==============================================================================
# F. GALLERY MATCHING (Mode M3 – cosine on latent mu)
# ==============================================================================

def match_gallery(mu_probe, logvar_probe, gallery: dict,
                  tau_S: float, tau_U: float, tau_K: float) -> dict:
    """
    Match probe against gallery using Mode M3 (direct cosine on latent mu).
    Apply tri-threshold open-set rejection.

    Args:
        mu_probe, logvar_probe : (1, latent_dim) probe tensors
        gallery                : dict built by build_gallery()
        tau_S / tau_U / tau_K  : similarity / uncertainty / KL thresholds

    Returns:
        dict with keys: accepted, person_id, name, score, uncertainty, d_skl,
                        gate_U, gate_S, gate_K, all_scores
    """
    if not gallery:
        return {"accepted": False, "reason": "Gallery is empty"}

    # Gate 1: Uncertainty
    U_p = uncertainty_score(logvar_probe)
    gate_U = bool(U_p <= tau_U)

    # Stack gallery mus
    ids       = list(gallery.keys())
    mus       = torch.cat([gallery[i]["mu"] for i in ids], dim=0)   # (N, D)

    # Cosine similarity (Mode M3)
    p_norm = F.normalize(mu_probe, dim=1)
    g_norm = F.normalize(mus,      dim=1)
    scores = (p_norm @ g_norm.T).squeeze(0)                          # (N,)

    best_idx   = int(scores.argmax().item())
    best_id    = ids[best_idx]
    best_score = float(scores[best_idx].item())

    # Gate 2: Similarity
    gate_S = bool(best_score >= tau_S)

    # Gate 3: Symmetric KL to best gallery template
    mu_g  = gallery[best_id]["mu"]
    lv_g  = gallery[best_id]["logvar"]
    d_skl = symmetric_kl(mu_probe, logvar_probe, mu_g, lv_g)
    gate_K = bool(d_skl <= tau_K)

    accepted = gate_U and gate_S and gate_K

    return {
        "accepted":    accepted,
        "person_id":   best_id,
        "name":        gallery[best_id].get("name", best_id),
        "score":       best_score,
        "uncertainty": U_p,
        "d_skl":       d_skl,
        "gate_U":      gate_U,
        "gate_S":      gate_S,
        "gate_K":      gate_K,
        "all_scores":  {ids[i]: float(scores[i].item()) for i in range(len(ids))},
    }


# ==============================================================================
# G. GALLERY BUILDING
# ==============================================================================

def build_gallery(model, gallery_dir: str, selected_persons: list,
                  hand: str = "both", max_imgs: int = 20,
                  progress_cb=None) -> dict:
    """
    Embed images for selected persons and build a gallery dict.

    gallery[person_id] = {
        "mu":     Tensor (1, latent_dim),   # mean embedding
        "logvar": Tensor (1, latent_dim),   # mean log-variance
        "name":   str,
        "n_imgs": int,
    }

    The returned dict also contains a special key ``"_errors"`` (list of str)
    with any per-image error messages collected during embedding, so callers
    can surface them to the user instead of silently swallowing failures.

    Args:
        model           : loaded model from load_model()
        gallery_dir     : root dir containing person_X/ subdirs
        selected_persons: list of person_id strings (e.g. ['person_1', ...])
        hand            : 'left', 'right', or 'both'
        max_imgs        : max images per person to embed (speed vs accuracy)
        progress_cb     : optional callable(current, total, person_id) for progress
    """
    gallery = {}
    errors  = []  # collect per-image errors for caller to inspect
    hands   = ["left", "right"] if hand == "both" else [hand]
    total   = len(selected_persons)

    for idx, person_id in enumerate(selected_persons):
        if progress_cb:
            progress_cb(idx, total, person_id)

        person_dir = os.path.join(gallery_dir, person_id)
        img_paths  = []

        for h in hands:
            h_dir = os.path.join(person_dir, h)
            if os.path.isdir(h_dir):
                for fname in sorted(os.listdir(h_dir)):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_paths.append(os.path.join(h_dir, fname))

        if not img_paths:
            errors.append(f"{person_id}: no images found in '{person_dir}' for hand='{hand}'")
            continue

        mus, logvars = [], []
        for path in img_paths[:max_imgs]:
            try:
                img    = Image.open(path).convert("RGB")
                tensor = preprocess_pil(img)
                mu, lv = embed_image(model, tensor)
                mus.append(mu)
                logvars.append(lv)
            except Exception as exc:
                errors.append(f"{person_id} | {os.path.basename(path)}: {exc}")
                continue

        if not mus:
            errors.append(f"{person_id}: all {len(img_paths[:max_imgs])} images failed to embed")
            continue

        # Average embedding (template = mean of all enrolled images)
        mean_mu = torch.stack(mus,    dim=0).mean(dim=0)   # (1, D)
        mean_lv = torch.stack(logvars, dim=0).mean(dim=0)  # (1, D)

        gallery[person_id] = {
            "mu":     mean_mu,
            "logvar": mean_lv,
            "name":   person_id,
            "n_imgs": len(mus),
        }

    gallery["_errors"] = errors
    return gallery
