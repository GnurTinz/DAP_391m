"""
src/engine/represent.py
=======================
Các hàm hỗ trợ tối ưu biểu diễn (representation optimization) tại inference.

Mô-đun này cung cấp ba nhóm chức năng:

    A. Negative Sampling
       - decoder_loop_negatives   : sinh mẫu âm bản qua vòng lặp decoder → re-encode
       - hard_negative_mining     : chọn hard negative từ gallery

    B. Template Optimization
       - optimize_r_in_projected_space : tối ưu r trong Projected Space (đầu ra MLP)
       - optimize_r_from_latent        : tối ưu r trong Latent Space (mu)

    C. High-level Entry Point
       - optimize_representation  : nhận ảnh đầu vào, trả về r tối ưu
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


# ==============================================================================
# A. NEGATIVE SAMPLING
# ==============================================================================

def decoder_loop_negatives(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    model,
    x_ref: torch.Tensor,
    num_samples: int = 64,
    temperature_range: tuple = (2.0, 5.0),
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sinh mẫu âm bản bằng cách khai thác decoder của mô hình generative.

    Pipeline:
        z_prt = mu_q + T * sigma_q * eps     (T ~ Uniform[T_min, T_max])
        x_gen = Decoder(z_prt, x_ref)        ← ảnh palmprint mới trên data manifold
        mu_neg = Encoder(x_gen)              ← re-encode để lấy biểu diễn mới

    Lý do hiệu quả:
        - Decoder đã được train → x_gen luôn nằm trên data manifold (không phải noise).
        - Re-encode trả về mu_neg trong latent space thực, không "trôi nổi" xa.
        - T lớn → z rời xa mu_q → mu_neg có xu hướng biểu diễn "identity khác".

    Args:
        mu_q             : (1, latent_dim) — latent mean của query.
        logvar_q         : (1, latent_dim) — latent log-variance của query.
        model            : UNetPalmModel hoặc ProbabilisticPalmModel có decoder.
        x_ref            : (1, C, H, W)   — ảnh gốc (dùng làm skip-connection cho UNet).
                           Với ProbabilisticPalmModel (VAE thuần), x_ref không dùng.
        num_samples      : số mẫu âm bản cần sinh.
        temperature_range: (T_min, T_max) — khoảng nhiệt độ perturbation.
        device           : 'cpu' hoặc 'cuda'.

    Returns:
        mu_neg : (num_samples, latent_dim) — mu vectors âm bản trên data manifold.

    Raises:
        RuntimeError nếu model không có decoder (use_decoder=False).
    """
    model.eval()

    has_unet_dec = hasattr(model, "decode_from_z_and_x") and getattr(model, "use_decoder", False)
    has_vae_dec  = hasattr(model, "decoder") and getattr(model, "use_decoder", False)

    if not (has_unet_dec or has_vae_dec):
        raise RuntimeError(
            "decoder_loop_negatives yeu cau model co decoder (use_decoder=True). "
            "Kiem tra config YAML hoac dung neg_strategy='spherical' thay the."
        )

    T_min, T_max = temperature_range
    sigma_q    = torch.exp(0.5 * logvar_q)   # (1, latent_dim)
    latent_dim = mu_q.size(1)
    neg_list   = []

    with torch.no_grad():
        for _ in range(num_samples):
            T   = T_min + (T_max - T_min) * torch.rand(1, device=device).item()
            eps = torch.randn(1, latent_dim, device=device)
            z   = mu_q + T * sigma_q * eps         # (1, latent_dim)

            if has_unet_dec:
                x_gen = model.decode_from_z_and_x(z, x_ref)
            else:
                x_gen = model.decoder(z)

            out = model(x_gen, decode=False)
            neg_list.append(out["mu"])              # (1, latent_dim)

    return torch.cat(neg_list, dim=0)               # (num_samples, latent_dim)


def hard_negative_mining(
    query_proj: torch.Tensor,
    all_proj: torch.Tensor,
    all_labels: list,
    query_label: int,
    top_k: int = 64,
) -> torch.Tensor:
    """
    Chọn hard negative từ pool: identity khác nhưng gần nhất trong projected space.

    Hard negative mining là kỹ thuật chuẩn trong metric learning SOTA
    (ArcFace, AdaFace, MagFace). Mẫu gần nhưng khác label buộc model học
    ranh giới phân biệt chắc chắn hơn mẫu random.

    Args:
        query_proj  : (1, feat_dim) — projected vector của query.
        all_proj    : (N, feat_dim) — toàn bộ projected vectors trong pool.
        all_labels  : list[int] len=N — identity labels tương ứng.
        query_label : int — identity của query (bị loại ra khỏi pool).
        top_k       : số hard negative muốn lấy.

    Returns:
        hard_neg_proj : (k, feat_dim) — k <= top_k hard negative vectors.
    """
    labels_t   = torch.tensor(all_labels, device=all_proj.device)
    other_mask = (labels_t != query_label)
    other_proj = all_proj[other_mask]                          # (M, feat_dim)

    q_norm    = F.normalize(query_proj, p=2, dim=1)           # (1, feat_dim)
    o_norm    = F.normalize(other_proj, p=2, dim=1)           # (M, feat_dim)
    sims      = torch.mm(q_norm, o_norm.t()).squeeze(0)        # (M,)

    k = min(top_k, sims.size(0))
    _, top_idx = sims.topk(k, largest=True)
    return other_proj[top_idx]                                 # (k, feat_dim)


# ==============================================================================
# PRIVATE HELPER: SAMPLE NEGATIVES (dùng chung cho cả 2 hàm tối ưu)
# ==============================================================================

def _sample_negatives(
    strategy: str,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    model,
    mu_others: torch.Tensor,
    logvar_others: torch.Tensor,
    num_samples: int,
    neg_temp: float,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Sinh num_samples âm bản trong latent space theo strategy được chọn.

    Args:
        strategy    : 'real' | 'spherical' | 'decoder_loop'
        mu_q        : (1, latent_dim) — mean query
        logvar_q    : (1, latent_dim) — logvar query
        model       : model có projector (và decoder nếu dùng decoder_loop)
        mu_others   : (M, latent_dim) | None — mu của identities khác
        logvar_others: (M, latent_dim) | None
        num_samples : số mẫu cần sinh
        neg_temp    : nhiệt độ nhiễu (scaling sigma)
        config      : dict config (dùng decoder_T_min, decoder_T_max, x_ref)
        device      : torch.device

    Returns:
        z_neg : (num_samples, latent_dim) — negative latent vectors
    """
    latent_dim = mu_q.size(1)
    sigma_q    = torch.exp(0.5 * logvar_q)                    # (1, latent_dim)
    repr_cfg   = config.get("represent", {})

    # ── Strategy: dùng mu của người khác (Real negatives) ─────────────────────
    if strategy == "real" and mu_others is not None and mu_others.size(0) > 0:
        idx          = torch.randint(0, mu_others.size(0), (num_samples,), device=device)
        mu_n         = mu_others[idx]
        sigma_n      = torch.exp(0.5 * logvar_others[idx]) if logvar_others is not None else torch.ones_like(mu_n)
        eps          = torch.randn(num_samples, latent_dim, device=device)
        return mu_n + neg_temp * sigma_n * eps

    # ── Strategy: Decoder-loop (sinh ảnh → re-encode) ─────────────────────────
    elif strategy == "decoder_loop":
        x_ref  = repr_cfg.get("x_ref", None)
        t_min  = repr_cfg.get("decoder_T_min", 2.0)
        t_max  = repr_cfg.get("decoder_T_max", 5.0)
        if x_ref is None and hasattr(model, "decode_from_z_and_x"):
            raise ValueError(
                "neg_strategy='decoder_loop' voi UNetPalmModel yeu cau "
                "config['represent']['x_ref'] = anh goc tensor (1,C,H,W).to(device)"
            )
        return decoder_loop_negatives(
            mu_q, logvar_q, model,
            x_ref=x_ref if x_ref is not None else mu_q,
            num_samples=num_samples,
            temperature_range=(t_min, t_max),
            device=device,
        )

    # ── Strategy: Spherical rotation (xoay vector mu ngẫu nhiên 45°–180°) ─────
    else:
        # Fallback về spherical cho cả 'spherical' và 'real' không có mu_others
        mu_norm     = torch.norm(mu_q, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        mu_unit     = mu_q / mu_norm
        # Góc xoay theta ∈ [π/4, π]
        theta       = (math.pi / 4) + (math.pi * 3 / 4) * torch.rand(num_samples, 1, device=device)
        # Tìm hướng trực giao ngẫu nhiên (batched)
        v           = torch.randn(num_samples, latent_dim, device=device)
        v_proj      = (v * mu_unit).sum(dim=1, keepdim=True) * mu_unit
        v_ortho     = v - v_proj
        v_ortho     = v_ortho / v_ortho.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        # Xoay trên mặt phẳng 2D
        z_rot       = mu_norm * (mu_unit * torch.cos(theta) + v_ortho * torch.sin(theta))
        eps         = torch.randn(num_samples, latent_dim, device=device)
        return z_rot + neg_temp * sigma_q * eps


# ==============================================================================
# B. TEMPLATE OPTIMIZATION
# ==============================================================================

def optimize_r_in_projected_space(
    mu_c: torch.Tensor,
    logvar_c: torch.Tensor,
    mu_others: torch.Tensor,
    logvar_others: torch.Tensor,
    model,
    device: torch.device,
    config: dict = None,
    num_samples: int = 256,
    max_steps: int = 200,
    lr: float = 0.01,
    loss_type: str = "bce",
    verbose: bool = False,
) -> torch.Tensor:
    """
    Tối ưu vector đại diện r TRỰC TIẾP trong Projected Space.

    r được tìm trên hypersphere sao cho:
      - Gần với các mẫu Positive (samples từ mu_c)
      - Xa với các mẫu Negative (từ mu_others hoặc spherical/decoder_loop)

    Args:
        mu_c, logvar_c   : (1, latent_dim) — mean và logvar của identity cần đăng ký.
        mu_others        : (M, latent_dim) — mu của các identity khác (âm bản thật).
        logvar_others    : (M, latent_dim) — logvar tương ứng.
        model            : model có attribute `projector`.
        device           : torch.device.
        config           : dict, dùng khóa `represent` để lấy các tham số.
        num_samples      : số mẫu pos/neg dùng cho tối ưu.
        max_steps        : số bước Adam tối đa.
        lr               : learning rate.
        loss_type        : 'bce' | 'triplet'.
        verbose          : in progress bar hay không.

    Returns:
        r_final : (proj_dim,) — vector đại diện tối ưu, L2-normalized.
    """
    if config is None:
        config = {}
    repr_cfg = config.get("represent", {})

    pos_temp     = repr_cfg.get("pos_temperature", 0.5)
    neg_temp     = repr_cfg.get("neg_temperature", 1.0)
    neg_strategy = repr_cfg.get("neg_strategy", "real")

    # Đảm bảo shape (1, latent_dim)
    if mu_c.dim() == 1:
        mu_c     = mu_c.unsqueeze(0)
        logvar_c = logvar_c.unsqueeze(0)

    mu_c     = mu_c.to(device)
    logvar_c = logvar_c.to(device)
    if mu_others is not None:
        mu_others     = mu_others.to(device)
        logvar_others = logvar_others.to(device) if logvar_others is not None else None

    latent_dim = mu_c.size(1)
    sigma_c    = torch.exp(0.5 * logvar_c)

    model.eval()

    # ── Sinh mẫu Positive (reparameterize quanh mu_c) ─────────────────────────
    eps_pos = torch.randn(num_samples, latent_dim, device=device)
    z_pos   = mu_c + pos_temp * sigma_c * eps_pos             # (N, latent_dim)

    # ── Sinh mẫu Negative ─────────────────────────────────────────────────────
    z_neg = _sample_negatives(
        neg_strategy, mu_c, logvar_c, model,
        mu_others, logvar_others,
        num_samples, neg_temp, config, device,
    )                                                          # (N, latent_dim)

    # ── Project sang Projected Space ──────────────────────────────────────────
    with torch.no_grad():
        v_pos = F.normalize(model.projector(z_pos), p=2, dim=1)  # (N, proj_dim)
        v_neg = F.normalize(model.projector(z_neg), p=2, dim=1)  # (N, proj_dim)

    # ── Tối ưu r trên hypersphere ─────────────────────────────────────────────
    r         = nn.Parameter(v_pos.mean(dim=0, keepdim=True).detach())  # init = centroid pos
    optimizer = optim.Adam([r], lr=lr)
    y_pos     = torch.ones(num_samples, 1, device=device)
    y_neg     = torch.zeros(num_samples, 1, device=device)

    pbar = tqdm(range(max_steps), leave=False, desc="Opt(proj)") if verbose else range(max_steps)
    for step in pbar:
        optimizer.zero_grad()
        r_norm   = F.normalize(r, p=2, dim=1)                 # clamp lên hypersphere
        sim_pos  = torch.mm(v_pos, r_norm.t())                # (N, 1)
        sim_neg  = torch.mm(v_neg, r_norm.t())                # (N, 1)

        if loss_type == "triplet":
            loss = torch.clamp(sim_neg - sim_pos + 0.3, min=0.0).mean()
        else:  # bce (default)
            logits  = torch.cat([sim_pos, sim_neg], dim=0) * 10.0
            targets = torch.cat([y_pos, y_neg], dim=0)
            loss    = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

        loss.backward()
        optimizer.step()

        if verbose:
            acc = (sim_pos > sim_neg).float().mean().item() * 100
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.1f}%"})

    return F.normalize(r.detach(), p=2, dim=1).squeeze(0)     # (proj_dim,)


def optimize_r_from_latent(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_others: torch.Tensor,
    logvar_others: torch.Tensor,
    model,
    device: torch.device,
    config: dict = None,
    num_samples: int = 512,
    max_steps: int = 200,
    lr: float = 0.01,
    bce_threshold: float = 0.05,
    freeze_net: bool = False,
    verbose: bool = False,
    pbar=None,
):
    """
    Tối ưu vector bổ sung r TRONG Latent Space: r được cộng trực tiếp vào mu.
    Template cuối = Projected(mu_q + r).

    Mô hình: T = Proj(mu_q + r).  Tối ưu BCE giữa T·pos và T·neg.

    Args:
        mu_q, logvar_q   : (1, latent_dim) — query distribution.
        mu_others        : (M, latent_dim) — mu của identities khác.
        logvar_others    : (M, latent_dim).
        model            : model có `projector`.
        device           : torch.device.
        config           : dict với khóa 'represent'.
        num_samples      : số mẫu pos/neg.
        max_steps        : số bước Adam tối đa.
        lr               : learning rate.
        bce_threshold    : dừng sớm nếu loss <= threshold.
        freeze_net       : không dùng (kept for backward-compat).
        verbose          : in progress.
        pbar             : external tqdm bar (nếu có).

    Returns:
        r_final  : (1, latent_dim) — perturbation tối ưu trong latent space.
        z_pos    : (N, latent_dim) — positive samples đã dùng.
        z_neg    : (N, latent_dim) — negative samples đã dùng.
    """
    if config is None:
        config = {}
    repr_cfg = config.get("represent", {})

    pos_temp     = repr_cfg.get("pos_temperature", 1.0)
    neg_temp     = repr_cfg.get("neg_temperature", 1.0)
    neg_strategy = repr_cfg.get("neg_strategy", "real")

    mu_q     = mu_q.to(device)
    logvar_q = logvar_q.to(device)
    if mu_others is not None:
        mu_others     = mu_others.to(device)
        logvar_others = logvar_others.to(device) if logvar_others is not None else None

    latent_dim = mu_q.size(1)
    sigma_q    = torch.exp(0.5 * logvar_q)

    # ── Sinh mẫu Positive ─────────────────────────────────────────────────────
    eps_pos = torch.randn(num_samples, latent_dim, device=device)
    z_pos   = mu_q + pos_temp * sigma_q * eps_pos             # (N, latent_dim)

    # ── Sinh mẫu Negative ─────────────────────────────────────────────────────
    z_neg = _sample_negatives(
        neg_strategy, mu_q, logvar_q, model,
        mu_others, logvar_others,
        num_samples, neg_temp, config, device,
    )                                                          # (N, latent_dim)

    # ── Project pos/neg (cố định, không đổi trong vòng lặp) ──────────────────
    model.eval()
    with torch.no_grad():
        p_pos = F.normalize(model.projector(z_pos), p=2, dim=1)   # (N, proj_dim)
        p_neg = F.normalize(model.projector(z_neg), p=2, dim=1)   # (N, proj_dim)

    y_pos = torch.ones(num_samples, 1, device=device)
    y_neg = torch.zeros(num_samples, 1, device=device)

    # ── Khởi tạo r = 0 (cộng vào mu) ─────────────────────────────────────────
    r         = nn.Parameter(torch.zeros_like(mu_q))
    optimizer = optim.Adam([r], lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    if verbose and pbar is None:
        print(f"Optimizing r in latent space | steps={max_steps} | neg={neg_strategy}")

    _iter = range(max_steps)
    for step in _iter:
        optimizer.zero_grad()

        # Template: Projected(mu_q + r)
        T       = F.normalize(model.projector(mu_q + r), p=2, dim=1)  # (1, proj_dim)
        sim_pos = torch.mm(p_pos, T.t())   # (N, 1)
        sim_neg = torch.mm(p_neg, T.t())   # (N, 1)

        logits  = torch.cat([sim_pos * 10.0, sim_neg * 10.0], dim=0)
        targets = torch.cat([y_pos, y_neg], dim=0)
        loss    = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        if pbar is not None:
            pbar.set_postfix({"bce": f"{loss.item():.4f}"})

        if loss.item() <= bce_threshold:
            if verbose:
                print(f"  Early stop at step {step+1} (loss={loss.item():.4f})")
            break

    return r.detach(), z_pos, z_neg


# ==============================================================================
# C. HIGH-LEVEL ENTRY POINT
# ==============================================================================

def optimize_representation(
    model,
    image: torch.Tensor,
    config: dict,
    device: torch.device,
    mu_others: torch.Tensor = None,
    logvar_others: torch.Tensor = None,
    num_samples: int = 512,
    max_steps: int = 200,
    lr: float = 0.01,
    bce_threshold: float = 0.05,
    pbar=None,
):
    """
    High-level entry point: nhận ảnh đầu vào, trả về r tối ưu trong Latent Space.

    Hỗ trợ config:
        represent.mode       : 'single' | 'average'
        represent.re_encode  : True/False — dùng x_hat để re-encode mu trước khi tối ưu

    Args:
        model        : model generative (UNetPalmModel hoặc ProbabilisticPalmModel).
        image        : (B, C, H, W) — ảnh đầu vào (B >= 2 nếu mode='average').
        config       : dict Hydra config.
        device       : torch.device.
        mu_others    : (M, latent_dim) — tùy chọn, mu của identities khác.
        logvar_others: (M, latent_dim).
        num_samples  : số mẫu tối ưu.
        max_steps    : số bước Adam.
        lr           : learning rate.
        bce_threshold: dừng sớm.
        pbar         : external tqdm bar.

    Returns:
        r       : (1, latent_dim) — perturbation tối ưu.
        z_pos   : (N, latent_dim) — positive samples.
        z_neg   : (N, latent_dim) — negative samples.
    """
    model.eval()
    repr_cfg  = config.get("represent", {})
    mode      = repr_cfg.get("mode", "single")
    re_encode = repr_cfg.get("re_encode", False)

    with torch.no_grad():
        if mode == "average" and image.size(0) >= 2:
            out1     = model(image[0:1], decode=True)
            out2     = model(image[1:2], decode=True)
            target_mu     = (out1["mu"]     + out2["mu"])     / 2.0
            target_logvar = (out1["logvar"] + out2["logvar"]) / 2.0
            # Sinh ảnh đại diện từ mu_avg (dùng cho re_encode hoặc decoder_loop)
            if hasattr(model, "decode_from_z_and_x"):
                x_ref = (image[0:1] + image[1:2]) / 2.0
                x_hat = model.decode_from_z_and_x(target_mu, x_ref)
            else:
                x_hat = model.decoder(target_mu) if getattr(model, "use_decoder", False) else None
        else:
            out           = model(image[0:1], decode=True)
            target_mu     = out["mu"]
            target_logvar = out["logvar"]
            x_hat         = out.get("x_hat", None)
            x_ref         = image[0:1]

        # Re-encode từ ảnh khôi phục (để lấy mu ổn định hơn)
        if re_encode and x_hat is not None:
            out_re        = model(x_hat, decode=False)
            target_mu     = out_re["mu"].detach()
            target_logvar = out_re["logvar"].detach()
        else:
            target_mu     = target_mu.detach()
            target_logvar = target_logvar.detach()

    # Ghi x_ref vào config để decoder_loop_negatives dùng nếu cần
    if "represent" not in config:
        config["represent"] = {}
    config["represent"]["x_ref"] = x_ref.to(device) if x_ref is not None else None

    return optimize_r_from_latent(
        target_mu, target_logvar,
        mu_others, logvar_others,
        model, device,
        config=config,
        num_samples=num_samples,
        max_steps=max_steps,
        lr=lr,
        bce_threshold=bce_threshold,
        pbar=pbar,
        verbose=(pbar is None),
    )
