"""Home page – project overview and architecture, fully aligned with the research paper."""

import os
import streamlit as st
from PIL import Image

IMPLEMENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "implement-idea")
ARCH_IMG    = os.path.join(IMPLEMENT_DIR, "report", "Palm_Generative_Approach.drawio.png")
OVERALL_IMG = os.path.join(IMPLEMENT_DIR, "report", "overall.png")
DETAIL_IMG  = os.path.join(IMPLEMENT_DIR, "report", "detail-block.png")


def render():
    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-hero">
        <div class="page-title">🖐️ Probabilistic Palmprint Embedding</div>
        <div class="page-subtitle">
            with Generative Regularization for Open-Set Identification
        </div>
        <div style="margin-top:1rem; display:flex; gap:8px; flex-wrap:wrap;">
            <span class="badge badge-violet">Probabilistic Embedding</span>
            <span class="badge badge-cyan">Open-Set Biometrics</span>
            <span class="badge badge-emerald">Generative Regularization</span>
            <span class="badge badge-amber">Uncertainty Estimation</span>
            <span class="badge badge-pink">ArcFace + U-Net</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Abstract ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📄 Abstract</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Open-set palmprint identification must identify enrolled subjects from imperfect images and
    <b>reject impostors</b>. We propose modeling the Region of Interest (ROI) of each palmprint as a
    <b>diagonal Gaussian distribution</b> q(z|x) = N(μ, diag(σ²)):<br><br>
    &nbsp;• <b>Mean μ</b> — deterministic identity representation (Identity signal).<br>
    &nbsp;• <b>Variance σ²</b> — input-dependent estimate of aleatoric uncertainty (image quality, blur, ROI misalignment).<br><br>
    During training, an auxiliary <b>U-Net decoder</b> conditioned on stochastic samples z via
    <b>Feature-wise Linear Modulation (FiLM)</b> encourages the latent code to retain spatial
    palm-line structure. The decoder is <b>fully dropped at inference time</b>, making inference lightweight
    (Encoder + Projector only). The training loss combines angular-margin classification (ArcFace),
    covariance regularization (KL), image reconstruction, and heteroscedastic uncertainty calibration.
    </div>
    """, unsafe_allow_html=True)

    # ── Architecture diagram ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏗️ Model Architecture</div>', unsafe_allow_html=True)

    ic1, ic2 = st.columns([1, 1])
    with ic1:
        if os.path.exists(ARCH_IMG):
            st.image(Image.open(ARCH_IMG), caption="Probabilistic U-Net + PalmEncoder + MLP Projector", use_container_width=True)
        else:
            st.info(f"Architecture diagram not found at:\n`{ARCH_IMG}`")
    with ic2:
        st.markdown("""
        <div style="padding: 0.5rem 0;">
        <div style="font-weight:700; margin-bottom:0.8rem; color:var(--text-primary);">Training Pipeline</div>
        <div class="info-box" style="font-size:0.82rem; line-height:1.9;">
            <b>① Input Palm ROI</b> → <b>PalmEncoder</b> (CCNet / ResNet18 / PalmNet)<br>
            ↓ Produces <b>μ ∈ ℝᵈ</b> + <b>log σ² ∈ ℝᵈ</b><br>
            <b>② Reparameterization:</b> <code>z = μ + σ ⊙ ε,  ε ~ N(0,I)</code><br>
            <b>③ U-Net Decoder</b> (FiLM conditioned on z) → reconstructed palm image <code>x̂</code><br>
            <b>④ MLP Projector</b> maps μ → projected space → <b>ArcFace classification loss</b>
        </div>
        <div style="font-weight:700; margin: 0.8rem 0; color:var(--text-primary);">Inference Pipeline</div>
        <div class="info-box" style="font-size:0.82rem; line-height:1.9;">
            <b>Decoder is fully removed.</b> Only Encoder + Projector remain.<br>
            Input → Encoder → μ, σ → Matching / Rejection
        </div>
        </div>
        """, unsafe_allow_html=True)

    if os.path.exists(DETAIL_IMG):
        with st.expander("🔍 FiLM + UpBlock Detail"):
            st.image(Image.open(DETAIL_IMG), caption="FiLM (Feature-wise Linear Modulation) conditioning in the U-Net decoder", use_container_width=True)

    # ── Key contributions ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Key Contributions</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    contribs = [
        ("🎯", "Probabilistic Embedding",
         "Each palmprint → diagonal Gaussian N(μ, σ²). μ = identity signal. σ² = aleatoric uncertainty. Decouples representation, uncertainty, and generation.",
         ""),
        ("🔁", "Generative Regularization",
         "FiLM-conditioned U-Net decoder uses stochastic z to reconstruct the palm image, forcing the latent code to retain physical palm-line structure.",
         "warm"),
        ("⚡", "4 Inference Modes",
         "M0 (proj-mean) → M1 (proj-adapt) → M2 (latent-adapt+KL) → M3 (latent-mean) — M3 proves most effective experimentally.",
         ""),
        ("🚦", "Tri-threshold Open-Set",
         "Probe accepted only if: U_p ≤ τ_U (low uncertainty) AND S(p,g*) ≥ τ_S (high similarity) AND D_SKL(p,g*) ≤ τ_K (low KL distance).",
         "cool"),
    ]
    for col, (icon, title, desc, variant) in zip([c1, c2, c3, c4], contribs):
        col.markdown(f"""
        <div class="metric-card {variant}" style="height:100%;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">{icon}</div>
            <div style="font-weight:700; font-size:0.9rem; margin-bottom:0.5rem;">{title}</div>
            <div style="font-size:0.75rem; color:var(--text-muted); line-height:1.55;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Loss formulation ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚖️ Combined Loss Function</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="font-family:'JetBrains Mono',monospace; font-size:0.88rem; line-height:2;">
        <b>L<sub>total</sub></b> = λ<sub>arc</sub>·L<sub>ArcFace</sub>
                       + λ<sub>rec</sub>·L<sub>recon</sub>
                       + β·L<sub>KL</sub>
                       + λ<sub>unc</sub>·L<sub>unc</sub>
    </div>
    """, unsafe_allow_html=True)

    lc = st.columns(4)
    losses = [
        ("🎯", "L_ArcFace", "badge-violet",
         "Angular-margin classification on the MLP-projected space. ArcFace pushes different identities apart on a hypersphere. Primary identity learning objective."),
        ("🖼️", "L_recon", "badge-cyan",
         "MSE / L1 reconstruction loss from the FiLM-conditioned U-Net decoder. Acts as Generative Regularizer — forces latent code to encode spatial palm structure."),
        ("📐", "L_KL", "badge-amber",
         "KL(q(z|x) ‖ N(0,I)). Prevents latent-space collapse. Weight β is annealed gradually (KL Annealing) to avoid killing identity information early."),
        ("⚠️", "L_unc", "badge-pink",
         "Heteroscedastic uncertainty calibration. Encourages σ² to be higher for blurry / noisy / misaligned samples and lower for clean, in-distribution inputs."),
    ]
    for col, (icon, name, badge, desc) in zip(lc, losses):
        col.markdown(f"""
        <div class="arch-card" style="padding:1rem; height:100%;">
            <div style="font-size:1.6rem; margin-bottom:0.4rem;">{icon}</div>
            <span class="badge {badge}" style="margin-bottom:0.5rem; display:inline-block;">{name}</span>
            <div class="arch-desc" style="margin-top:0.4rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Training curriculum ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 Training Curriculum (contrastive_first)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        The training follows a <b>staged warm-up + annealing</b> curriculum to avoid gradient conflicts
        between the ArcFace (discriminative) and Reconstruction (generative) losses:
    </div>
    """, unsafe_allow_html=True)
    stages = [
        ("Stage 1\nStable Identity",
         "ArcFace only (no decoder). High LR. Encoder + Projector learn discriminative identity clustering. Contrastive-first warm-up.",
         "badge-violet"),
        ("Stage 2\nProbabilistic Head",
         "Probabilistic embedding activated: z = μ + σ·ε. Uncertainty loss L_unc activated. σ² starts carrying aleatoric uncertainty signal.",
         "badge-cyan"),
        ("Stage 3\nReconstruction Warm-up",
         "Decoder activated gradually (linear schedule, e.g. steps 2400→7200). λ_rec ramps up slowly so the generative branch doesn't overwhelm ArcFace.",
         "badge-amber"),
        ("Stage 4\nKL Annealing",
         "β (KL weight) increases from a small value (e.g. 0.001) up to a target (e.g. 0.05) to prevent posterior collapse while keeping identity information.",
         "badge-emerald"),
    ]
    sc = st.columns(4)
    for col, (title, desc, badge) in zip(sc, stages):
        col.markdown(f"""
        <div class="metric-card" style="padding:1rem; height:100%;">
            <span class="badge {badge}" style="margin-bottom:0.6rem; display:inline-block; white-space:pre-line;">{title}</span>
            <div style="font-size:0.75rem; color:var(--text-muted); line-height:1.55;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Inference modes ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚡ 4 Inference Modes</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        At inference the decoder is discarded. Four scoring procedures are evaluated for
        matching probe <b>p</b> against gallery template <b>g</b>:
    </div>
    """, unsafe_allow_html=True)
    modes = [
        ("M0", "badge-violet", "Projected-mean matching",
         "s = cos(proj(μₚ), proj(μ_g))",
         "Cosine similarity between the MLP-projected mean vectors. Baseline — no adaptation."),
        ("M1", "badge-cyan", "Projected-space adaptation",
         "Adapts residual δᵣ in projected space before scoring",
         "Finds a residual correction δᵣ in the projected space. Penalises deviation from the original probe projection."),
        ("M2", "badge-amber", "Latent-space adaptation",
         "Adapts δ_μ in latent space; penalises KL(q‖p)",
         "Residual δ_μ adapted in latent space. KL divergence from original probe distribution acts as regulariser to prevent drift."),
        ("M3", "badge-emerald", "Latent-mean matching ⭐ Best",
         "s = cos(μₚ, μ_g)",
         "Direct cosine similarity between raw latent means. Empirically most effective — highest Rank-1 & lowest EER across all settings."),
    ]
    mc = st.columns(4)
    for col, (mode, badge, name, formula, desc) in zip(mc, modes):
        col.markdown(f"""
        <div class="metric-card" style="padding:1.1rem; height:100%;">
            <span class="badge {badge}">{mode}</span>
            <div style="font-weight:700; font-size:0.85rem; margin:0.5rem 0 0.3rem;">{name}</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#a78bfa; margin-bottom:0.4rem; line-height:1.4;">{formula}</div>
            <div style="font-size:0.73rem; color:var(--text-muted); line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Open-set rejection rule ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🚦 Open-Set Rejection Rule</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        A probe <b>p</b> is accepted as identity <b>g*</b> only when <b>all three</b> of the following
        conditions hold simultaneously. Thresholds are calibrated jointly on the validation split —
        never tuned on the test set.
    </div>
    """, unsafe_allow_html=True)
    rc = st.columns(3)
    conditions = [
        ("⚠️", "Condition 1 – Uncertainty Gate",
         "U_p ≤ τ_U",
         "Probe uncertainty must be low. If σ² is large the image is likely blurry, misaligned, or out-of-distribution → reject.",
         ""),
        ("🎯", "Condition 2 – Similarity Gate",
         "S(p, g*) ≥ τ_S",
         "Cosine similarity to the best gallery match must exceed the acceptance threshold τ_S.",
         "warm"),
        ("📐", "Condition 3 – Distribution Gate",
         "D_SKL(p, g*) ≤ τ_K",
         "Symmetric KL divergence D_SKL(N(μₚ,σₚ²) ‖ N(μ_g,σ_g²)) must be small — distributions must be compatible.",
         "cool"),
    ]
    for col, (icon, title, formula, desc, variant) in zip(rc, conditions):
        col.markdown(f"""
        <div class="metric-card {variant}" style="padding:1.2rem;">
            <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
            <div style="font-weight:700; margin-bottom:0.3rem;">{title}</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:700; color:var(--text-primary); margin-bottom:0.5rem;">{formula}</div>
            <div style="font-size:0.75rem; color:var(--text-muted); line-height:1.55;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Backbone comparison ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔧 Backbone Encoders</div>', unsafe_allow_html=True)
    bc = st.columns(3)
    backbones = [
        ("ResNet18", "badge-violet", "unet_resnet",
         "🏆 Best accuracy",
         [
             ("Pretrained ImageNet weights", True),
             ("Multi-scale spatial features", True),
             ("Strong noise / blur robustness", True),
             ("Heavier inference (kept lightweight after dropping decoder)", False),
         ],
         "Standard workhorse. ResNet18 backbone + U-Net decoder forms the proposed model. High Rank-1 on all datasets."),
        ("CCNet", "badge-cyan", "unet_ccnet",
         "🔬 Experimental",
         [
             ("Criss-Cross Attention for global context", True),
             ("Captures full-hand structure", True),
             ("High computational cost (attention matrix)", False),
             ("Palmprint is local-texture heavy → marginal gain", False),
         ],
         "CCNet captures long-range dependencies via criss-cross attention. Interesting academically; computationally expensive."),
        ("PalmNet / PalmNet-Gabor", "badge-pink", "unet_palmnet",
         "🚀 Lightweight",
         [
             ("Gabor-initialized Conv filters (texture prior)", True),
             ("Ultra-lightweight (few thousand params)", True),
             ("Real-time on embedded devices", True),
             ("Requires very accurate ROI alignment", False),
         ],
         "Gabor-initialized shallow CNN. Fast inference for deployment on edge devices. Sensitive to ROI crop quality."),
    ]
    for col, (name, badge, model_id, status, pros_cons, summary) in zip(bc, backbones):
        bullets = "".join([
            f"<div style='font-size:0.73rem; color:{\"#6ee7b7\" if ok else \"#f9a8d4\"}; margin:2px 0;'>"
            f"{'✅' if ok else '⚠️'} {text}</div>"
            for text, ok in pros_cons
        ])
        col.markdown(f"""
        <div class="metric-card" style="padding:1.2rem; height:100%;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                <span class="badge {badge}">{name}</span>
                <span style="font-size:0.72rem; color:var(--text-muted);">{status}</span>
            </div>
            <div style="font-size:0.73rem; color:var(--text-muted); margin-bottom:0.7rem; line-height:1.5;">{summary}</div>
            {bullets}
        </div>
        """, unsafe_allow_html=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💽 Datasets & Evaluation Protocols</div>', unsafe_allow_html=True)
    dc = st.columns(3)
    datasets = [
        ("🗂️", "Own Dataset", ["CCNet", "ResNet18", "PalmNet"],
         "Collected in-house. Train/val/test split by identity. Includes enrolled-test (known) and unknown-test (stranger) subsets for open-set evaluation."),
        ("🗂️", "Tongji", ["CCNet", "ResNet18"],
         "Multi-session contactless palmprint DB. Cross-session evaluation (train session1 → test session2) measures robustness to data drift."),
        ("🗂️", "IITD", ["CCNet", "ResNet18"],
         "IIT Delhi contactless palmprint DB. Left/right hand splits test cross-domain generalization (train left → test right hand)."),
    ]
    for col, (icon, name, models, desc) in zip(dc, datasets):
        badges = " ".join([f'<span class="badge badge-violet">{m}</span>' for m in models])
        col.markdown(f"""
        <div class="metric-card" style="padding:1.2rem;">
            <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
            <div style="font-weight:700; margin-bottom:0.4rem;">{name}</div>
            <div style="margin-bottom:0.5rem;">{badges}</div>
            <div style="font-size:0.75rem; color:var(--text-muted); line-height:1.55;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Key design decisions from paper ──────────────────────────────────────
    with st.expander("📑 Key Design Decisions from the Paper"):
        st.markdown("""
        | Decision | Rationale |
        |----------|-----------|
        | **Decoder as Regularizer only** | Reconstruction loss preserves physical ridge/line structure in latent code. Dropped at inference → no deployment cost. |
        | **Separate latent_dim ≠ proj_dim** | Prevents gradient conflict between ArcFace (identity) and Reconstruction (structure) losses. |
        | **FiLM conditioning on z (not μ)** | Stochastic samples encourage the decoder to learn a distribution-aware reconstruction, not just reconstruct from the mean. |
        | **KL Annealing** | β starts small to avoid posterior collapse while the encoder is still learning identity features early in training. |
        | **Validation-calibrated thresholds** | τ_U, τ_S, τ_K are selected jointly on the validation set — never tuned on the held-out test set to avoid data leakage. |
        | **M3 (cos(μₚ, μ_g)) as best mode** | Projection head + ArcFace may introduce non-linearity that hurts raw matching; direct latent-mean cosine is most stable. |
        | **contrastive_first curriculum** | ArcFace warms up the encoder first; decoder then refines the latent structure rather than fighting a randomly initialised encoder. |
        """)
