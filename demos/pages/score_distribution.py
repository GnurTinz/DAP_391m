"""Score & uncertainty distribution page."""

import os
import glob
import streamlit as st
from PIL import Image

LOGS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "logs")

MODE_DIRS = {
    "mode0_baseline_real": "M0 – Projected-mean",
    "mode1_opt_proj_real": "M1 – Projected Adaptation",
    "mode2_opt_latent_real": "M2 – Latent Adaptation",
    "mode3_baseline_mu_real": "M3 – Latent-mean ⭐",
    "mode4_opt_latent_mu_real": "M4 – Latent-mu Adaptation",
}

DIST_IMAGES = {
    "score_distribution_openset.png":   ("🎯", "Score Distribution", "Genuine vs. Impostor / Stranger cosine similarity"),
    "sigma_distribution_openset.png":   ("📐", "Sigma Distribution", "Predicted variance (σ) across probe/gallery/stranger"),
    "uncertainty_distribution_openset.png": ("⚠️", "Uncertainty Distribution", "Aleatoric uncertainty U_p across identity groups"),
    "tsne_mu.png":  ("🔵", "t-SNE: Latent μ",  "t-SNE on latent means coloured by identity"),
    "tsne_proj.png":("🟣", "t-SNE: Projected", "t-SNE on projected embeddings after ArcFace"),
}


def _find_eval_runs():
    """Find all eval/mode* directories in logs."""
    results = {}
    for root, dirs, files in os.walk(LOGS_ROOT):
        bn = os.path.basename(root)
        if bn in MODE_DIRS:
            # Check for distribution images
            has_imgs = any(
                os.path.exists(os.path.join(root, img))
                for img in DIST_IMAGES.keys()
            )
            if has_imgs:
                # Parent version label
                rel = os.path.relpath(root, LOGS_ROOT)
                results[rel] = {
                    "path": root,
                    "mode": MODE_DIRS.get(bn, bn),
                    "bn": bn,
                }
    return results


def render():
    st.markdown("""
    <div class="page-hero">
        <div class="page-title">📈 Score & Distribution Analysis</div>
        <div class="page-subtitle">
            Visualize similarity score distributions, uncertainty estimates,
            and σ² statistics for genuine vs. impostor vs. stranger probes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    eval_runs = _find_eval_runs()

    if not eval_runs:
        st.warning("No evaluation distribution plots found. Run evaluation first.")
        st.code("python tools/eval_attendance.py checkpoint=... dataset=... +eval=eval", language="bash")
        return

    # ── Run selector ──────────────────────────────────────────────────────────
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        run_options = {k: f"{k.split(os.sep)[-3]}  |  {v['mode']}" for k, v in eval_runs.items()}
        selected = st.selectbox(
            "Evaluation run",
            list(run_options.keys()),
            format_func=lambda k: run_options[k],
            key="sd_run",
        )

    run_info = eval_runs[selected]
    run_path = run_info["path"]
    mode_name = run_info["mode"]

    st.markdown(f"""
    <div class="info-box">
        📁 <b>Path:</b> <code>{os.path.relpath(run_path, os.path.join(LOGS_ROOT, '..')).replace(os.sep, '/')}</code><br>
        ⚡ <b>Mode:</b> {mode_name}
    </div>
    """, unsafe_allow_html=True)

    # ── Load available images ─────────────────────────────────────────────────
    available = {}
    for fname, (icon, title, desc) in DIST_IMAGES.items():
        fpath = os.path.join(run_path, fname)
        if os.path.exists(fpath):
            available[fname] = (icon, title, desc, Image.open(fpath))

    if not available:
        st.info("No distribution images found in this run directory.")
        return

    # ── Distribution plots ────────────────────────────────────────────────────
    dist_keys = [k for k in available if "distribution" in k or "sigma" in k or "uncertainty" in k]
    tsne_keys = [k for k in available if k.startswith("tsne_")]

    tab_dist, tab_tsne, tab_compare = st.tabs([
        "📊 Distributions",
        "🌀 t-SNE",
        "🔀 Mode Comparison",
    ])

    # ── Distributions ─────────────────────────────────────────────────────────
    with tab_dist:
        if not dist_keys:
            st.info("No distribution images available.")
        else:
            for i, fname in enumerate(dist_keys):
                icon, title, desc, img = available[fname]
                with st.expander(f"{icon} {title}", expanded=(i == 0)):
                    dc1, dc2 = st.columns([3, 1])
                    with dc2:
                        st.markdown(f"""
                        <div class="info-box" style="margin-top:1rem;">
                            <b>{title}</b><br><br>
                            {desc}
                        </div>
                        """, unsafe_allow_html=True)
                    with dc1:
                        st.image(img, caption=f"{title} – {mode_name}", use_container_width=True)

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    with tab_tsne:
        if not tsne_keys:
            st.info("No t-SNE images available.")
        else:
            tc = st.columns(len(tsne_keys))
            for col, fname in zip(tc, tsne_keys):
                icon, title, desc, img = available[fname]
                col.markdown(f"""
                <div style="text-align:center; margin-bottom:0.5rem;">
                    <span class="badge badge-violet">{title}</span>
                </div>
                """, unsafe_allow_html=True)
                col.image(img, caption=desc, use_container_width=True)

            with st.expander("💡 Interpretation guide"):
                st.markdown("""
                **What to look for:**
                - **Well-separated clusters** → the model has learned discriminative features
                - **Tight intra-class clusters** → consistent embeddings for the same identity
                - **Large margins between classes** → good open-set rejection potential
                - **Comparison μ vs proj** → ArcFace projection should produce sharper margins

                **Color encoding:** Each color represents a unique identity.
                Stranger (unknown) probes are typically plotted in a different color/marker.
                """)

    # ── Cross-mode comparison ─────────────────────────────────────────────────
    with tab_compare:
        st.markdown("""
        <div class="info-box">
            Compare the same distribution plot across different inference modes.
            Select the image type and the runs to compare.
        </div>
        """, unsafe_allow_html=True)

        # Find all runs for the same parent version
        parent_version = os.path.dirname(run_path)
        sibling_modes = {}
        if os.path.exists(parent_version):
            for d in os.listdir(parent_version):
                if d in MODE_DIRS:
                    sibling_modes[d] = os.path.join(parent_version, d)

        if len(sibling_modes) < 2:
            st.info("Multiple mode directories not found under the same version. Browse to a checkpoint with multiple evaluated modes.")
        else:
            img_type_opts = [f for f in DIST_IMAGES.keys() if any(
                os.path.exists(os.path.join(p, f)) for p in sibling_modes.values()
            )]
            if not img_type_opts:
                st.info("No common images found.")
            else:
                sel_img_type = st.selectbox(
                    "Image type",
                    img_type_opts,
                    format_func=lambda k: DIST_IMAGES[k][1],
                    key="cmp_img_type",
                )
                cmp_modes = [m for m in sibling_modes if os.path.exists(os.path.join(sibling_modes[m], sel_img_type))]

                cols = st.columns(min(len(cmp_modes), 3))
                for col, mode_bn in zip(cols, cmp_modes):
                    fpath = os.path.join(sibling_modes[mode_bn], sel_img_type)
                    img = Image.open(fpath)
                    label = MODE_DIRS.get(mode_bn, mode_bn)
                    col.image(img, caption=label, use_container_width=True)

    # ── Key findings ──────────────────────────────────────────────────────────
    with st.expander("📋 Key Findings from Score Analysis"):
        st.markdown("""
        Based on the distribution analysis across experiments:

        | Observation | Implication |
        |-------------|-------------|
        | **Genuine scores ≫ Impostor/Stranger** | Good identity separation at the score level |
        | **Low σ² for enrolled probes** | Model is confident on in-distribution data |
        | **High σ² for strangers** | Uncertainty is a useful rejection signal |
        | **M3 produces tightest genuine peak** | Latent-mean matching has best intra-class consistency |
        | **Symmetric KL as rejection signal** | Distribution distance complements cosine similarity |
        """)
