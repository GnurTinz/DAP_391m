"""Latent Space visualization page – PCA, t-SNE, distributions."""

import os
import glob
import streamlit as st
from PIL import Image

# ── Path helpers ──────────────────────────────────────────────────────────────
LOGS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "logs")


def _find_pca_dirs():
    """Scan logs/** for pca_latent directories."""
    results = {}
    for root, dirs, files in os.walk(LOGS_ROOT):
        if "pca_latent" in dirs:
            pca_dir = os.path.join(root, "pca_latent")
            rel = os.path.relpath(pca_dir, LOGS_ROOT)
            results[rel] = pca_dir
    return results


def _find_tsne_dirs():
    """Scan logs/** for eval/mode* directories with tsne images."""
    results = {}
    for root, dirs, files in os.walk(LOGS_ROOT):
        if any(f.startswith("tsne_") and f.endswith(".png") for f in files):
            rel = os.path.relpath(root, LOGS_ROOT)
            results[rel] = root
    return results


def _load_images_in_dir(directory, pattern="*.png"):
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    return [(os.path.basename(p), Image.open(p)) for p in paths if os.path.exists(p)]


def render():
    st.markdown("""
    <div class="page-hero">
        <div class="page-title">🔬 Latent Space Visualization</div>
        <div class="page-subtitle">
            PCA and t-SNE projections of the latent mean (μ) and projected space embeddings.
            These visualizations reveal class separability and cluster structure.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar selectors ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗂️ Select Run</div>', unsafe_allow_html=True)

    tab_pca, tab_tsne = st.tabs(["📐 PCA Latent", "🌀 t-SNE Embeddings"])

    # ── PCA Tab ───────────────────────────────────────────────────────────────
    with tab_pca:
        pca_dirs = _find_pca_dirs()
        if not pca_dirs:
            st.warning("No `pca_latent` directories found in `logs/`. Run the PCA analysis script first.")
        else:
            selected_run = st.selectbox(
                "Training run",
                list(pca_dirs.keys()),
                key="pca_run_select",
            )
            pca_dir = pca_dirs[selected_run]

            st.markdown(f"""
            <div class="info-box">
                📂 <b>Source:</b> <code>{os.path.relpath(pca_dir, os.path.join(LOGS_ROOT, '..')).replace(os.sep, '/')}</code>
            </div>
            """, unsafe_allow_html=True)

            images = _load_images_in_dir(pca_dir, "*.png")
            if not images:
                st.info("No PNG images found in this directory.")
            else:
                # Group by prefix
                mu_imgs    = [(n, i) for n, i in images if n.startswith("mu_")]
                proj_imgs  = [(n, i) for n, i in images if n.startswith("proj_")]
                other_imgs = [(n, i) for n, i in images if not n.startswith(("mu_", "proj_"))]

                def _show_group(group, title, badge_cls):
                    if not group:
                        return
                    st.markdown(f"""
                    <div class="section-header">
                        <span class="badge {badge_cls}">{title}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    cols = st.columns(min(len(group), 3))
                    for i, (name, img) in enumerate(group):
                        label = (name
                                 .replace("mu_", "μ – ")
                                 .replace("proj_", "Proj – ")
                                 .replace("_", " ")
                                 .replace(".png", "")
                                 .title())
                        with cols[i % 3]:
                            st.image(img, caption=label, use_container_width=True)

                _show_group(mu_imgs,   "Latent Mean (μ)", "badge-violet")
                _show_group(proj_imgs, "Projected Space", "badge-cyan")
                _show_group(other_imgs, "Other", "badge-amber")

    # ── t-SNE Tab ─────────────────────────────────────────────────────────────
    with tab_tsne:
        tsne_dirs = _find_tsne_dirs()
        if not tsne_dirs:
            st.warning("No t-SNE images found in eval directories.")
        else:
            selected_tsne = st.selectbox(
                "Evaluation run",
                list(tsne_dirs.keys()),
                key="tsne_run_select",
            )
            tsne_dir = tsne_dirs[selected_tsne]

            tsne_images = _load_images_in_dir(tsne_dir, "tsne_*.png")
            all_images  = _load_images_in_dir(tsne_dir, "*.png")

            if not all_images:
                st.info("No PNG images in this directory.")
            else:
                # Show tsne + score/sigma distribution images
                st.markdown('<div class="section-header">🌀 t-SNE Plots</div>', unsafe_allow_html=True)
                if tsne_images:
                    tc = st.columns(len(tsne_images))
                    for col, (name, img) in zip(tc, tsne_images):
                        label = name.replace("tsne_", "t-SNE: ").replace(".png", "").replace("_", " ").title()
                        col.image(img, caption=label, use_container_width=True)
                else:
                    st.info("No t-SNE images found. Looking for other images…")

                # Other images in this dir
                other = [(n, i) for n, i in all_images if not n.startswith("tsne_")]
                if other:
                    st.markdown('<div class="section-header">📊 Additional Plots</div>', unsafe_allow_html=True)
                    oc = st.columns(min(len(other), 3))
                    for i, (name, img) in enumerate(other):
                        label = name.replace(".png", "").replace("_", " ").title()
                        with oc[i % 3]:
                            oc[i % 3].image(img, caption=label, use_container_width=True)

    # ── Interpretation guide ──────────────────────────────────────────────────
    with st.expander("💡 How to read these plots"):
        st.markdown("""
        | Plot | What to look for |
        |------|-----------------|
        | **PCA 2D** | Well-separated class clusters → good identity embedding |
        | **t-SNE μ** | Tight intra-class clusters, wide inter-class gaps → discriminative latent mean |
        | **t-SNE proj** | Projected space after ArcFace training – should show cleaner margins |
        | **PCA Explained Variance** | Rapid accumulation → lower-dim structure, fewer PCs needed |
        | **μ Distribution** | Roughly Gaussian → well-regularized posterior |
        | **μ Norm** | Consistent L2 norms across classes → uniform embedding magnitude |
        """)
