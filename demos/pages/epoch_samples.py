"""Epoch Samples page – browse reconstruction images across training epochs."""

import os
import glob
import streamlit as st
from PIL import Image

LOGS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "logs")


def _find_epoch_sample_dirs():
    results = {}
    for root, dirs, files in os.walk(LOGS_ROOT):
        if "epoch_samples" in dirs:
            ep_dir = os.path.join(root, "epoch_samples")
            rel = os.path.relpath(ep_dir, LOGS_ROOT)
            # Count images
            n = len(glob.glob(os.path.join(ep_dir, "*.png")))
            if n > 0:
                results[rel] = (ep_dir, n)
    return results


def render():
    st.markdown("""
    <div class="page-hero">
        <div class="page-title">🖼️ Epoch Samples</div>
        <div class="page-subtitle">
            Browse training and validation reconstruction samples across epochs.
            Observe how the U-Net decoder progressively learns to reconstruct
            palm-line structure from the stochastic latent code.
        </div>
    </div>
    """, unsafe_allow_html=True)

    epoch_dirs = _find_epoch_sample_dirs()

    if not epoch_dirs:
        st.warning("No `epoch_samples` directories found in `logs/`.")
        return

    # ── Run selector ──────────────────────────────────────────────────────────
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        run_options = {k: f"{k}  ({v[1]} images)" for k, v in epoch_dirs.items()}
        selected = st.selectbox(
            "Training run",
            list(run_options.keys()),
            format_func=lambda k: run_options[k],
            key="ep_run",
        )
    ep_dir = epoch_dirs[selected][0]

    # Load all images
    train_imgs = sorted(glob.glob(os.path.join(ep_dir, "train_epoch_*.png")))
    val_imgs   = sorted(glob.glob(os.path.join(ep_dir, "val_epoch_*.png")))

    def _epoch_num(path):
        base = os.path.basename(path)
        try:
            return int(base.split("_epoch_")[1].replace(".png", ""))
        except Exception:
            return 0

    train_epochs = [_epoch_num(p) for p in train_imgs]
    val_epochs   = [_epoch_num(p) for p in val_imgs]

    st.markdown(f"""
    <div class="info-box">
        📁 <b>{len(train_imgs)}</b> train samples · <b>{len(val_imgs)}</b> val samples
        · Epochs: <b>{min(train_epochs, default=0)}–{max(train_epochs, default=0)}</b>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode: single epoch or animation ──────────────────────────────────────
    tab_browse, tab_compare, tab_anim = st.tabs([
        "🔍 Browse Epoch",
        "🔀 Side-by-Side Compare",
        "🎞️ Epoch Animation",
    ])

    # ── Browse tab ────────────────────────────────────────────────────────────
    with tab_browse:
        split = st.radio("Split", ["Train", "Validation"], horizontal=True, key="ep_split")
        imgs  = train_imgs if split == "Train" else val_imgs
        epochs = train_epochs if split == "Train" else val_epochs

        if not imgs:
            st.info(f"No {split.lower()} samples found.")
        else:
            ep_sel = st.slider(
                "Epoch",
                min_value=min(epochs),
                max_value=max(epochs),
                value=max(epochs) // 2,
                key="ep_slider",
            )
            # Find closest
            idx = min(range(len(epochs)), key=lambda i: abs(epochs[i] - ep_sel))
            img = Image.open(imgs[idx])

            bc1, bc2 = st.columns([3, 1])
            with bc2:
                st.markdown(f"""
                <div class="metric-card" style="margin-top:0.5rem;">
                    <div class="metric-label">Epoch</div>
                    <div class="metric-value">{epochs[idx]:03d}</div>
                    <div class="metric-delta" style="color:var(--text-muted);">
                        {os.path.basename(imgs[idx])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                zoom = st.slider("Zoom", 0.3, 1.0, 0.85, 0.05, key="ep_zoom")
            with bc1:
                w = int(img.width * zoom)
                h = int(img.height * zoom)
                st.image(img.resize((w, h)), caption=f"Epoch {epochs[idx]:03d} · {split}")

    # ── Compare tab ───────────────────────────────────────────────────────────
    with tab_compare:
        if not train_imgs:
            st.info("No samples to compare.")
        else:
            cc1, cc2 = st.columns(2)
            with cc1:
                ep_a = st.selectbox("Epoch A", train_epochs, index=0, key="ep_cmp_a")
            with cc2:
                ep_b = st.selectbox("Epoch B", train_epochs, index=len(train_epochs) - 1, key="ep_cmp_b")

            idx_a = train_epochs.index(ep_a)
            idx_b = train_epochs.index(ep_b)

            ic1, ic2 = st.columns(2)
            ic1.image(Image.open(train_imgs[idx_a]), caption=f"Train Epoch {ep_a:03d}", use_container_width=True)
            ic2.image(Image.open(train_imgs[idx_b]), caption=f"Train Epoch {ep_b:03d}", use_container_width=True)

            if val_imgs:
                vc1, vc2 = st.columns(2)
                ve_a_idx = min(range(len(val_epochs)), key=lambda i: abs(val_epochs[i] - ep_a))
                ve_b_idx = min(range(len(val_epochs)), key=lambda i: abs(val_epochs[i] - ep_b))
                vc1.image(Image.open(val_imgs[ve_a_idx]), caption=f"Val Epoch {val_epochs[ve_a_idx]:03d}", use_container_width=True)
                vc2.image(Image.open(val_imgs[ve_b_idx]), caption=f"Val Epoch {val_epochs[ve_b_idx]:03d}", use_container_width=True)

    # ── Animation tab ─────────────────────────────────────────────────────────
    with tab_anim:
        st.markdown("""
        <div class="info-box">
            Manually step through epochs below to observe reconstruction quality evolving over training.
            Use the Next/Prev buttons or adjust the epoch slider.
        </div>
        """, unsafe_allow_html=True)

        anim_split = st.radio("Split", ["Train", "Validation"], horizontal=True, key="anim_split")
        a_imgs   = train_imgs if anim_split == "Train" else val_imgs
        a_epochs = train_epochs if anim_split == "Train" else val_epochs

        if not a_imgs:
            st.info("No images available.")
        else:
            if "anim_idx" not in st.session_state:
                st.session_state["anim_idx"] = 0
            if "anim_split_prev" not in st.session_state:
                st.session_state["anim_split_prev"] = anim_split
            if st.session_state["anim_split_prev"] != anim_split:
                st.session_state["anim_idx"] = 0
                st.session_state["anim_split_prev"] = anim_split

            ab1, ab2, ab3 = st.columns([1, 3, 1])
            with ab1:
                if st.button("⬅️ Prev", key="anim_prev"):
                    st.session_state["anim_idx"] = max(0, st.session_state["anim_idx"] - 1)
            with ab3:
                if st.button("Next ➡️", key="anim_next"):
                    st.session_state["anim_idx"] = min(len(a_imgs) - 1, st.session_state["anim_idx"] + 1)

            idx = st.session_state["anim_idx"]
            prog = (idx + 1) / len(a_imgs)
            st.progress(prog, text=f"Epoch {a_epochs[idx]:03d}  ({idx+1}/{len(a_imgs)})")

            img = Image.open(a_imgs[idx])
            st.image(img, caption=f"{anim_split} – Epoch {a_epochs[idx]:03d}", use_container_width=True)

    # ── Interpretation guide ──────────────────────────────────────────────────
    with st.expander("💡 What to look for in epoch samples"):
        st.markdown("""
        | Stage | Expected behaviour |
        |-------|-------------------|
        | **Early epochs (0–10)** | Blurry, noisy reconstructions – the decoder is still learning |
        | **Mid training (20–50)** | Palm-line structure starts to appear; overall shape is correct |
        | **Late training (50+)** | Sharp, fine-grained reconstructions with visible crease lines |
        | **Train vs. Val gap** | Large gap → overfitting; small gap → good generalisation |

        The U-Net decoder is **dropped at inference time** and only serves as a training regularizer
        to force the latent code to retain spatial information.
        """)
