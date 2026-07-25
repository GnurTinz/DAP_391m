"""
demos/pages/attendance_demo.py
==============================
Real-time Palm Attendance Demo page.

Flow:
  Phase 1 – Gallery Build : load model + embed enrolled persons
  Phase 2 – Identification: snapshot via camera -> ROI crop -> match -> log
"""

import os
import sys
import csv
import math
import datetime
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ── Path setup ─────────────────────────────────────────────────────────────────
DEMO_DIR = os.path.dirname(__file__)
ROOT     = os.path.abspath(os.path.join(DEMO_DIR, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "demos") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "demos"))

from inference_utils import (
    load_model, preprocess_pil, crop_palm_roi,
    embed_image, match_gallery, build_gallery,
)

# ── Paths ───────────────────────────────────────────────────────────────────────
DEFAULT_CKPT    = os.path.join(
    ROOT, "logs",
    "UnetPalmModel_Resnet18_version_32_Original_KL1",
    "version_32", "checkpoints", "last.ckpt",
)
DEFAULT_GALLERY = os.path.join(ROOT, "data", "collect")
ATTENDANCE_CSV  = os.path.join(ROOT, "tasks", "attendance.csv")


# ── Model loader (cached across sessions) ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(ckpt_path: str):
    return load_model(ckpt_path)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _list_persons(gallery_dir: str) -> list:
    if not os.path.isdir(gallery_dir):
        return []
    return sorted(
        [d for d in os.listdir(gallery_dir)
         if d.startswith("person_") and os.path.isdir(os.path.join(gallery_dir, d))],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0,
    )


def _append_attendance_log(row: dict):
    os.makedirs(os.path.dirname(ATTENDANCE_CSV), exist_ok=True)
    write_header = not os.path.exists(ATTENDANCE_CSV)
    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _result_badge_html(accepted: bool, name: str, score: float) -> str:
    if accepted:
        return (
            "<div style='"
            "background:linear-gradient(135deg,#065f46,#10b981);"
            "border-radius:16px; padding:1.5rem 2rem; text-align:center;"
            "box-shadow:0 0 32px rgba(16,185,129,0.4);'>"
            "<div style='font-size:3rem;'>&#10003;</div>"
            "<div style='font-size:1.4rem; font-weight:800; color:#d1fae5;'>ACCEPTED</div>"
            "<div style='font-size:1.1rem; color:#a7f3d0; margin-top:0.4rem;'>" + name + "</div>"
            "<div style='font-size:0.85rem; color:#6ee7b7; margin-top:0.2rem;'>"
            "Similarity {:.3f}</div></div>".format(score)
        )
    else:
        return (
            "<div style='"
            "background:linear-gradient(135deg,#7f1d1d,#ef4444);"
            "border-radius:16px; padding:1.5rem 2rem; text-align:center;"
            "box-shadow:0 0 32px rgba(239,68,68,0.4);'>"
            "<div style='font-size:3rem;'>&#215;</div>"
            "<div style='font-size:1.4rem; font-weight:800; color:#fee2e2;'>REJECTED</div>"
            "<div style='font-size:0.85rem; color:#fca5a5; margin-top:0.4rem;'>"
            "Best match: {} ({:.3f})</div></div>".format(name, score)
        )


def _gate_row(label, formula, value_str, passed: bool) -> str:
    color  = "#6ee7b7" if passed else "#f9a8d4"
    icon   = "&#10003;" if passed else "&#215;"
    bg     = "rgba(16,185,129,0.1)" if passed else "rgba(239,68,68,0.1)"
    border = "#10b981" if passed else "#ef4444"
    return (
        "<div style='display:flex; align-items:center; gap:12px;"
        "background:{}; border:1px solid {}; border-radius:10px;"
        "padding:0.6rem 1rem; margin:4px 0;'>".format(bg, border)
        + "<span style='font-size:1.2rem; color:{};'>{}</span>".format(color, icon)
        + "<div style='flex:1;'>"
        + "<div style='font-weight:600; font-size:0.82rem; color:#f0f4ff;'>" + label + "</div>"
        + "<div style='font-family:monospace; font-size:0.72rem; color:#94a3b8;'>" + formula + "</div>"
        + "</div>"
        + "<div style='font-weight:700; color:{}; font-size:0.9rem;'>{}</div>".format(color, value_str)
        + "</div>"
    )


# ==============================================================================
# RENDER
# ==============================================================================

def render():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-hero">
        <div class="page-title">&#127919; Attendance Demo</div>
        <div class="page-subtitle">
            Demo điểm danh thực tế bằng lòng bàn tay.<br>
            Chụp ảnh bàn tay &#8594; MediaPipe crop ROI &#8594; Encoder embed &#8594;
            Mode M3 matching &#8594; Tri-threshold open-set decision.
        </div>
        <div style="margin-top:1rem; display:flex; gap:8px; flex-wrap:wrap;">
            <span class="badge badge-cyan">Mode M3 – cos(&#956;&#8346;, &#956;&#7523;)</span>
            <span class="badge badge-violet">Tri-threshold Rejection</span>
            <span class="badge badge-emerald">MediaPipe ROI</span>
            <span class="badge badge-amber">ResNet18 + UNet</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: config ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-weight:700; font-size:0.9rem; color:#a78bfa; "
            "margin-bottom:0.5rem;'>&#9881;&#65039; Attendance Config</div>",
            unsafe_allow_html=True,
        )

        # Checkpoint
        ckpt_path = st.text_input(
            "Checkpoint path",
            value=DEFAULT_CKPT,
            key="att_ckpt",
            help="Absolute path to .ckpt file",
        )
        ckpt_valid = os.path.isfile(ckpt_path)
        if not ckpt_valid:
            st.warning("Checkpoint file not found.")

        # Gallery directory
        gallery_dir = st.text_input(
            "Gallery directory",
            value=DEFAULT_GALLERY,
            key="att_gallery_dir",
            help="Root dir containing person_X/ subdirs",
        )

        # Person selection
        all_persons = _list_persons(gallery_dir)
        if all_persons:
            n_default = min(10, len(all_persons))
            selected_persons = st.multiselect(
                "Persons in gallery",
                options=all_persons,
                default=all_persons[:n_default],
                key="att_persons",
            )
        else:
            st.info("No person_X folders found in gallery directory.")
            selected_persons = []

        # Hand filter
        hand_filter = st.selectbox("Hand", ["both", "left", "right"], key="att_hand")

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        # Thresholds
        st.markdown(
            "<div style='font-weight:600; font-size:0.82rem; color:#94a3b8; "
            "margin-bottom:0.3rem;'>Thresholds</div>",
            unsafe_allow_html=True,
        )
        tau_S = st.slider("&#964;&#8209;S  (Similarity)", 0.0, 1.0, 0.50, 0.01, key="att_tau_S",
                          help="Minimum cosine similarity to accept")
        tau_U = st.slider("&#964;&#8209;U  (Uncertainty)", 0.0, 2.0, 0.80, 0.01, key="att_tau_U",
                          help="Maximum sigma mean to accept (lower = stricter quality gate)")
        tau_K = st.slider("&#964;&#8209;K  (KL distance)", 0.0, 50.0, 10.0, 0.5, key="att_tau_K",
                          help="Maximum symmetric KL divergence to accept")

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        # Build gallery button
        build_btn = st.button(
            "&#128190; Build Gallery",
            key="att_build",
            disabled=(not ckpt_valid or not selected_persons),
            use_container_width=True,
        )

    # ── Phase 1: Gallery Build ─────────────────────────────────────────────────
    if "att_gallery" not in st.session_state:
        st.session_state.att_gallery = {}
    if "att_log" not in st.session_state:
        st.session_state.att_log = []

    if build_btn:
        if not ckpt_valid:
            st.error("Cannot build gallery: checkpoint not found.")
        elif not selected_persons:
            st.error("Select at least one person.")
        else:
            with st.spinner("Loading model..."):
                try:
                    model = get_model(ckpt_path)
                    st.success("Model loaded.")
                except Exception as e:
                    st.error("Failed to load model: {}".format(e))
                    st.stop()

            progress_bar = st.progress(0, text="Building gallery…")

            def _progress(cur, total, pid):
                pct = int(cur / max(total, 1) * 100)
                progress_bar.progress(pct, text="Embedding {} ({}/{})…".format(pid, cur + 1, total))

            try:
                gallery = build_gallery(
                    model,
                    gallery_dir,
                    selected_persons,
                    hand=hand_filter,
                    progress_cb=_progress,
                )
                progress_bar.progress(100, text="Done!")

                # Extract and display any per-image errors
                build_errors = gallery.pop("_errors", [])
                st.session_state.att_gallery = gallery

                st.success(
                    "Gallery built: **{}** persons enrolled.".format(len(gallery))
                )
                if build_errors:
                    with st.expander(
                        "⚠️ {} image(s) failed to embed — click to see details".format(
                            len(build_errors)
                        ),
                        expanded=(len(gallery) == 0),  # auto-open if nothing enrolled
                    ):
                        for err in build_errors:
                            st.error(err)
            except Exception as e:
                st.error("Gallery build failed: {}".format(e))

    # ── Gallery status ─────────────────────────────────────────────────────────
    gallery = st.session_state.att_gallery
    if gallery:
        st.markdown(
            "<div class='section-header'>&#128196; Gallery Status</div>",
            unsafe_allow_html=True,
        )
        gcols = st.columns(4)
        gcols[0].markdown(
            "<div class='metric-card'><div class='metric-label'>Persons enrolled</div>"
            "<div class='metric-value'>{}</div></div>".format(len(gallery)),
            unsafe_allow_html=True,
        )
        total_imgs = sum(v["n_imgs"] for v in gallery.values() if isinstance(v, dict))
        gcols[1].markdown(
            "<div class='metric-card cool'><div class='metric-label'>Images embedded</div>"
            "<div class='metric-value'>{}</div></div>".format(total_imgs),
            unsafe_allow_html=True,
        )
        gcols[2].markdown(
            "<div class='metric-card'><div class='metric-label'>Inference Mode</div>"
            "<div class='metric-value' style='font-size:1rem;'>M3 – cos(&#956;)</div></div>",
            unsafe_allow_html=True,
        )
        gcols[3].markdown(
            "<div class='metric-card warm'><div class='metric-label'>Backbone</div>"
            "<div class='metric-value' style='font-size:1rem;'>ResNet18</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "&#128190; Gallery is empty. Configure and click **Build Gallery** in the sidebar "
            "to enroll persons before running identification."
        )

    # ── Phase 2: Camera Capture & Identification ───────────────────────────────
    st.markdown(
        "<div class='section-header'>&#128247; Capture & Identify</div>",
        unsafe_allow_html=True,
    )

    if not gallery:
        st.warning("Build the gallery first before capturing.")
    else:
        model = get_model(ckpt_path)

        cam_col, result_col = st.columns([1, 1])

        with cam_col:
            st.markdown(
                "<div style='font-weight:600; color:#94a3b8; margin-bottom:0.5rem;'>"
                "📷 Upload ảnh ROI lòng bàn tay</div>",
                unsafe_allow_html=True,
            )
            camera_img = st.file_uploader(
                "",
                type=["jpg", "jpeg", "png"],
                key="att_upload",
                help="Upload ảnh ROI lòng bàn tay để nhận diện",
                label_visibility="collapsed",
            )

        with result_col:
            if camera_img is not None:
                roi = Image.open(camera_img).convert("RGB")

                # ── Embed & Match ─────────────────────────────────────
                with st.spinner("Identifying…"):
                    tensor      = preprocess_pil(roi)
                    mu_p, lv_p  = embed_image(model, tensor)
                    result      = match_gallery(
                        mu_p, lv_p, gallery, tau_S, tau_U, tau_K
                    )



                    # ── Gate details ──────────────────────────────────────
                    st.markdown(
                        "<div style='margin-top:1rem; font-weight:600; "
                        "font-size:0.82rem; color:#94a3b8;'>&#128272; Open-Set Gates</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        _gate_row(
                            "Uncertainty Gate",
                            "U_p = {:.4f}  &#8804;  &#964;-U = {:.2f}".format(
                                result["uncertainty"], tau_U),
                            "{:.4f}".format(result["uncertainty"]),
                            result["gate_U"],
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        _gate_row(
                            "Similarity Gate",
                            "S(p,g*) = {:.4f}  &#8805;  &#964;-S = {:.2f}".format(
                                result["score"], tau_S),
                            "{:.4f}".format(result["score"]),
                            result["gate_S"],
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        _gate_row(
                            "KL Gate",
                            "D_SKL = {:.4f}  &#8804;  &#964;-K = {:.2f}".format(
                                result["d_skl"], tau_K),
                            "{:.4f}".format(result["d_skl"]),
                            result["gate_K"],
                        ),
                        unsafe_allow_html=True,
                    )

                    # ── Top-K similarity bar ──────────────────────────────
                    if result.get("all_scores"):
                        with st.expander("&#128202; Top similarity scores"):
                            import plotly.graph_objects as go
                            sorted_scores = sorted(
                                result["all_scores"].items(),
                                key=lambda x: x[1], reverse=True,
                            )[:10]
                            ids_top   = [s[0] for s in sorted_scores]
                            vals_top  = [s[1] for s in sorted_scores]
                            bar_colors = [
                                "#10b981" if i == ids_top.index(result["person_id"]) else "#7c3aed"
                                for i in range(len(ids_top))
                            ]
                            fig = go.Figure(go.Bar(
                                x=vals_top, y=ids_top,
                                orientation="h",
                                marker_color=bar_colors,
                                text=["{:.3f}".format(v) for v in vals_top],
                                textposition="outside",
                            ))
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#94a3b8", family="Inter"),
                                margin=dict(l=0, r=20, t=20, b=0),
                                height=max(200, len(ids_top) * 32),
                                xaxis=dict(
                                    range=[0, 1],
                                    gridcolor="rgba(255,255,255,0.05)",
                                ),
                                yaxis=dict(autorange="reversed"),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # ── Log attendance ────────────────────────────────────
                    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    row = {
                        "timestamp":   ts,
                        "person_id":   result.get("person_id", "?"),
                        "name":        result.get("name", "?"),
                        "accepted":    result["accepted"],
                        "score":       round(result.get("score", 0.0), 4),
                        "uncertainty": round(result.get("uncertainty", 0.0), 4),
                        "d_skl":       round(result.get("d_skl", 0.0), 4),
                        "gate_U":      result.get("gate_U", False),
                        "gate_S":      result.get("gate_S", False),
                        "gate_K":      result.get("gate_K", False),
                        "tau_S":       tau_S,
                        "tau_U":       tau_U,
                        "tau_K":       tau_K,
                    }
                    st.session_state.att_log.append(row)
                    _append_attendance_log(row)
            else:
                st.markdown(
                    "<div style='padding:3rem 1rem; text-align:center; "
                    "color:var(--text-muted); font-size:0.9rem;'>"
                    "&#128247; Chụp ảnh để bắt đầu nhận diện</div>",
                    unsafe_allow_html=True,
                )

    # ── Phase 3: Attendance Log ────────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>&#128203; Attendance Log</div>",
        unsafe_allow_html=True,
    )

    log = st.session_state.att_log
    if log:
        import pandas as pd

        df_log = pd.DataFrame(log[::-1])  # newest first

        # KPI summary
        lc1, lc2, lc3, lc4 = st.columns(4)
        total     = len(df_log)
        accepted  = int(df_log["accepted"].sum())
        rejected  = total - accepted
        acc_rate  = accepted / total * 100 if total else 0.0

        lc1.markdown(
            "<div class='metric-card'><div class='metric-label'>Total checks</div>"
            "<div class='metric-value'>{}</div></div>".format(total),
            unsafe_allow_html=True,
        )
        lc2.markdown(
            "<div class='metric-card cool'><div class='metric-label'>Accepted</div>"
            "<div class='metric-value'>{}</div></div>".format(accepted),
            unsafe_allow_html=True,
        )
        lc3.markdown(
            "<div class='metric-card warm'><div class='metric-label'>Rejected</div>"
            "<div class='metric-value'>{}</div></div>".format(rejected),
            unsafe_allow_html=True,
        )
        lc4.markdown(
            "<div class='metric-card'><div class='metric-label'>Accept rate</div>"
            "<div class='metric-value'>{:.1f}%</div></div>".format(acc_rate),
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
        st.dataframe(
            df_log[["timestamp", "name", "accepted", "score",
                     "uncertainty", "d_skl", "gate_U", "gate_S", "gate_K"]],
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns([1, 4])
        with c1:
            csv_bytes = df_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                "&#8595; Export CSV",
                data=csv_bytes,
                file_name="attendance_{}.csv".format(
                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ),
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            if st.button("&#128465; Clear log (this session)", use_container_width=False):
                st.session_state.att_log = []
                st.rerun()
    else:
        st.info("No attendance records yet. Capture images to start logging.")

    # ── Footer info ────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("&#8505;&#65039; How it works"):
        st.markdown("""
        | Step | Detail |
        |------|--------|
        | **1. Capture** | `st.camera_input()` chụp ảnh từ webcam |
        | **2. ROI Crop** | MediaPipe HandLandmarker (IMAGE mode) phát hiện bàn tay, crop + pad 30px |
        | **3. Preprocess** | Resize 128×128 → ToTensor → Normalize([0.5],[0.5]) |
        | **4. Encode** | `model.encoder(x)` → (μ, log σ²) — latent_dim=32 |
        | **5. Match (M3)** | cos(μ_probe, μ_gallery) cho từng enrolled person |
        | **6. Reject** | Tri-threshold: U_p ≤ τ_U AND S ≥ τ_S AND D_SKL ≤ τ_K |
        | **7. Log** | Ghi vào `tasks/attendance.csv` và session state |
        """)
