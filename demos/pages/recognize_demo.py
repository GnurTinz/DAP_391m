"""
demos/pages/recognize_demo.py
==============================
Palm Recognition Demo – Gallery + Upload

Features:
  • Gallery viewer: browse enrolled persons with sample images
  • Upload mode: đẩy ảnh lòng bàn tay → nhận diện ngay
  • Enroll mode: thêm người mới vào gallery bằng cách upload ảnh
  • Full result panel: identity, confidence, gates, top-K chart
"""

import os
import sys
import shutil
import datetime
import math
import numpy as np
import streamlit as st
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(DEMO_DIR, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "demos") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "demos"))

from inference_utils import (
    load_model, preprocess_pil, crop_palm_roi,
    embed_image, match_gallery, build_gallery, uncertainty_score,
)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_CKPT    = os.path.join(
    ROOT, "logs",
    "UnetPalmModel_Resnet18_version_32_Original_KL1",
    "version_32", "checkpoints", "last.ckpt",
)
DEFAULT_GALLERY = os.path.join(ROOT, "data", "collect")
PADDING         = 30
GALLERY_THUMB_W = 110   # px for gallery thumbnails

# ── Model loader (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(ckpt_path: str):
    return load_model(ckpt_path)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _list_persons(gallery_dir: str) -> list[str]:
    if not os.path.isdir(gallery_dir):
        return []
    persons = [
        d for d in os.listdir(gallery_dir)
        if d.startswith("person_") and os.path.isdir(os.path.join(gallery_dir, d))
    ]
    return sorted(persons, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)


def _get_person_images(gallery_dir: str, person_id: str,
                       hand: str = "both", max_imgs: int = 8) -> list[str]:
    """Return up to max_imgs image paths for a person."""
    hands = ["left", "right"] if hand == "both" else [hand]
    paths = []
    for h in hands:
        h_dir = os.path.join(gallery_dir, person_id, h)
        if os.path.isdir(h_dir):
            for f in sorted(os.listdir(h_dir)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(h_dir, f))
    return paths[:max_imgs]


def _count_person_images(gallery_dir: str, person_id: str) -> dict:
    """Count images per hand for a person."""
    counts = {"left": 0, "right": 0}
    for h in ["left", "right"]:
        h_dir = os.path.join(gallery_dir, person_id, h)
        if os.path.isdir(h_dir):
            counts[h] = len([
                f for f in os.listdir(h_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
    return counts


def _next_person_id(gallery_dir: str) -> str:
    """Auto-increment person ID."""
    existing = _list_persons(gallery_dir)
    if not existing:
        return "person_1"
    nums = [int(p.split("_")[1]) for p in existing if p.split("_")[1].isdigit()]
    return f"person_{max(nums) + 1}" if nums else "person_1"


def _save_uploaded_images(gallery_dir: str, person_id: str,
                           hand: str, uploaded_files) -> int:
    """Save uploaded PIL images to gallery dir. Returns number saved."""
    hand_dir = os.path.join(gallery_dir, person_id, hand)
    os.makedirs(hand_dir, exist_ok=True)
    existing = [f for f in os.listdir(hand_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    start_idx = len(existing) + 1
    saved = 0
    for i, uf in enumerate(uploaded_files):
        try:
            img = Image.open(uf).convert("RGB")
            out_path = os.path.join(hand_dir, f"{start_idx + i}.jpg")
            img.save(out_path, "JPEG", quality=90)
            saved += 1
        except Exception:
            continue
    return saved


# ── HTML builders ──────────────────────────────────────────────────────────────
def _badge(text: str, color: str = "#7c3aed") -> str:
    return (
        f"<span style='background:{color}22; color:{color}; border:1px solid {color}44;"
        f"border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:600;'>{text}</span>"
    )


def _result_card(accepted: bool, name: str, score: float) -> str:
    if accepted:
        return f"""
        <div style='background:linear-gradient(135deg,#064e3b,#065f46);
             border:1px solid #10b981; border-radius:16px; padding:1.6rem 1.8rem;
             text-align:center; box-shadow:0 0 40px rgba(16,185,129,0.25);'>
          <div style='font-size:2.8rem;'>✅</div>
          <div style='font-size:1.5rem; font-weight:800; color:#d1fae5; margin-top:0.3rem;'>NHẬN DIỆN THÀNH CÔNG</div>
          <div style='font-size:1.1rem; color:#6ee7b7; margin-top:0.5rem; font-weight:600;'>{name}</div>
          <div style='font-size:0.85rem; color:#a7f3d0; margin-top:0.3rem;'>Cosine Similarity: <b>{score:.4f}</b></div>
        </div>"""
    else:
        return f"""
        <div style='background:linear-gradient(135deg,#450a0a,#7f1d1d);
             border:1px solid #ef4444; border-radius:16px; padding:1.6rem 1.8rem;
             text-align:center; box-shadow:0 0 40px rgba(239,68,68,0.25);'>
          <div style='font-size:2.8rem;'>❌</div>
          <div style='font-size:1.5rem; font-weight:800; color:#fee2e2; margin-top:0.3rem;'>KHÔNG NHẬN DIỆN ĐƯỢC</div>
          <div style='font-size:0.85rem; color:#fca5a5; margin-top:0.5rem;'>Best match: {name} ({score:.4f})</div>
          <div style='font-size:0.75rem; color:#f87171; margin-top:0.2rem;'>Ảnh không đạt threshold — người lạ hoặc ảnh kém chất lượng</div>
        </div>"""


def _gate_html(label: str, formula: str, value_str: str, passed: bool) -> str:
    color  = "#6ee7b7" if passed else "#fca5a5"
    icon   = "✓" if passed else "✗"
    bg     = "rgba(16,185,129,0.1)" if passed else "rgba(239,68,68,0.1)"
    border = "#10b981" if passed else "#ef4444"
    return f"""
    <div style='display:flex; align-items:center; gap:10px;
         background:{bg}; border:1px solid {border}; border-radius:10px;
         padding:0.55rem 1rem; margin:4px 0;'>
      <span style='font-size:1.1rem; color:{color}; font-weight:700;'>{icon}</span>
      <div style='flex:1;'>
        <div style='font-weight:600; font-size:0.82rem; color:#e2e8f0;'>{label}</div>
        <div style='font-family:monospace; font-size:0.7rem; color:#64748b;'>{formula}</div>
      </div>
      <div style='font-weight:700; color:{color}; font-size:0.88rem;'>{value_str}</div>
    </div>"""


def _person_card_html(person_id: str, counts: dict, is_enrolled: bool) -> str:
    left_c  = counts.get("left", 0)
    right_c = counts.get("right", 0)
    total   = left_c + right_c
    enrolled_badge = (
        "<span style='font-size:0.65rem; background:#065f46; color:#6ee7b7; "
        "border-radius:20px; padding:1px 8px; font-weight:600;'>enrolled</span>"
        if is_enrolled else ""
    )
    return f"""
    <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
         border-radius:12px; padding:0.8rem; cursor:pointer; transition:all 0.2s;'>
      <div style='display:flex; align-items:center; justify-content:space-between;'>
        <span style='font-weight:700; color:#e2e8f0; font-size:0.9rem;'>🖐 {person_id}</span>
        {enrolled_badge}
      </div>
      <div style='margin-top:0.4rem; font-size:0.72rem; color:#64748b;'>
        L:{left_c} · R:{right_c} · Total:{total}
      </div>
    </div>"""


# ==============================================================================
# GALLERY SECTION
# ==============================================================================

def _render_gallery_section(gallery_dir: str, enrolled_ids: set):
    st.markdown(
        "<div class='section-header'>🖼️ Gallery – Danh sách người đã đăng ký</div>",
        unsafe_allow_html=True,
    )

    all_persons = _list_persons(gallery_dir)
    if not all_persons:
        st.info("Chưa có người nào trong gallery. Sử dụng tab **Enroll** để thêm người mới.")
        return

    # ── Filters bar ────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search_q = st.text_input(
            "🔍 Tìm kiếm", placeholder="Nhập ID hoặc tên...",
            key="gal_search", label_visibility="collapsed"
        )
    with fc2:
        gal_hand = st.selectbox("Hand", ["both", "left", "right"], key="gal_hand_filter")
    with fc3:
        show_enrolled_only = st.checkbox("Chỉ enrolled", key="gal_enrolled_only")

    filtered = [p for p in all_persons if search_q.lower() in p.lower()]
    if show_enrolled_only:
        filtered = [p for p in filtered if p in enrolled_ids]

    st.markdown(
        f"<div style='font-size:0.78rem; color:#64748b; margin:0.3rem 0 0.8rem;'>"
        f"Hiển thị <b>{len(filtered)}</b> / {len(all_persons)} người · "
        f"{len(enrolled_ids)} đã build gallery</div>",
        unsafe_allow_html=True,
    )

    # ── Grid of person cards ────────────────────────────────────────────────────
    COLS = 4
    for row_start in range(0, len(filtered), COLS):
        row_persons = filtered[row_start: row_start + COLS]
        cols = st.columns(COLS)
        for col, person_id in zip(cols, row_persons):
            with col:
                counts   = _count_person_images(gallery_dir, person_id)
                enrolled = person_id in enrolled_ids
                img_paths = _get_person_images(gallery_dir, person_id, gal_hand, max_imgs=1)

                # Thumbnail
                if img_paths and os.path.exists(img_paths[0]):
                    try:
                        thumb = Image.open(img_paths[0]).convert("RGB")
                        # Square crop
                        w, h = thumb.size
                        s = min(w, h)
                        left_c = (w - s) // 2
                        top_c  = (h - s) // 2
                        thumb  = thumb.crop((left_c, top_c, left_c + s, top_c + s))
                        thumb  = thumb.resize((GALLERY_THUMB_W, GALLERY_THUMB_W), Image.LANCZOS)
                        st.image(thumb, use_container_width=True)
                    except Exception:
                        st.markdown(
                            "<div style='height:110px; background:rgba(255,255,255,0.05); "
                            "border-radius:8px; display:flex; align-items:center; "
                            "justify-content:center; color:#475569;'>🖐️</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        "<div style='height:110px; background:rgba(255,255,255,0.05); "
                        "border-radius:8px; display:flex; align-items:center; "
                        "justify-content:center; color:#475569;'>🖐️</div>",
                        unsafe_allow_html=True
                    )

                left_c  = counts.get("left", 0)
                right_c = counts.get("right", 0)
                enr_dot = "🟢 " if enrolled else "⚪ "
                st.markdown(
                    f"<div style='text-align:center; margin-top:0.3rem;'>"
                    f"<div style='font-size:0.82rem; font-weight:700; color:#cbd5e1;'>"
                    f"{enr_dot}{person_id}</div>"
                    f"<div style='font-size:0.68rem; color:#64748b;'>L:{left_c} · R:{right_c}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Detail viewer ───────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    with st.expander("🔍 Xem chi tiết ảnh của một người"):
        detail_person = st.selectbox(
            "Chọn người", all_persons, key="gal_detail_select"
        )
        detail_hand = st.radio("Hand", ["both", "left", "right"],
                               horizontal=True, key="gal_detail_hand")
        detail_max  = st.slider("Số ảnh hiển thị", 4, 20, 8, key="gal_detail_max")

        img_paths = _get_person_images(gallery_dir, detail_person, detail_hand, detail_max)
        if img_paths:
            dcols = st.columns(min(4, len(img_paths)))
            for i, path in enumerate(img_paths):
                with dcols[i % 4]:
                    try:
                        img = Image.open(path).convert("RGB")
                        st.image(img, caption=os.path.basename(path), use_container_width=True)
                    except Exception:
                        st.markdown("⚠️ Lỗi đọc ảnh")
        else:
            st.info("Không tìm thấy ảnh.")


# ==============================================================================
# RECOGNIZE SECTION (upload mode)
# ==============================================================================

def _render_recognize_section(
    gallery: dict, ckpt_path: str,
    tau_S: float, tau_K: float,
):
    st.markdown(
        "<div class='section-header'>🔍 Nhận diện – Upload ảnh lòng bàn tay</div>",
        unsafe_allow_html=True,
    )

    if not gallery:
        st.warning(
            "⚠️ Gallery chưa được build. "
            "Vui lòng vào sidebar và bấm **Build Gallery** trước."
        )
        return

    model = get_model(ckpt_path)

    # ── Upload widget ───────────────────────────────────────────────────────────
    st.markdown(
        "<div style='color:#94a3b8; font-size:0.88rem; margin-bottom:0.5rem;'>"
        "Upload một hoặc nhiều ảnh ROI lòng bàn tay (JPG/PNG) để nhận diện.</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Chọn ảnh lòng bàn tay",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="recog_upload",
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown(
            "<div style='padding:3rem; text-align:center; "
            "background:rgba(255,255,255,0.03); border:2px dashed rgba(255,255,255,0.1); "
            "border-radius:16px; color:#475569;'>"
            "📁 Kéo thả hoặc chọn ảnh để nhận diện</div>",
            unsafe_allow_html=True,
        )
        return

    # Process each uploaded image
    for uf in uploaded:
        st.markdown(
            f"<div style='margin-top:1.2rem; border-top:1px solid rgba(255,255,255,0.08); "
            f"padding-top:1rem;'></div>",
            unsafe_allow_html=True,
        )
        roi = Image.open(uf).convert("RGB")

        img_col, res_col = st.columns([1, 1.2])

        with img_col:
            st.markdown(
                "<div style='font-size:0.78rem; color:#64748b; margin-bottom:0.3rem;'>"
                f"📸 {uf.name}</div>",
                unsafe_allow_html=True,
            )
            st.image(roi, use_container_width=True)

        with res_col:
            # Embed & match
            with st.spinner("Nhận diện..."):
                tensor         = preprocess_pil(roi)
                mu_p, lv_p     = embed_image(model, tensor)
                result         = match_gallery(mu_p, lv_p, gallery, tau_S, tau_K)

            # Result card
            st.markdown(
                _result_card(
                    result["accepted"],
                    result.get("name", "?"),
                    result.get("score", 0.0),
                ),
                unsafe_allow_html=True,
            )

            # Gates
            st.markdown(
                "<div style='margin-top:0.8rem; font-size:0.78rem; "
                "font-weight:600; color:#64748b;'>Open-Set Gates</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                _gate_html(
                    "Similarity",
                    f"S={result['score']:.4f} ≥ τS={tau_S:.2f}",
                    f"{result['score']:.4f}",
                    result["gate_S"],
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                _gate_html(
                    "KL Distance",
                    f"D_SKL={result['d_skl']:.4f} ≤ τK={tau_K:.2f}",
                    f"{result['d_skl']:.4f}",
                    result["gate_K"],
                ),
                unsafe_allow_html=True,
            )

            # Top-K bar chart
            if result.get("all_scores") and len(result["all_scores"]) > 1:
                with st.expander("📊 Top similarity"):
                    import plotly.graph_objects as go
                    sorted_s = sorted(
                        result["all_scores"].items(),
                        key=lambda x: x[1], reverse=True,
                    )[:8]
                    ids_top  = [s[0] for s in sorted_s]
                    vals_top = [s[1] for s in sorted_s]
                    clr      = [
                        "#10b981" if p == result["person_id"] else "#7c3aed"
                        for p in ids_top
                    ]
                    fig = go.Figure(go.Bar(
                        x=vals_top, y=ids_top,
                        orientation="h",
                        marker_color=clr,
                        text=[f"{v:.3f}" for v in vals_top],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8", family="Inter", size=11),
                        margin=dict(l=0, r=30, t=10, b=0),
                        height=max(180, len(ids_top) * 30),
                        xaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# ENROLL SECTION
# ==============================================================================

def _render_enroll_section(gallery_dir: str):
    st.markdown(
        "<div class='section-header'>➕ Enroll – Thêm người mới</div>",
        unsafe_allow_html=True,
    )

    all_persons = _list_persons(gallery_dir)
    auto_id     = _next_person_id(gallery_dir)

    ec1, ec2 = st.columns([1, 1])
    with ec1:
        st.markdown(
            "<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:0.8rem;'>"
            "Thêm người mới hoặc bổ sung ảnh cho người đã có.</div>",
            unsafe_allow_html=True,
        )
        enroll_mode = st.radio(
            "Chế độ",
            ["Thêm người mới", "Bổ sung ảnh cho người cũ"],
            key="enroll_mode",
            horizontal=True,
        )

    if enroll_mode == "Thêm người mới":
        ec1b, ec2b = st.columns([1, 1])
        with ec1b:
            new_id = st.text_input(
                "Person ID (để trống = tự động)",
                value="",
                placeholder=auto_id,
                key="enroll_new_id",
            )
            if not new_id.strip():
                new_id = auto_id
            if not new_id.startswith("person_"):
                new_id = f"person_{new_id}" if not new_id.startswith("person") else new_id
        with ec2b:
            enroll_hand = st.selectbox(
                "Tay chụp", ["left", "right", "both"], key="enroll_hand_new"
            )
    else:
        ec1b, ec2b = st.columns([1, 1])
        with ec1b:
            if not all_persons:
                st.info("Chưa có người nào. Hãy thêm người mới trước.")
                return
            new_id = st.selectbox("Chọn người", all_persons, key="enroll_existing_id")
        with ec2b:
            enroll_hand = st.selectbox(
                "Tay chụp", ["left", "right"], key="enroll_hand_old"
            )

    if enroll_hand == "both":
        # Two separate uploaders
        st.markdown(
            "<div style='font-size:0.82rem; color:#94a3b8; margin:0.5rem 0;'>"
            "Upload ảnh tay trái và tay phải riêng</div>",
            unsafe_allow_html=True,
        )
        uc1, uc2 = st.columns(2)
        with uc1:
            st.markdown("**Tay trái**")
            up_left = st.file_uploader(
                "Tay trái", type=["jpg", "jpeg", "png"],
                accept_multiple_files=True, key="enroll_up_left",
                label_visibility="collapsed",
            )
        with uc2:
            st.markdown("**Tay phải**")
            up_right = st.file_uploader(
                "Tay phải", type=["jpg", "jpeg", "png"],
                accept_multiple_files=True, key="enroll_up_right",
                label_visibility="collapsed",
            )

        if st.button("💾 Lưu ảnh vào gallery", key="enroll_save_both", use_container_width=False):
            saved = 0
            if up_left:
                saved += _save_uploaded_images(gallery_dir, new_id, "left", up_left)
            if up_right:
                saved += _save_uploaded_images(gallery_dir, new_id, "right", up_right)
            if saved > 0:
                st.success(f"✅ Đã lưu **{saved}** ảnh cho **{new_id}**.")
                st.cache_data.clear()
            else:
                st.warning("Chưa có ảnh nào được chọn.")
    else:
        uploaded_enroll = st.file_uploader(
            f"Chọn ảnh tay {'trái' if enroll_hand == 'left' else 'phải'}",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="enroll_up_single",
            label_visibility="collapsed",
        )

        # Preview
        if uploaded_enroll:
            st.markdown(
                f"<div style='font-size:0.78rem; color:#64748b; margin:0.3rem 0;'>"
                f"Preview {len(uploaded_enroll)} ảnh:</div>",
                unsafe_allow_html=True,
            )
            prev_cols = st.columns(min(5, len(uploaded_enroll)))
            for i, uf in enumerate(uploaded_enroll[:5]):
                with prev_cols[i]:
                    st.image(Image.open(uf).convert("RGB"), use_container_width=True)

        if st.button("💾 Lưu ảnh vào gallery", key="enroll_save_single", use_container_width=False):
            if not uploaded_enroll:
                st.warning("Chưa chọn ảnh nào.")
            else:
                saved = _save_uploaded_images(gallery_dir, new_id, enroll_hand, uploaded_enroll)
                if saved > 0:
                    st.success(f"✅ Đã lưu **{saved}** ảnh cho **{new_id}** ({enroll_hand}).")
                    st.cache_data.clear()
                else:
                    st.error("Lỗi khi lưu ảnh.")

    # ── Current status of this person ─────────────────────────────────────────
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    counts = _count_person_images(gallery_dir, new_id)
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1); "
        f"border-radius:10px; padding:0.8rem 1rem; font-size:0.82rem; color:#94a3b8;'>"
        f"📂 <b>{new_id}</b> — "
        f"Tay trái: <b style='color:#06b6d4;'>{counts['left']}</b> ảnh · "
        f"Tay phải: <b style='color:#a78bfa;'>{counts['right']}</b> ảnh</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN RENDER
# ==============================================================================

def render():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-hero">
        <div class="page-title">🖐️ Palm Recognition Demo</div>
        <div class="page-subtitle">
            Nhận diện danh tính qua lòng bàn tay.<br>
            Upload ảnh → MediaPipe ROI crop → Probabilistic Encoder → Cosine Matching → Open-Set Decision
        </div>
        <div style="margin-top:1rem; display:flex; gap:8px; flex-wrap:wrap;">
            <span class="badge badge-cyan">Mode M3 – cos(μ_p, μ_g)</span>
            <span class="badge badge-violet">Tri-threshold Rejection</span>
            <span class="badge badge-emerald">Upload & Enroll</span>
            <span class="badge badge-amber">ResNet18 + UNet</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: config ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-weight:700; font-size:0.9rem; color:#a78bfa; "
            "margin-bottom:0.5rem;'>⚙️ Config</div>",
            unsafe_allow_html=True,
        )

        ckpt_path = st.text_input(
            "Checkpoint", value=DEFAULT_CKPT,
            key="rec_ckpt", help="Path to .ckpt file",
        )
        ckpt_valid = os.path.isfile(ckpt_path)
        if not ckpt_valid:
            st.warning("⚠️ Checkpoint không tìm thấy")

        gallery_dir = st.text_input(
            "Gallery dir", value=DEFAULT_GALLERY,
            key="rec_gallery_dir",
        )

        all_persons = _list_persons(gallery_dir)
        if all_persons:
            n_default = min(20, len(all_persons))
            selected_persons = st.multiselect(
                "Persons enrolled",
                options=all_persons,
                default=all_persons[:n_default],
                key="rec_persons",
            )
        else:
            st.info("Không tìm thấy person_X folders.")
            selected_persons = []

        hand_filter = st.selectbox("Hand", ["both", "left", "right"], key="rec_hand")

        st.markdown(
            "<hr style='border-color:rgba(255,255,255,0.08);'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:0.78rem; font-weight:600; color:#64748b; "
            "margin-bottom:0.4rem;'>Thresholds</div>",
            unsafe_allow_html=True,
        )
        tau_S = st.slider("τ-S (Similarity)", 0.0, 1.0, 0.50, 0.01, key="rec_tau_S")
        tau_K = st.slider("τ-K (KL distance)", 0.0, 50.0, 10.0, 0.5,  key="rec_tau_K")

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        build_btn = st.button(
            "💾 Build Gallery",
            key="rec_build",
            disabled=(not ckpt_valid or not selected_persons),
            use_container_width=True,
        )

    # ── Session state init ─────────────────────────────────────────────────────
    if "rec_gallery" not in st.session_state:
        st.session_state.rec_gallery = {}

    # ── Build gallery ──────────────────────────────────────────────────────────
    if build_btn:
        if not ckpt_valid:
            st.error("Checkpoint không hợp lệ.")
        elif not selected_persons:
            st.error("Chưa chọn người nào.")
        else:
            with st.spinner("Đang load model..."):
                try:
                    model = get_model(ckpt_path)
                except Exception as e:
                    st.error(f"Load model thất bại: {e}")
                    st.stop()

            bar = st.progress(0, text="Building gallery…")

            def _cb(cur, total, pid):
                pct = int(cur / max(total, 1) * 100)
                bar.progress(pct, text=f"Embedding {pid} ({cur+1}/{total})…")

            try:
                gal = build_gallery(
                    model, gallery_dir, selected_persons,
                    hand=hand_filter, progress_cb=_cb,
                )
                bar.progress(100, text="Done!")

                # Extract and display any per-image errors
                build_errors = gal.pop("_errors", [])
                st.session_state.rec_gallery = gal

                st.success(f"✅ Gallery built: **{len(gal)}** người đã enrolled.")
                if build_errors:
                    with st.expander(
                        f"⚠️ {len(build_errors)} ảnh không embed được — click để xem chi tiết",
                        expanded=(len(gal) == 0),
                    ):
                        for err in build_errors:
                            st.error(err)
            except Exception as e:
                st.error(f"Build gallery thất bại: {e}")

    gallery = st.session_state.rec_gallery
    enrolled_ids = set(gallery.keys())

    # ── Gallery status strip ───────────────────────────────────────────────────
    if gallery:
        kc1, kc2, kc3, kc4 = st.columns(4)
        total_imgs = sum(v["n_imgs"] for v in gallery.values() if isinstance(v, dict))
        kc1.markdown(
            f"<div class='metric-card'><div class='metric-label'>Enrolled</div>"
            f"<div class='metric-value'>{len(gallery)}</div></div>",
            unsafe_allow_html=True,
        )
        kc2.markdown(
            f"<div class='metric-card cool'><div class='metric-label'>Images</div>"
            f"<div class='metric-value'>{total_imgs}</div></div>",
            unsafe_allow_html=True,
        )
        kc3.markdown(
            "<div class='metric-card'><div class='metric-label'>Mode</div>"
            "<div class='metric-value' style='font-size:0.9rem;'>M3 cos(μ)</div></div>",
            unsafe_allow_html=True,
        )
        kc4.markdown(
            f"<div class='metric-card warm'><div class='metric-label'>τ-S</div>"
            f"<div class='metric-value'>{tau_S:.2f}</div></div>",
            unsafe_allow_html=True,
        )

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_gallery, tab_recognize, tab_enroll = st.tabs([
        "🖼️  Gallery",
        "🔍  Nhận diện",
        "➕  Enroll người mới",
    ])

    with tab_gallery:
        _render_gallery_section(gallery_dir, enrolled_ids)

    with tab_recognize:
        _render_recognize_section(gallery, ckpt_path, tau_S, tau_K)

    with tab_enroll:
        _render_enroll_section(gallery_dir)

    # ── How it works ───────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("ℹ️ Cách hoạt động"):
        st.markdown("""
        | Bước | Mô tả |
        |------|-------|
        | **1. Build Gallery** | Chọn persons → encode ảnh → tính mean μ mỗi người |
        | **2. Upload ảnh** | Upload ảnh lòng bàn tay (JPG/PNG) |
        | **3. ROI Crop** | MediaPipe HandLandmarker cắt vùng bàn tay + padding 30px |
        | **4. Encode** | `model.encoder(x)` → `(μ, log σ²)` — ResNet18 backbone |
        | **5. Match (M3)** | `cos(μ_probe, μ_gallery)` với tất cả người enrolled |
        | **6. Decision** | Tri-threshold: `U_p ≤ τU` AND `S ≥ τS` AND `D_SKL ≤ τK` |
        | **7. Enroll** | Upload ảnh mới → lưu vào `data/collect/person_X/hand/` |
        """)
