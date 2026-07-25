"""
PALM Project Dashboard
Probabilistic Palmprint Embedding with Generative Regularization for Open-Set Identification
"""

import streamlit as st
import sys
import os

# ── Path setup ──────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_DIR = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
sys.path.insert(0, DEMO_DIR)

# ── Page config (MUST be first Streamlit call) ──────────────────────────────────
st.set_page_config(
    page_title="PALM · Research Dashboard",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS injection ────────────────────────────────────────────────────────────────
css_path = os.path.join(DEMO_DIR, "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Pages ────────────────────────────────────────────────────────────────────────
from pages import home, results_overview, latent_viz, epoch_samples, inference_modes, score_distribution, paper_report, ccnet_demo, attendance_demo, recognize_demo

PAGES = {
    "🏠  Home": home,
    "📊  Results Overview": results_overview,
    "📋  Paper Report": paper_report,
    "🔬  Latent Space": latent_viz,
    "🖼️  Epoch Samples": epoch_samples,
    "⚡  Inference Modes": inference_modes,
    "📈  Score Distributions": score_distribution,
    "🔬  CCNet Demo": ccnet_demo,
    "🎯  Attendance Demo": attendance_demo,
    "🖐️  Palm Recognition": recognize_demo,
}

# ── Sidebar nav ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="logo-icon">🖐️</span>
        <div>
            <div class="brand-name">PALM</div>
            <div class="brand-sub">Research Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='sidebar-divider'/>", unsafe_allow_html=True)

    selected = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

    st.markdown("<hr class='sidebar-divider'/>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-meta">
        <b>Probabilistic Palmprint Embedding</b><br>
        with Generative Regularization<br>
        for Open-Set Identification
    </div>
    """, unsafe_allow_html=True)

# ── Render page ──────────────────────────────────────────────────────────────────
PAGES[selected].render()
