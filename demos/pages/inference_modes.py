"""Inference Modes comparison page."""

import os
import re
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

TASKS_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "tasks", "summary_table.csv")

MODE_INFO = {
    "mode0": {
        "name": "M0 – Projected-mean",
        "formula": "s = cos(proj(μ_p), proj(μ_g))",
        "desc": "Cosine similarity between MLP-projected mean vectors. Baseline — no adaptation. Encoder + Projector only.",
        "badge": "badge-violet",
    },
    "model0": {
        "name": "M0 – Projected-mean",
        "formula": "s = cos(proj(μ_p), proj(μ_g))",
        "desc": "Cosine similarity between MLP-projected mean vectors. Baseline — no adaptation. Encoder + Projector only.",
        "badge": "badge-violet",
    },
    "mode1": {
        "name": "M1 – Projected Adaptation",
        "formula": "s = cos(proj(μ_p) + δ_r, proj(μ_g))",
        "desc": "Residual δ_r is optimised in the projected space before scoring. Penalises large deviations from the original probe projection.",
        "badge": "badge-cyan",
    },
    "mode2": {
        "name": "M2 – Latent Adaptation",
        "formula": "s = cos(proj(μ_p + δ_μ), proj(μ_g)) − λ·KL(q‖p)",
        "desc": "Residual δ_μ adapted in latent space. KL divergence from the original probe distribution N(μ_p, σ_p²) acts as regulariser to prevent the adapted vector from drifting too far.",
        "badge": "badge-amber",
    },
    "mode3": {
        "name": "M3 – Latent-mean ⭐",
        "formula": "s = cos(μ_p, μ_g)",
        "desc": "Direct cosine similarity between raw latent means. No projection head. Empirically most effective: highest Rank-1 and lowest EER across all backbone–dataset combinations.",
        "badge": "badge-emerald",
    },
    "model3": {
        "name": "M3 – Latent-mean ⭐",
        "formula": "s = cos(μ_p, μ_g)",
        "desc": "Direct cosine similarity between raw latent means. No projection head. Empirically most effective: highest Rank-1 and lowest EER across all backbone–dataset combinations.",
        "badge": "badge-emerald",
    },
    "mode4": {
        "name": "M4 – Latent-mu Adaptation",
        "formula": "s = cos(μ_p + δ_μ, μ_g) − λ·KL(q‖p)",
        "desc": "Like M2 but scoring directly in latent-mean space (without projection head). KL penalty keeps adapted vector close to original probe distribution.",
        "badge": "badge-pink",
    },
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
)


def _parse_mean(val):
    try:
        return float(str(val).split("±")[0].strip())
    except Exception:
        return float("nan")


@st.cache_data
def load_data():
    df = pd.read_csv(TASKS_CSV)
    numeric_cols = [c for c in df.columns if c not in ("Model", "Method", "Mode", "Train", "Test")]
    for col in numeric_cols:
        df[col + "_mean"] = df[col].astype(str).apply(_parse_mean)
    return df


def render():
    df = load_data()

    st.markdown("""
    <div class="page-hero">
        <div class="page-title">⚡ Inference Modes</div>
        <div class="page-subtitle">
            Deep-dive comparison of four scoring strategies (M0–M4) for matching probe <b>p</b>
            against gallery template <b>g</b>.<br>
            The decoder is <b>removed at inference time</b>; only the Encoder (μ, σ) is used.
            <b>Mode M3</b> (direct cosine on latent means) proves most effective across all settings.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode cards ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📖 Mode Descriptions</div>', unsafe_allow_html=True)
    unique_modes = df["Mode"].unique().tolist()
    mode_cols = st.columns(len(unique_modes))
    for col, mode in zip(mode_cols, unique_modes):
        info = MODE_INFO.get(mode, {"name": mode, "formula": "", "desc": "", "badge": "badge-violet"})
        col.markdown(f"""
        <div class="metric-card" style="height:100%;">
            <span class="badge {info['badge']}">{info['name'].split('–')[0].strip()}</span>
            <div style="font-size:0.85rem; font-weight:700; margin:0.6rem 0 0.3rem; color:var(--text-primary);">
                {info['name'].split('–')[1].strip() if '–' in info['name'] else info['name']}
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#a78bfa; margin-bottom:0.5rem; line-height:1.4;">
                {info['formula']}
            </div>
            <div style="font-size:0.75rem; color:var(--text-muted); line-height:1.5;">{info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔧 Filters</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_models = st.multiselect("Model", df["Model"].unique().tolist(),
                                    default=df["Model"].unique().tolist(), key="im_models")
    with fc2:
        sel_methods = st.multiselect("Method", df["Method"].unique().tolist(),
                                     default=df["Method"].unique().tolist(), key="im_methods")
    with fc3:
        sel_train = st.multiselect("Train Dataset", df["Train"].unique().tolist(),
                                   default=df["Train"].unique().tolist(), key="im_train")

    mask = (
        df["Model"].isin(sel_models) &
        df["Method"].isin(sel_methods) &
        df["Train"].isin(sel_train)
    )
    fdf = df[mask].copy()

    if fdf.empty:
        st.warning("No data for selected filters.")
        return

    # ── Metric selector ───────────────────────────────────────────────────────
    metric_opts = {
        "Open Rank-1 (%)": ("Open_Rank1_mean", True),
        "Open EER (%)": ("Open_EER_mean", False),
        "Open AUROC (%)": ("Open_AUROC_mean", True),
        "Open DIR@1 (%)": ("Open_DIR_1_mean", True),
        "Open OSCR (%)": ("Open_OSCR_mean", True),
        "Closed Rank-1 (%)": ("Closed_Rank1_mean", True),
        "Closed EER (%)": ("Closed_EER_mean", False),
    }

    mc, _ = st.columns([2, 3])
    with mc:
        sel_metric = st.selectbox("Metric to compare", list(metric_opts.keys()), key="im_metric")

    mcol, higher_better = metric_opts[sel_metric]

    # ── Line chart: modes vs metric per model ─────────────────────────────────
    st.markdown('<div class="section-header">📊 Mode Comparison</div>', unsafe_allow_html=True)

    tab_line, tab_group_bar, tab_delta = st.tabs([
        "📈 Line per Model",
        "📊 Grouped Bar",
        "📐 Δ vs M0 Baseline",
    ])

    mode_order = ["mode0", "model0", "mode1", "mode2", "mode3", "model3", "mode4"]
    mode_label_map = {m: MODE_INFO.get(m, {}).get("name", m).split("–")[0].strip() for m in mode_order}

    # Sort fdf mode column
    fdf["Mode_order"] = fdf["Mode"].map(lambda m: mode_order.index(m) if m in mode_order else 99)
    fdf = fdf.sort_values("Mode_order")

    COLORS = ["#7c3aed", "#06b6d4", "#ec4899", "#10b981", "#f59e0b", "#8b5cf6", "#14b8a6"]

    with tab_line:
        agg = fdf.groupby(["Model", "Mode"])[mcol].mean().reset_index()
        fig = go.Figure()
        for i, model in enumerate(agg["Model"].unique()):
            mdf = agg[agg["Model"] == model].sort_values("Mode_order" if "Mode_order" in agg.columns else "Mode")
            mode_labels = [mode_label_map.get(m, m) for m in mdf["Mode"].tolist()]
            fig.add_trace(go.Scatter(
                x=mode_labels,
                y=mdf[mcol].tolist(),
                mode="lines+markers",
                name=model.replace("unet_", ""),
                line=dict(color=COLORS[i % len(COLORS)], width=2.5),
                marker=dict(size=8, color=COLORS[i % len(COLORS)]),
            ))
        arrow = "↑ higher is better" if higher_better else "↓ lower is better"
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{sel_metric} by Inference Mode  ({arrow})",
            xaxis_title="Inference Mode",
            yaxis_title=sel_metric,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_group_bar:
        agg2 = fdf.groupby(["Model", "Mode"])[mcol].mean().reset_index()
        agg2["Mode_label"] = agg2["Mode"].map(lambda m: mode_label_map.get(m, m))
        agg2["Model_label"] = agg2["Model"].str.replace("unet_", "")

        fig2 = px.bar(
            agg2,
            x="Mode_label",
            y=mcol,
            color="Model_label",
            barmode="group",
            color_discrete_sequence=COLORS,
            labels={mcol: sel_metric, "Mode_label": "Mode", "Model_label": "Model"},
            title=f"Grouped: {sel_metric} per Mode",
            height=420,
        )
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    with tab_delta:
        # Compute delta vs M0
        base_modes = {"mode0", "model0"}
        m0_df = fdf[fdf["Mode"].isin(base_modes)].groupby("Model")[mcol].mean().rename("m0")
        other_df = fdf[~fdf["Mode"].isin(base_modes)].groupby(["Model", "Mode"])[mcol].mean().reset_index()
        merged = other_df.merge(m0_df, on="Model", how="left")
        merged["delta"] = merged[mcol] - merged["m0"]
        merged["Mode_label"] = merged["Mode"].map(lambda m: mode_label_map.get(m, m))
        merged["Model_label"] = merged["Model"].str.replace("unet_", "")

        if merged.empty:
            st.info("Insufficient data to compute Δ vs M0.")
        else:
            fig3 = px.bar(
                merged,
                x="Mode_label",
                y="delta",
                color="Model_label",
                barmode="group",
                color_discrete_sequence=COLORS,
                labels={"delta": f"Δ {sel_metric} vs M0", "Mode_label": "Mode", "Model_label": "Model"},
                title=f"Improvement over M0 Baseline  (Δ {sel_metric})",
                height=420,
            )
            fig3.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
            fig3.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

            # Best mode summary
            best_mode = merged.loc[merged["delta"].idxmax()]
            st.markdown(f"""
            <div class="highlight-best">
                🏆 Best improvement: <b>{best_mode['Mode_label']}</b> on <b>{best_mode['Model_label']}</b>
                &nbsp;|&nbsp; Δ = <b>{best_mode['delta']:+.2f}%</b> vs M0
            </div>
            """, unsafe_allow_html=True)

    # ── Per-dataset breakdown ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗂️ Per-Dataset Breakdown</div>', unsafe_allow_html=True)
    pivot = fdf.groupby(["Model", "Mode", "Train", "Test"])[mcol].mean().reset_index()
    pivot["Config"] = pivot["Train"] + "→" + pivot["Test"]
    pivot["Model_label"] = pivot["Model"].str.replace("unet_", "")
    pivot["Mode_label"] = pivot["Mode"].map(lambda m: mode_label_map.get(m, m))

    fig4 = px.scatter(
        pivot,
        x="Mode_label",
        y=mcol,
        color="Model_label",
        symbol="Config",
        size_max=14,
        color_discrete_sequence=COLORS,
        labels={mcol: sel_metric, "Mode_label": "Mode", "Model_label": "Model", "Config": "Train→Test"},
        title=f"{sel_metric} – per Dataset Configuration",
        height=450,
    )
    fig4.update_traces(marker=dict(size=10, opacity=0.85))
    fig4.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig4, use_container_width=True)
