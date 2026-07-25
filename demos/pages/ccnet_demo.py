"""CCNet Demo page - interactive showcase of the CCNet backbone encoder."""

import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

TASKS_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "tasks", "summary_table.csv")
LOGS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "logs")

MODEL_ID = "unet_ccnet"

MODE_LABELS = {
    "mode0":  "M0 - Proj-mean",
    "model0": "M0 - Proj-mean",
    "mode1":  "M1 - Proj-adapt",
    "mode2":  "M2 - Lat-adapt",
    "mode3":  "M3 - Lat-mean",
    "model3": "M3 - Lat-mean",
    "mode4":  "M4 - Lat-mu-adapt",
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

CCNET_CYAN    = "#06b6d4"
CCNET_VIOLET  = "#7c3aed"
CCNET_EMERALD = "#10b981"
COLORS = [CCNET_CYAN, CCNET_VIOLET, CCNET_EMERALD, "#ec4899", "#f59e0b"]


def _parse_mean(val) -> float:
    try:
        return float(str(val).split("+-")[0].split("\u00b1")[0].strip())
    except Exception:
        return float("nan")


@st.cache_data
def load_all_data():
    if not os.path.exists(TASKS_CSV):
        return pd.DataFrame()
    df = pd.read_csv(TASKS_CSV)
    numeric_cols = [c for c in df.columns if c not in ("Model", "Method", "Mode", "Train", "Test")]
    for col in numeric_cols:
        df[col + "_mean"] = df[col].astype(str).apply(_parse_mean)
    df["Mode_Label"] = df["Mode"].map(MODE_LABELS).fillna(df["Mode"])
    return df


@st.cache_data
def load_ccnet_data():
    df = load_all_data()
    if df.empty:
        return df
    return df[df["Model"] == MODEL_ID].copy()


def _find_ccnet_viz_dirs():
    results = {}
    for root, dirs, files in os.walk(LOGS_ROOT):
        rel = os.path.relpath(root, LOGS_ROOT)
        if "ccnet" in rel.lower() and any(f.endswith(".png") for f in files):
            results[rel] = root
    return results


def render():
    ccnet_df = load_ccnet_data()

    # Hero
    st.markdown("""
    <div class="page-hero">
        <div class="page-title">\U0001f52c CCNet Demo</div>
        <div class="page-subtitle">
            Interactive showcase of the <b>CCNet (Criss-Cross Network)</b> backbone encoder
            within the PALM Probabilistic Palmprint framework.<br>
            CCNet captures <b>long-range dependencies</b> via recurrent criss-cross attention
            over the full spatial extent of the feature map.
        </div>
        <div style="margin-top:1rem; display:flex; gap:8px; flex-wrap:wrap;">
            <span class="badge badge-cyan">CCNet Backbone</span>
            <span class="badge badge-violet">Criss-Cross Attention</span>
            <span class="badge badge-emerald">Global Context</span>
            <span class="badge badge-amber">Experimental</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Architecture
    st.markdown('<div class="section-header">\U0001f3d7\ufe0f CCNet Architecture</div>', unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns(3)
    arch_cards = [
        ("\U0001f500", "Criss-Cross Attention", "badge-cyan",
         "Captures horizontal and vertical long-range dependencies simultaneously. "
         "O(H+W) complexity per pixel vs O(HxW) for full self-attention."),
        ("\U0001f501", "Recurrent CCA (RCCA)", "badge-violet",
         "Two stacked CCA modules share weights. After two recurrent passes every position "
         "can attend to every other position via row x column intersection."),
        ("\U0001f310", "Context Aggregation Head", "badge-emerald",
         "Final context feature map concatenated with original backbone features before "
         "the probabilistic head (mu, log sigma^2). Rich spatial priors for uncertainty."),
    ]
    for col, (icon, title, badge, desc) in zip([ac1, ac2, ac3], arch_cards):
        col.markdown(
            "<div class='metric-card' style='padding:1.2rem; height:100%;'>"
            "<div style='font-size:1.8rem; margin-bottom:0.5rem;'>" + icon + "</div>"
            "<span class='badge " + badge + "' style='margin-bottom:0.5rem; display:inline-block;'>" + title + "</span>"
            "<div style='font-size:0.75rem; color:var(--text-muted); line-height:1.6; margin-top:0.4rem;'>" + desc + "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with st.expander("\U0001f4d0 Criss-Cross Attention - How it works"):
        st.markdown("""
        <div class="info-box" style="font-size:0.82rem; line-height:2;">
            <b>Input feature map:</b>  X in R^(B x C x H x W)<br>
            <b>Step 1 - Horizontal pass:</b>  Each pixel (i,j) attends to all pixels in row i<br>
            <b>Step 2 - Vertical pass:</b>  Each pixel (i,j) attends to all pixels in column j<br>
            <b>Result:</b>  After 2 recurrent passes, every pixel has seen every other pixel<br><br>
            Complexity: <b>O(H x W x (H+W))</b> vs O((H x W)^2) for full attention
        </div>
        """, unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="section-header">\U0001f4ca CCNet Performance Metrics</div>', unsafe_allow_html=True)

    if ccnet_df.empty:
        st.info("No CCNet data found in tasks/summary_table.csv. Run the evaluation pipeline first.")
    else:
        best_rank1 = ccnet_df["Open_Rank1_mean"].max() if "Open_Rank1_mean" in ccnet_df.columns else float("nan")
        best_eer   = ccnet_df["Open_EER_mean"].min()   if "Open_EER_mean"   in ccnet_df.columns else float("nan")
        best_auroc = ccnet_df["Open_AUROC_mean"].max() if "Open_AUROC_mean" in ccnet_df.columns else float("nan")
        best_oscr  = ccnet_df["Open_OSCR_mean"].max()  if "Open_OSCR_mean"  in ccnet_df.columns else float("nan")

        k1, k2, k3, k4 = st.columns(4)
        for col, label, value, fmt, variant in [
            (k1, "Best Open Rank-1", best_rank1, "{:.1f}%", ""),
            (k2, "Best Open EER",    best_eer,   "{:.2f}%", "warm"),
            (k3, "Best AUROC",       best_auroc, "{:.1f}%", "cool"),
            (k4, "Best OSCR",        best_oscr,  "{:.1f}%", ""),
        ]:
            import math
            display = fmt.format(value) if not (isinstance(value, float) and math.isnan(value)) else "N/A"
            col.markdown(
                "<div class='metric-card " + variant + "'>"
                "<div class='metric-label'>" + label + "</div>"
                "<div class='metric-value'>" + display + "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        metric_opts = {
            "Open Rank-1 (%)":  ("Open_Rank1_mean",  True),
            "Open EER (%)":     ("Open_EER_mean",     False),
            "Open AUROC (%)":   ("Open_AUROC_mean",   True),
            "Open DIR@1 (%)":   ("Open_DIR_1_mean",   True),
            "Open OSCR (%)":    ("Open_OSCR_mean",    True),
            "Closed Rank-1 (%)":("Closed_Rank1_mean", True),
            "Closed EER (%)":   ("Closed_EER_mean",   False),
        }

        tab_perf, tab_modes, tab_datasets, tab_compare = st.tabs([
            "\U0001f4c8 By Inference Mode",
            "\U0001f4ca Mode Breakdown",
            "\U0001f5c2\ufe0f By Dataset",
            "\U0001f198 vs Other Backbones",
        ])

        with tab_perf:
            mc, _ = st.columns([2, 3])
            with mc:
                sel_metric = st.selectbox("Metric", list(metric_opts.keys()), key="ccnet_metric_perf")
            mcol, higher_better = metric_opts[sel_metric]
            arrow = "higher is better" if higher_better else "lower is better"
            mode_order = ["mode0", "model0", "mode1", "mode2", "mode3", "model3", "mode4"]
            ccnet_local = ccnet_df.copy()
            ccnet_local["Mode_order"] = ccnet_local["Mode"].map(
                lambda m: mode_order.index(m) if m in mode_order else 99
            )
            if mcol in ccnet_local.columns:
                agg = (
                    ccnet_local.groupby(["Mode_Label", "Mode_order"])[mcol]
                    .mean().reset_index().sort_values("Mode_order")
                )
                if not agg[mcol].isna().all():
                    fig_perf = go.Figure(go.Bar(
                        x=agg["Mode_Label"].tolist(),
                        y=agg[mcol].tolist(),
                        marker=dict(
                            color=agg[mcol].tolist(),
                            colorscale=[[0, "#1e293b"], [1, CCNET_CYAN]],
                            line=dict(color="rgba(255,255,255,0.1)", width=1),
                        ),
                        text=["{:.2f}%".format(v) for v in agg[mcol].tolist()],
                        textposition="outside",
                        textfont=dict(color="#f0f4ff", size=11),
                    ))
                    fig_perf.update_layout(
                        **PLOTLY_LAYOUT,
                        title="CCNet - {} ({})".format(sel_metric, arrow),
                        xaxis_title="Inference Mode",
                        yaxis_title=sel_metric,
                        height=420,
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)
                    best_idx = agg[mcol].idxmax() if higher_better else agg[mcol].idxmin()
                    best_row = agg.loc[best_idx]
                    st.markdown(
                        "<div class='highlight-best'>"
                        "\U0001f3c6 Best mode for <b>{}</b>: <b>{}</b> | Score = <b>{:.2f}%</b>"
                        "</div>".format(sel_metric, best_row["Mode_Label"], best_row[mcol]),
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No data for this metric.")

        with tab_modes:
            sel_m2 = st.selectbox("Metric", list(metric_opts.keys()), key="ccnet_metric_modes")
            mcol2, _ = metric_opts[sel_m2]
            if mcol2 in ccnet_df.columns:
                pivot_df = ccnet_df.groupby(["Mode_Label", "Train", "Test"])[mcol2].mean().reset_index()
                pivot_df["Config"] = pivot_df["Train"] + " -> " + pivot_df["Test"]
                fig_modes = px.bar(
                    pivot_df, x="Mode_Label", y=mcol2, color="Config",
                    barmode="group", color_discrete_sequence=COLORS,
                    labels={mcol2: sel_m2, "Mode_Label": "Inference Mode"},
                    title="CCNet - {} by Mode x Dataset Config".format(sel_m2), height=440,
                )
                fig_modes.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_modes, use_container_width=True)
            else:
                st.info("Metric not available.")

        with tab_datasets:
            sel_m3 = st.selectbox("Metric", list(metric_opts.keys()), key="ccnet_metric_ds")
            mcol3, _ = metric_opts[sel_m3]
            if mcol3 in ccnet_df.columns:
                ds_df = ccnet_df.copy()
                ds_df["Config"] = ds_df["Train"] + " -> " + ds_df["Test"]
                ds_agg = ds_df.groupby(["Config", "Mode_Label"])[mcol3].mean().reset_index()
                fig_ds = px.bar(
                    ds_agg, x="Config", y=mcol3, color="Mode_Label",
                    barmode="group", color_discrete_sequence=COLORS,
                    labels={mcol3: sel_m3, "Config": "Train -> Test"},
                    title="CCNet - {} by Dataset Config".format(sel_m3), height=440,
                )
                fig_ds.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_ds, use_container_width=True)
            else:
                st.info("Metric not available.")

        with tab_compare:
            all_df = load_all_data()
            if all_df.empty:
                st.info("No data found for comparison.")
            else:
                sel_m4 = st.selectbox("Metric", list(metric_opts.keys()), key="ccnet_metric_cmp")
                mcol4, higher_better4 = metric_opts[sel_m4]
                if mcol4 in all_df.columns:
                    cmp_df = all_df.groupby(["Model", "Mode_Label"])[mcol4].mean().reset_index()
                    cmp_df["Model_Label"] = cmp_df["Model"].map({
                        "unet_ccnet": "CCNet", "unet_resnet": "ResNet18", "unet_palmnet": "PalmNet"
                    }).fillna(cmp_df["Model"])
                    fig_cmp = px.bar(
                        cmp_df, x="Mode_Label", y=mcol4, color="Model_Label",
                        barmode="group",
                        color_discrete_map={"CCNet": CCNET_CYAN, "ResNet18": CCNET_VIOLET, "PalmNet": "#ec4899"},
                        labels={mcol4: sel_m4, "Mode_Label": "Inference Mode"},
                        title="CCNet vs All Backbones - {}".format(sel_m4), height=460,
                    )
                    fig_cmp.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig_cmp, use_container_width=True)

                    ccnet_vals = cmp_df[cmp_df["Model_Label"] == "CCNet"].set_index("Mode_Label")[mcol4]
                    other_vals = cmp_df[cmp_df["Model_Label"] != "CCNet"].groupby("Mode_Label")[mcol4].max()
                    delta = (ccnet_vals - other_vals).dropna().reset_index()
                    delta.columns = ["Mode", "Delta"]
                    if not delta.empty:
                        fig_delta = go.Figure(go.Bar(
                            x=delta["Mode"].tolist(),
                            y=delta["Delta"].tolist(),
                            marker=dict(
                                color=[CCNET_EMERALD if v >= 0 else "#ef4444" for v in delta["Delta"].tolist()],
                                line=dict(color="rgba(255,255,255,0.1)", width=1),
                            ),
                            text=["{:+.2f}%".format(v) for v in delta["Delta"].tolist()],
                            textposition="outside",
                            textfont=dict(color="#f0f4ff", size=11),
                        ))
                        fig_delta.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
                        fig_delta.update_layout(
                            **PLOTLY_LAYOUT,
                            title="CCNet Delta vs Best Competitor - {}".format(sel_m4),
                            xaxis_title="Inference Mode",
                            yaxis_title="Delta {}".format(sel_m4),
                            height=380,
                        )
                        st.plotly_chart(fig_delta, use_container_width=True)

    # Viz Gallery
    st.markdown('<div class="section-header">\U0001f5bc\ufe0f Visualisation Gallery</div>', unsafe_allow_html=True)
    viz_dirs = _find_ccnet_viz_dirs()
    if viz_dirs:
        sel_run = st.selectbox("Select eval run", list(viz_dirs.keys()), key="ccnet_viz_run")
        run_path = viz_dirs[sel_run]
        png_files = sorted(glob.glob(os.path.join(run_path, "*.png")))
        if png_files:
            img_cols = st.columns(2)
            for i, png in enumerate(png_files):
                with img_cols[i % 2]:
                    try:
                        from PIL import Image as PILImage
                        st.image(PILImage.open(png), caption=os.path.basename(png), use_container_width=True)
                    except Exception:
                        st.image(png, use_container_width=True)
        else:
            st.info("No PNG images found in this run directory.")
    else:
        st.info(
            "No CCNet visualisation images found yet. "
            "Run evaluation with --model unet_ccnet to generate score/uncertainty distributions and t-SNE plots."
        )

    # Strengths & Limitations
    st.markdown('<div class="section-header">\u2696\ufe0f CCNet - Strengths vs Limitations</div>', unsafe_allow_html=True)
    sl1, sl2 = st.columns(2)
    with sl1:
        st.markdown("<div style='font-weight:700; color:#6ee7b7; margin-bottom:0.6rem;'>\u2705 Strengths</div>", unsafe_allow_html=True)
        for title, desc in [
            ("Global Context",
             "Every feature pixel attends to the full row + column in just 2 RCCA passes. "
             "Captures long-range ridge continuity and structural coherence."),
            ("Efficient Attention",
             "O(H+W) complexity per pixel vs O(H*W) for full self-attention. "
             "Practical for 128x128 or 224x224 palm ROI crops."),
            ("Weight Sharing",
             "Recurrent passes share weights - global coverage with fewer parameters "
             "compared to stacking multiple full attention layers."),
        ]:
            st.markdown(
                "<div class='metric-card' style='padding:0.9rem 1rem; margin-bottom:0.5rem;'>"
                "<div style='font-weight:600; font-size:0.85rem; margin-bottom:0.3rem;'>" + title + "</div>"
                "<div style='font-size:0.73rem; color:var(--text-muted); line-height:1.55;'>" + desc + "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    with sl2:
        st.markdown("<div style='font-weight:700; color:#f9a8d4; margin-bottom:0.6rem;'>\u26a0\ufe0f Limitations</div>", unsafe_allow_html=True)
        for title, desc in [
            ("Compute Cost",
             "Training is noticeably slower than ResNet18 at equivalent batch sizes "
             "due to the attention operations."),
            ("Local Texture Dominance",
             "Palmprint identity is primarily encoded in local ridge micro-texture. "
             "Global attention provides only marginal gains on this domain."),
            ("Marginal Gain in Practice",
             "CCNet is competitive but rarely outperforms the simpler ResNet18 backbone "
             "on palmprint benchmarks. Worth investigating for larger datasets."),
        ]:
            st.markdown(
                "<div class='metric-card' style='padding:0.9rem 1rem; margin-bottom:0.5rem;'>"
                "<div style='font-weight:600; font-size:0.85rem; margin-bottom:0.3rem;'>" + title + "</div>"
                "<div style='font-size:0.73rem; color:var(--text-muted); line-height:1.55;'>" + desc + "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    with st.expander("\u2699\ufe0f CCNet Training Config Reference"):
        st.markdown("""
        | Parameter | Recommended Value | Notes |
        |-----------|-----------------|-------|
        | `backbone` | `ccnet` | Selects the CrissCrossEncoder |
        | `latent_dim` | `256` | Same as ResNet18 baseline |
        | `proj_dim` | `512` | ArcFace projection head |
        | `batch_size` | `32-64` | Reduce if OOM due to attention |
        | `lr` | `1e-4` | Adam / AdamW |
        | `rcca_steps` | `2` | Number of recurrent CCA passes |
        | `dropout` | `0.1` | Applied after RCCA module |
        """)

    st.markdown('<div class="section-header">\U0001f680 Run CCNet Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box' style='font-size:0.82rem;'>"
        "Run the following command from the project root to evaluate the CCNet model:"
        "</div>",
        unsafe_allow_html=True,
    )
    st.code("python evaluate.py --model unet_ccnet --mode mode3 --dataset own", language="bash")
