"""Results Overview page – tables, bar charts, heatmaps for all experiments."""

import os
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Data loading ──────────────────────────────────────────────────────────────
TASKS_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "tasks", "summary_table.csv")
EXP_BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "baseline")

PLOT_COLORS = {
    "unet_ccnet":   "#7c3aed",
    "unet_resnet":  "#06b6d4",
    "unet_palmnet": "#ec4899",
}

MODE_LABELS = {
    "mode0": "M0 – Proj-mean",
    "model0": "M0 – Proj-mean",
    "mode1": "M1 – Proj-adapt",
    "mode2": "M2 – Lat-adapt",
    "mode3": "M3 – Lat-mean",
    "model3": "M3 – Lat-mean",
    "mode4": "M4 – Lat-mu-adapt",
}

MODEL_LABELS = {
    "unet_ccnet":   "CCNet",
    "unet_resnet":  "ResNet18",
    "unet_palmnet": "PalmNet",
}


def _parse_mean(val: str) -> float:
    """Extract mean from '88.09 ± 1.40' strings."""
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
    df["Model_Label"] = df["Model"].map(MODEL_LABELS).fillna(df["Model"])
    df["Mode_Label"] = df["Mode"].map(MODE_LABELS).fillna(df["Mode"])
    return df


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
)


def render():
    df = load_data()

    st.markdown("""
    <div class="page-hero">
        <div class="page-title">📊 Results Overview</div>
        <div class="page-subtitle">
            Comprehensive evaluation across models, datasets, and inference modes.<br>
            <b>Mode M3 (latent-mean matching)</b> proves most effective — highest Open Rank-1, lowest EER.
            All thresholds (τ_U, τ_S, τ_K) are calibrated on the <b>validation split only</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        sel_models = st.multiselect("Model", df["Model_Label"].unique().tolist(),
                                    default=df["Model_Label"].unique().tolist(), key="ro_models")
    with fc2:
        sel_methods = st.multiselect("Method", df["Method"].unique().tolist(),
                                     default=df["Method"].unique().tolist(), key="ro_methods")
    with fc3:
        sel_train = st.multiselect("Train Dataset", df["Train"].unique().tolist(),
                                   default=df["Train"].unique().tolist(), key="ro_train")
    with fc4:
        sel_test = st.multiselect("Test Dataset", df["Test"].unique().tolist(),
                                  default=df["Test"].unique().tolist(), key="ro_test")

    mask = (
        df["Model_Label"].isin(sel_models) &
        df["Method"].isin(sel_methods) &
        df["Train"].isin(sel_train) &
        df["Test"].isin(sel_test)
    )
    fdf = df[mask].copy()

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return

    # ── Summary KPI row ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📌 Aggregate Metrics</div>', unsafe_allow_html=True)
    best_rank1  = fdf["Open_Rank1_mean"].max()
    best_eer    = fdf["Open_EER_mean"].min()
    best_auroc  = fdf["Open_AUROC_mean"].max()
    best_oscr   = fdf["Open_OSCR_mean"].max()

    k1, k2, k3, k4 = st.columns(4)
    def kpi(col, label, value, fmt, variant=""):
        col.markdown(f"""
        <div class="metric-card {variant}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{fmt.format(value)}</div>
        </div>
        """, unsafe_allow_html=True)

    kpi(k1, "Best Open Rank-1 ↑",  best_rank1, "{:.1f}%")
    kpi(k2, "Best Open EER ↓",     best_eer,   "{:.2f}%", "warm")
    kpi(k3, "Best AUROC ↑",        best_auroc, "{:.1f}%", "cool")
    kpi(k4, "Best OSCR ↑",         best_oscr,  "{:.1f}%")

    st.markdown("""
    <div class="info-box" style="font-size:0.8rem;">
        📋 <b>Evaluation protocol:</b> All three open-set thresholds (τ<sub>U</sub>, τ<sub>S</sub>, τ<sub>K</sub>)
        are selected jointly on a <b>held-out validation split</b>, then applied unchanged to the test set.
        Open Rank-1 measures the fraction of enrolled probes correctly matched;
        EER is the operating point where FAR = FRR; OSCR combines identification and rejection performance.
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Table", "📊 Bar Charts", "🔥 Heatmap", "📈 Radar"])

    # ── Tab 1: Table ──────────────────────────────────────────────────────────
    with tab1:
        show_cols = ["Model_Label", "Method", "Mode_Label", "Train", "Test",
                     "Closed_Rank1", "Closed_EER",
                     "Open_Rank1", "Open_EER", "Open_AUROC", "Open_DIR_1", "Open_OSCR"]
        show_cols = [c for c in show_cols if c in fdf.columns]
        rename = {"Model_Label": "Model", "Mode_Label": "Mode"}
        disp = fdf[show_cols].rename(columns=rename)
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
        )

        # Download button
        csv_data = fdf.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered data", csv_data, "filtered_results.csv", "text/csv")

    # ── Tab 2: Bar charts ─────────────────────────────────────────────────────
    with tab2:
        metric_opts = {
            "Open Rank-1 (%)": "Open_Rank1_mean",
            "Open EER (%)": "Open_EER_mean",
            "Open AUROC (%)": "Open_AUROC_mean",
            "Open DIR@1 (%)": "Open_DIR_1_mean",
            "Open OSCR (%)": "Open_OSCR_mean",
            "Closed Rank-1 (%)": "Closed_Rank1_mean",
            "Closed EER (%)": "Closed_EER_mean",
        }
        sel_metric = st.selectbox("Metric", list(metric_opts.keys()), key="bar_metric")
        mcol = metric_opts[sel_metric]

        group_key = f"Model_Label + Mode_Label + Train→Test"
        bar_df = fdf.copy()
        bar_df["Config"] = bar_df["Model_Label"] + " | " + bar_df["Mode_Label"] + "\n" + bar_df["Train"] + "→" + bar_df["Test"]

        bc1, bc2 = st.columns([3, 1])
        with bc2:
            group_by = st.selectbox("Group by", ["Model", "Mode", "Dataset"], key="bar_group")
            sort_desc = st.checkbox("Sort descending", value=True, key="bar_sort")

        with bc1:
            if group_by == "Model":
                x_col = "Config"
                color_col = "Model_Label"
            elif group_by == "Mode":
                x_col = "Config"
                color_col = "Mode_Label"
            else:
                x_col = "Config"
                color_col = "Train"

            plot_df = bar_df.dropna(subset=[mcol]).sort_values(mcol, ascending=not sort_desc)
            fig = px.bar(
                plot_df,
                x=x_col,
                y=mcol,
                color=color_col,
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={mcol: sel_metric, x_col: ""},
                title=f"{sel_metric} by Configuration",
                height=480,
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Heatmap ────────────────────────────────────────────────────────
    with tab3:
        hm_opts = {
            "Open Rank-1 (%)": "Open_Rank1_mean",
            "Open EER (%)": "Open_EER_mean",
            "Open AUROC (%)": "Open_AUROC_mean",
            "Open OSCR (%)": "Open_OSCR_mean",
        }
        hcol1, hcol2, hcol3 = st.columns([2, 2, 1])
        with hcol1:
            hm_metric = st.selectbox("Metric", list(hm_opts.keys()), key="hm_metric")
        with hcol2:
            hm_method = st.selectbox("Method", df["Method"].unique().tolist(), key="hm_method")

        mcol2 = hm_opts[hm_metric]
        hm_df = fdf[fdf["Method"] == hm_method].copy()
        hm_df["Train_Test"] = hm_df["Train"] + "→" + hm_df["Test"]

        if not hm_df.empty:
            pivot = hm_df.pivot_table(values=mcol2, index="Model_Label", columns="Train_Test", aggfunc="mean")
            fig2 = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="Viridis",
                text=[[f"{v:.1f}%" if not pd.isna(v) else "" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                showscale=True,
                colorbar=dict(title=hm_metric, tickfont=dict(color="#94a3b8")),
            ))
            fig2.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Heatmap: {hm_metric} | Method={hm_method}",
                xaxis_title="Train → Test",
                yaxis_title="Model",
                height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data for selected filters.")

    # ── Tab 4: Radar ──────────────────────────────────────────────────────────
    with tab4:
        radar_metrics = {
            "Open Rank-1": "Open_Rank1_mean",
            "1-Open EER": None,
            "AUROC": "Open_AUROC_mean",
            "DIR@1": "Open_DIR_1_mean",
            "OSCR": "Open_OSCR_mean",
            "Closed Rank-1": "Closed_Rank1_mean",
        }

        rc1, rc2 = st.columns(2)
        with rc1:
            radar_models = st.multiselect("Models to compare", fdf["Model_Label"].unique().tolist(),
                                          default=fdf["Model_Label"].unique().tolist()[:3], key="radar_models")
        with rc2:
            radar_mode = st.selectbox("Mode", fdf["Mode_Label"].unique().tolist(), key="radar_mode")

        radar_df = fdf[(fdf["Model_Label"].isin(radar_models)) & (fdf["Mode_Label"] == radar_mode)]
        agg = radar_df.groupby("Model_Label").mean(numeric_only=True).reset_index()

        categories = ["Open Rank-1", "Closed Rank-1", "AUROC", "DIR@1", "OSCR"]
        col_map = {
            "Open Rank-1": "Open_Rank1_mean",
            "Closed Rank-1": "Closed_Rank1_mean",
            "AUROC": "Open_AUROC_mean",
            "DIR@1": "Open_DIR_1_mean",
            "OSCR": "Open_OSCR_mean",
        }

        fig3 = go.Figure()
        colors = ["#7c3aed", "#06b6d4", "#ec4899", "#10b981", "#f59e0b"]
        for i, row in agg.iterrows():
            vals = [row.get(col_map[c], 0) for c in categories]
            vals.append(vals[0])
            fig3.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=colors[i % len(colors)].replace("#", "rgba(") + ",0.15)",
                line=dict(color=colors[i % len(colors)], width=2),
                name=row["Model_Label"],
            ))

        fig3.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#64748b")),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)"),
            ),
            title=f"Radar – {radar_mode}",
            height=450,
        )
        st.plotly_chart(fig3, use_container_width=True)
