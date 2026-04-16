from pathlib import Path
import sys

# Fix: Streamlit's file watcher tries to access torch.classes.__path__._path
# which triggers a PyTorch RuntimeError. Clearing the path prevents this.
try:
    import torch
    torch.classes.__path__ = []
except Exception:
    pass

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.hospital_config import HOSPITAL_CONFIGS
from dashboard.ui_core import apply_page_style


st.set_page_config(
    page_title="Federated Medical AI - Control Tower",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_history(checkpoint_path: Path):
    import pandas as pd
    import torch

    if not checkpoint_path.exists():
        return None
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history", [])
    if not history:
        return None
    return pd.DataFrame(history)


def _history_for_display(history_df):
    history_view = history_df.copy()
    if "per_hospital_metrics" in history_view.columns:
        history_view["per_hospital_metrics"] = history_view["per_hospital_metrics"].apply(
            lambda x: f"{len(x)} hospitals" if isinstance(x, list) else "-"
        )
    return history_view


def _latest_hospital_metrics(history_df):
    import pandas as pd

    if history_df is None or history_df.empty or "per_hospital_metrics" not in history_df.columns:
        return None
    latest = history_df.iloc[-1].get("per_hospital_metrics")
    if not isinstance(latest, list) or not latest:
        return None
    return pd.DataFrame(latest)


def _gauge_fig(value: float, label: str, colour: str):
    """Return a Plotly indicator/gauge for a single metric (0-1 scale)."""
    import plotly.graph_objects as go

    display = value if not (value != value) else 0.0  # handle NaN
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(display * 100, 2),
            title={"text": label, "font": {"color": "#cfd8dc", "size": 14}},
            number={"suffix": "%", "font": {"color": colour, "size": 22}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#cfd8dc"},
                "bar": {"color": colour},
                "bgcolor": "#1e2329",
                "bordercolor": "#2a3039",
                "steps": [
                    {"range": [0, 50], "color": "#1a2030"},
                    {"range": [50, 75], "color": "#1e2a38"},
                    {"range": [75, 100], "color": "#1e3040"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 2},
                    "thickness": 0.75,
                    "value": display * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cfd8dc"),
    )
    return fig


def _render_metric_charts(history_df):
    import plotly.express as px
    import plotly.graph_objects as go

    # ── Gauge row ──────────────────────────────────────────────────────────
    st.markdown("### Current Model Performance")
    latest = history_df.iloc[-1]

    auc_val = float(latest.get("avg_auc", 0) or 0)
    f1_val  = float(latest.get("avg_f1", 0) or 0)
    acc_val = float(latest.get("avg_accuracy", 0) or 0)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(_gauge_fig(auc_val, "Avg ROC-AUC", "#4fc3f7"), use_container_width=True)
    with g2:
        st.plotly_chart(_gauge_fig(f1_val, "Avg F1 Score", "#81d4fa"), use_container_width=True)
    with g3:
        st.plotly_chart(_gauge_fig(acc_val, "Avg Accuracy", "#b3e5fc"), use_container_width=True)

    st.markdown("### Training Progression")

    c1, c2 = st.columns(2)
    with c1:
        metric_cols = [c for c in ["avg_auc", "avg_f1", "avg_accuracy"] if c in history_df.columns]
        if metric_cols:
            plot_df = history_df[["round"] + metric_cols].copy()
            # Replace NaN with None so Plotly shows gaps rather than zero-lines
            for col in metric_cols:
                plot_df[col] = plot_df[col].where(plot_df[col].notna(), other=None)
            fig_metrics = px.line(
                plot_df,
                x="round",
                y=metric_cols,
                markers=True,
                title="Round-wise AUC / F1 / Accuracy",
                labels={"value": "Score", "variable": "Metric", "round": "FL Round"},
                color_discrete_map={
                    "avg_auc": "#4fc3f7",
                    "avg_f1": "#81d4fa",
                    "avg_accuracy": "#b3e5fc",
                },
            )
            fig_metrics.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,35,41,1)",
                font=dict(color="#cfd8dc"),
                legend_title_text="Metric",
                yaxis=dict(range=[0, 1.05], gridcolor="#2a3039"),
                xaxis=dict(gridcolor="#2a3039"),
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

    with c2:
        if "avg_val_loss" in history_df.columns:
            fig_loss = px.line(
                history_df,
                x="round",
                y="avg_val_loss",
                markers=True,
                title="Round-wise Validation Loss",
                labels={"avg_val_loss": "Val Loss", "round": "FL Round"},
                color_discrete_sequence=["#ef5350"],
            )
            fig_loss.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,35,41,1)",
                font=dict(color="#cfd8dc"),
                yaxis=dict(gridcolor="#2a3039"),
                xaxis=dict(gridcolor="#2a3039"),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    # ── Area chart: cumulative improvement ────────────────────────────────
    metric_cols_exist = [c for c in ["avg_auc", "avg_f1", "avg_accuracy"] if c in history_df.columns]
    if len(metric_cols_exist) >= 2 and len(history_df) > 1:
        st.markdown("### Metric Improvement Over Rounds")
        fig_area = go.Figure()
        colours = ["#4fc3f7", "#81d4fa", "#b3e5fc"]
        for col, colour in zip(metric_cols_exist, colours):
            clean = history_df[col].fillna(0)
            fig_area.add_trace(
                go.Scatter(
                    x=history_df["round"],
                    y=clean,
                    mode="lines+markers",
                    name=col.replace("avg_", "").upper(),
                    fill="tozeroy",
                    line=dict(color=colour, width=2),
                    fillcolor=colour.replace(")", ",0.12)").replace("rgb", "rgba")
                    if colour.startswith("rgb")
                    else colour + "1e",
                    opacity=0.6,
                )
            )
        fig_area.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,35,41,1)",
            font=dict(color="#cfd8dc"),
            yaxis=dict(range=[0, 1.05], gridcolor="#2a3039", title="Score"),
            xaxis=dict(gridcolor="#2a3039", title="FL Round"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_area, use_container_width=True)

    # ── Per-hospital comparison ────────────────────────────────────────────
    latest_hospital_df = _latest_hospital_metrics(history_df)
    if latest_hospital_df is not None:
        st.markdown("### Latest Round — Hospital Comparison")
        c_bar, c_radar = st.columns(2)

        with c_bar:
            bar = px.bar(
                latest_hospital_df,
                x="hospital_id",
                y=["roc_auc_macro", "f1_macro", "accuracy"],
                barmode="group",
                title="Per-Hospital Metrics (Latest Round)",
                labels={"value": "Score", "variable": "Metric", "hospital_id": "Hospital"},
                color_discrete_map={
                    "roc_auc_macro": "#4fc3f7",
                    "f1_macro": "#81d4fa",
                    "accuracy": "#b3e5fc",
                },
            )
            bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,35,41,1)",
                font=dict(color="#cfd8dc"),
                legend_title_text="Metric",
                yaxis=dict(range=[0, 1.05], gridcolor="#2a3039"),
            )
            st.plotly_chart(bar, use_container_width=True)

        with c_radar:
            hospitals = latest_hospital_df["hospital_id"].tolist()
            metrics_radar = ["roc_auc_macro", "f1_macro", "accuracy"]
            radar_colours = ["#4fc3f7", "#81d4fa", "#b3e5fc"]
            # Pre-convert hex colours to rgba for scatterpolar fillcolor
            def _hex_rgba(h: str, alpha: float = 0.2) -> str:
                h = h.lstrip("#")
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"rgba({r},{g},{b},{alpha})"

            fig_radar = go.Figure()
            for hosp, colour in zip(hospitals, radar_colours):
                row = latest_hospital_df[latest_hospital_df["hospital_id"] == hosp].iloc[0]
                values = [float(row.get(m, 0) or 0) for m in metrics_radar]
                values_closed = values + [values[0]]
                cats_closed = metrics_radar + [metrics_radar[0]]
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=values_closed,
                        theta=cats_closed,
                        fill="toself",
                        name=hosp,
                        line=dict(color=colour),
                        fillcolor=_hex_rgba(colour, 0.2),
                    )
                )
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a3039", color="#cfd8dc"),
                    angularaxis=dict(color="#cfd8dc"),
                    bgcolor="#1e2329",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cfd8dc"),
                title="Hospital Radar (Latest Round)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("#### Per-Hospital Detail Table")
        st.dataframe(latest_hospital_df, use_container_width=True, hide_index=True)


def main():
    apply_page_style("#4fc3f7", "#81d4fa")
    st.markdown('<div class="main-header">Federated Medical AI Control Tower</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Central overview for the three hospital interfaces and federated checkpoints.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="welcome-card">Use the page navigator to open Hospital A, Hospital B, or Hospital C. '
        'This admin page summarizes the current hospital configurations and the latest federated checkpoint history.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for col, config in zip(cols, HOSPITAL_CONFIGS.values()):
        with col:
            st.markdown(f"### {config.title}")
            st.caption(config.subtitle)
            st.markdown(f"- Node id: `{config.hospital_id}`")
            st.markdown(f"- Default model: `{config.model_type}`")
            st.markdown(f"- Threshold: `{config.threshold:.2f}`")
            st.markdown(f"- Note weight: `{config.note_weight:.2f}`")
            st.markdown(f"- Checkpoint: `{Path(config.checkpoint_path).name}`")

    st.divider()
    hdr_col, btn_col, auto_col = st.columns([4, 1, 2])
    hdr_col.markdown("### Federated Model — Accuracy & Evaluation Metrics")
    refresh_clicked = btn_col.button("Refresh", use_container_width=True, type="primary")
    auto_refresh = auto_col.checkbox("Auto-refresh every 30 s", value=False)
    if refresh_clicked:
        st.rerun()
    if auto_refresh:
        import time as _time
        _time.sleep(30)
        st.rerun()

    checkpoint_path = ROOT / "models" / "federated" / "global_model_final.pth"
    history_df = _load_history(checkpoint_path)
    if history_df is None:
        st.info("No federated checkpoint history found yet. Run `python run_federated.py` to generate metrics.")
    else:
        latest = history_df.iloc[-1]

        # ── Summary KPI strip ─────────────────────────────────────────────
        auc_val = float(latest.get("avg_auc", 0) or 0)
        f1_val  = float(latest.get("avg_f1", 0) or 0)
        acc_val = float(latest.get("avg_accuracy", 0) or 0)
        loss_val = float(latest.get("avg_val_loss", 0) or 0)
        rounds_done = int(latest["round"])
        clients     = int(latest["n_clients"])
        total_samples = int(latest.get("total_samples", 0))

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("FL Rounds", rounds_done)
        k2.metric("Clients", clients)
        k3.metric("Total Samples", f"{total_samples:,}")
        k4.metric("Avg ROC-AUC", f"{auc_val:.4f}" if auc_val else "N/A")
        k5.metric("Avg F1", f"{f1_val:.4f}")
        k6.metric("Avg Accuracy", f"{acc_val:.4f}" if acc_val else "N/A")

        if loss_val:
            st.caption(f"Validation loss (latest round): **{loss_val:.4f}**")

        # ── Full visualizations ───────────────────────────────────────────
        _render_metric_charts(history_df)

        with st.expander("Training History Table", expanded=False):
            st.dataframe(_history_for_display(history_df), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### How to Use")
    st.markdown("- Start the app with `streamlit run dashboard/app.py`.")
    st.markdown("- Use the page selector in the left sidebar to open a hospital-specific interface.")
    st.markdown("- Each hospital page has its own branding, default model, threshold, and note weighting.")


main()
