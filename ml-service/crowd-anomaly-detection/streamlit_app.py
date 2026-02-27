"""
DrishtiX Crowd Anomaly Detection — Streamlit Application
=========================================================
Full-featured monitoring dashboard for crowd anomaly detection.

Pages:
  🏠  Home / Overview
  📊  Dataset Explorer
  🔧  Train Models
  🔍  Single Reading Predictor
  📂  Batch Prediction
  📈  Model Performance
  🗺️  Zone Live Simulation
  📋  Incident Log
  🎥  Visual Detection  (CLIP-based image/video crowd scene analysis)

Run:  streamlit run streamlit_app.py
"""

import io
import json
import logging
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DrishtiX · Crowd Anomaly Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Constants / Paths
# ──────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).parent
DATA_PATH    = APP_DIR / "data" / "crowd_telemetry_features.csv"
RAW_PATH     = APP_DIR / "data" / "crowd_telemetry.csv"
MODEL_DIR    = APP_DIR / "models"
REPORT_PATH  = MODEL_DIR / "training_report.json"

ANOMALY_COLORS = {
    "NORMAL":               "#2ecc71",
    "CROWD_SURGE":          "#e74c3c",
    "STAMPEDE_PRECURSOR":   "#8e44ad",
    "BOTTLENECK":           "#e67e22",
    "FLOW_REVERSAL":        "#3498db",
    "ISOLATION_ZONE":       "#1abc9c",
    "SUSTAINED_OVERLOAD":   "#c0392b",
    "ANOMALY_UNCLASSIFIED": "#95a5a6",
}

SEVERITY_COLORS = {
    "NORMAL":   "#2ecc71",
    "LOW":      "#f1c40f",
    "MEDIUM":   "#e67e22",
    "HIGH":     "#e74c3c",
    "CRITICAL": "#7b0d1e",
}

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .drishtix-header {
        background: linear-gradient(135deg, #0a0e1a 0%, #112240 50%, #1a3a5c 100%);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 20px;
        color: white;
    }
    .drishtix-header h1 { margin: 0; font-size: 2rem; }
    .drishtix-header p  { margin: 4px 0 0; opacity: 0.75; font-size: 0.95rem; }

    /* Alert severity badges */
    .badge-normal   { background:#2ecc71; color:white; padding:3px 10px; border-radius:12px; font-size:.75rem; font-weight:600; }
    .badge-low      { background:#f1c40f; color:#333;  padding:3px 10px; border-radius:12px; font-size:.75rem; font-weight:600; }
    .badge-medium   { background:#e67e22; color:white; padding:3px 10px; border-radius:12px; font-size:.75rem; font-weight:600; }
    .badge-high     { background:#e74c3c; color:white; padding:3px 10px; border-radius:12px; font-size:.75rem; font-weight:600; }
    .badge-critical { background:#7b0d1e; color:white; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:700; }

    /* Metric cards */
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #2d2d3f;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0d1117; }
    section[data-testid="stSidebar"] .stMarkdown { color: #8b949e; }

    /* Tables */
    .stDataFrame { font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Cached data + model loaders
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_resource(show_spinner=False)
def load_detector(model_name: str = "best_model"):
    try:
        sys.path.insert(0, str(APP_DIR))
        from anomaly_detector import CrowdAnomalyDetector
        det = CrowdAnomalyDetector.from_trained(model_name, MODEL_DIR)
        return det
    except Exception as e:
        st.warning(f"Could not load detector: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_report() -> dict | None:
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0">
        <span style="font-size:2.5rem;">👁️</span>
        <h2 style="color:#58a6ff; margin:4px 0">DrishtiX</h2>
        <p style="color:#8b949e; margin:0; font-size:0.8rem">Crowd Anomaly Detection</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "🏠  Overview",
            "📊  Dataset Explorer",
            "🔧  Train Models",
            "🔍  Single Reading Predictor",
            "📂  Batch Prediction",
            "📈  Model Performance",
            "🗺️  Zone Simulation",
            "📋  Incident Log",
            "🎥  Visual Detection",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Quick Actions**")
    if st.button("⚡ Generate Dataset", use_container_width=True):
        with st.spinner("Generating synthetic crowd data…"):
            result = subprocess.run(
                [sys.executable, str(APP_DIR / "generate_dataset.py")],
                capture_output=True, text=True, cwd=str(APP_DIR)
            )
        if result.returncode == 0:
            st.success("Dataset generated!")
            load_csv.clear()
        else:
            st.error(f"Error: {result.stderr[-500:]}")

    if st.button("🤖 Train Models", use_container_width=True):
        with st.spinner("Training models… (this may take 1-3 min)"):
            result = subprocess.run(
                [sys.executable, str(APP_DIR / "train_model.py")],
                capture_output=True, text=True, cwd=str(APP_DIR)
            )
        if result.returncode == 0:
            st.success("Models trained!")
            load_detector.clear()
            load_report.clear()
        else:
            st.error(f"Error: {result.stderr[-500:]}")

    st.divider()
    # Status indicators
    data_ok  = DATA_PATH.exists()
    model_ok = (MODEL_DIR / "best_model.pkl").exists()
    st.markdown(
        f"{'✅' if data_ok else '❌'} Dataset  \n"
        f"{'✅' if model_ok else '❌'} Trained Model"
    )


# ──────────────────────────────────────────────────────────────
# ── PAGE: Overview ────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
if "Overview" in page:
    st.markdown("""
    <div class="drishtix-header">
        <h1>👁️ DrishtiX · Crowd Anomaly Detection</h1>
        <p>Isolation Forest–based real-time crowd safety monitoring for live events</p>
    </div>
    """, unsafe_allow_html=True)

    feat_df = load_csv(DATA_PATH)

    # Top KPI row
    if feat_df is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Readings",   f"{len(feat_df):,}")
        with col2:
            n_anom = feat_df["is_anomaly"].sum()
            st.metric("Anomalies",        f"{n_anom:,}",
                      delta=f"{n_anom/len(feat_df)*100:.1f}%")
        with col3:
            n_events = feat_df["event_id"].nunique()
            st.metric("Events Simulated", str(n_events))
        with col4:
            n_zones  = feat_df["zone_id"].nunique()
            st.metric("Zones Monitored",  str(n_zones))
        with col5:
            peak_d = feat_df["crowd_density"].max()
            st.metric("Peak Density",     f"{peak_d:.3f}")

        st.divider()

        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.subheader("Crowd Density Over Time (all zones)")
            sample = feat_df[feat_df["event_id"] == feat_df["event_id"].unique()[0]].copy()
            sample["timestamp"] = pd.to_datetime(sample["timestamp"])
            sample_agg = (
                sample.groupby("timestamp")["crowd_density"]
                .mean().reset_index()
            )
            anomalies_ts = sample[sample["is_anomaly"] == 1]
            fig = go.Figure()
            fig.add_scatter(
                x=sample_agg["timestamp"], y=sample_agg["crowd_density"],
                mode="lines", name="Avg Density",
                line=dict(color="#3498db", width=2),
            )
            if not anomalies_ts.empty:
                anom_agg = anomalies_ts.groupby("timestamp")["crowd_density"].mean().reset_index()
                fig.add_scatter(
                    x=anom_agg["timestamp"], y=anom_agg["crowd_density"],
                    mode="markers", name="Anomaly",
                    marker=dict(color="#e74c3c", size=7, symbol="x"),
                )
            fig.update_layout(
                template="plotly_dark", height=280, margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="Time", yaxis_title="Density",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("Anomaly Type Distribution")
            anom_only = feat_df[feat_df["is_anomaly"] == 1]
            if not anom_only.empty:
                counts = anom_only["anomaly_label"].value_counts().reset_index()
                counts.columns = ["Type", "Count"]
                colors = [ANOMALY_COLORS.get(t, "#95a5a6") for t in counts["Type"]]
                fig2 = px.bar(counts, x="Count", y="Type", orientation="h",
                              color="Type",
                              color_discrete_map=ANOMALY_COLORS,
                              template="plotly_dark")
                fig2.update_layout(
                    height=280, showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    yaxis_title="", xaxis_title="Count",
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Architecture — How It Works")
        st.markdown("""
        ```
        ┌──────────────────────────────────────────────────────────────────┐
        │                    DrishtiX Anomaly Pipeline                     │
        │                                                                  │
        │  Camera / IoT MQTT → Crowd Density Calculator                    │
        │           ↓                                                      │
        │  Zone Telemetry (density, flow, temp, noise …)                   │
        │           ↓                                                      │
        │  Feature Engineering (rolling stats, z-scores, deltas)           │
        │           ↓                                                      │
        │  ┌── RobustScaler ──── Isolation Forest ──┐                     │
        │  │                                         │ → Anomaly Score    │
        │  │  (trained on NORMAL-only crowd data)    │ → Severity         │
        │  └─────────────────────────────────────────┘ → Type             │
        │           ↓                                                      │
        │  Heuristic Rule Layer (business logic classification)            │
        │           ↓                                                      │
        │  Alert  │  Dashboard  │  DrishtiX REST API  │  SNS Push         │
        └──────────────────────────────────────────────────────────────────┘
        ```
        """)
    else:
        st.info("No dataset found. Click **⚡ Generate Dataset** in the sidebar to begin.")


# ──────────────────────────────────────────────────────────────
# ── PAGE: Dataset Explorer ────────────────────────────────────
# ──────────────────────────────────────────────────────────────
elif "Dataset" in page:
    st.title("📊 Dataset Explorer")
    feat_df = load_csv(DATA_PATH)

    if feat_df is None:
        st.warning("No dataset. Generate it first with the sidebar button.")
        st.stop()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sel_events = st.multiselect("Event ID", feat_df["event_id"].unique(),
                                    default=list(feat_df["event_id"].unique()[:3]))
    with col_f2:
        sel_zones  = st.multiselect("Zone",  feat_df["zone_id"].unique(),
                                    default=list(feat_df["zone_id"].unique()[:3]))
    with col_f3:
        show_anom  = st.selectbox("Show", ["All", "Anomalies only", "Normal only"])

    filtered = feat_df.copy()
    if sel_events:
        filtered = filtered[filtered["event_id"].isin(sel_events)]
    if sel_zones:
        filtered = filtered[filtered["zone_id"].isin(sel_zones)]
    if show_anom == "Anomalies only":
        filtered = filtered[filtered["is_anomaly"] == 1]
    elif show_anom == "Normal only":
        filtered = filtered[filtered["is_anomaly"] == 0]

    st.markdown(f"Showing **{len(filtered):,}** rows  "
                f"({filtered['is_anomaly'].sum():,} anomalies)")

    # Density scatter
    st.subheader("Density Timeline (filtered)")
    filtered_copy = filtered.copy()
    filtered_copy["timestamp"] = pd.to_datetime(filtered_copy["timestamp"])
    fig = px.scatter(
        filtered_copy.sort_values("timestamp"),
        x="timestamp", y="crowd_density",
        color="anomaly_label",
        color_discrete_map=ANOMALY_COLORS,
        template="plotly_dark",
        hover_data=["zone_id", "flow_rate_in", "flow_rate_out", "net_flow"],
        height=320,
        title="",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Time",
                      legend_title="Label")
    st.plotly_chart(fig, use_container_width=True)

    # Feature distributions
    st.subheader("Feature Distributions")
    num_cols = ["crowd_density", "density_delta", "density_z_score",
                "flow_rate_in", "flow_rate_out", "net_flow",
                "temperature_celsius", "noise_level_db"]
    sel_feat = st.selectbox("Select feature", num_cols, index=0)
    fig_hist = px.histogram(
        filtered, x=sel_feat, color="anomaly_label",
        color_discrete_map=ANOMALY_COLORS,
        barmode="overlay", opacity=0.6,
        template="plotly_dark", height=280,
        marginal="box",
    )
    fig_hist.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_hist, use_container_width=True)

    # Correlation heatmap
    if st.checkbox("Show correlation heatmap"):
        corr_cols = ["crowd_density", "density_delta", "density_z_score",
                     "flow_rate_in", "flow_rate_out", "net_flow",
                     "adjacent_zone_pressure", "exit_proximity_score",
                     "dwell_time_minutes", "crowding_pressure",
                     "temperature_celsius", "noise_level_db", "is_anomaly"]
        corr_cols = [c for c in corr_cols if c in filtered.columns]
        corr_mat  = filtered[corr_cols].corr()
        fig_corr  = px.imshow(
            corr_mat, template="plotly_dark", color_continuous_scale="RdBu",
            zmin=-1, zmax=1, aspect="auto", height=480,
            text_auto=".2f",
        )
        fig_corr.update_layout(margin=dict(t=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Raw table
    st.subheader("Raw Data")
    display_cols = ["event_id", "zone_id", "timestamp", "crowd_density",
                    "density_delta", "flow_rate_in", "flow_rate_out",
                    "net_flow", "anomaly_label", "is_anomaly"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].head(500),
        use_container_width=True,
        height=320,
    )

    # Download
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button("⬇️ Download filtered CSV", csv_bytes,
                       "filtered_crowd_data.csv", "text/csv")


# ──────────────────────────────────────────────────────────────
# ── PAGE: Train Models ────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
elif "Train" in page:
    st.title("🔧 Train Anomaly Detection Models")

    with st.expander("ℹ️ About the models", expanded=False):
        st.markdown("""
        | Model | Principle | Pros | Cons |
        |-------|-----------|------|------|
        | **Isolation Forest** | Randomly partitions data; anomalies isolated faster | Fast, scalable, handles high dims | Contamination param sensitive |
        | **Local Outlier Factor** | Density-based; compares local neighbourhood | Catches local clusters | Slow on large N |
        | **One-Class SVM** | Kernel-based boundary around normal | Powerful non-linear boundary | Very slow on large N |
        """)

    st.subheader("Training Parameters")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        cont   = st.slider("Contamination (anomaly rate)", 0.01, 0.20, 0.03, 0.01,
                           help="Expected fraction of anomalies in the dataset")
    with col_p2:
        n_est  = st.slider("IF: n_estimators", 50, 300, 150, 10)
    with col_p3:
        seed   = st.number_input("Random seed", 1, 999, 42)

    data_ok = DATA_PATH.exists()
    if not data_ok:
        st.warning("Dataset not found. Generate it first.")

    if st.button("🚀 Start Training", disabled=not data_ok, type="primary"):
        log_area = st.empty()
        prog_bar = st.progress(0)
        cmd = [
            sys.executable, str(APP_DIR / "train_model.py"),
            "--contamination", str(cont),
            "--n-estimators", str(n_est),
            "--seed", str(seed),
        ]
        log_lines = []
        with st.spinner("Training in progress …"):
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(APP_DIR),
            )
            step = 0
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                log_area.code("\n".join(log_lines[-25:]), language="")
                step = min(step + 2, 95)
                prog_bar.progress(step)
            proc.wait()
        prog_bar.progress(100)

        if proc.returncode == 0:
            st.success("✅ Training complete! Models saved to `models/`")
            load_detector.clear()
            load_report.clear()
        else:
            st.error("Training failed. Check the logs above.")

    # Show existing report
    report = load_report()
    if report:
        st.divider()
        st.subheader("Last Training Report")
        st.markdown(f"Trained at: `{report.get('trained_at', 'N/A')}`  |  "
                    f"Best model: **{report.get('best_model', 'N/A')}**")

        models_data = report.get("models", {})
        rows = []
        for mname, mdata in models_data.items():
            rows.append({
                "Model":       mname.replace("_", " ").title(),
                "F1":          mdata.get("f1_score", 0),
                "Precision":   mdata.get("precision", 0),
                "Recall":      mdata.get("recall", 0),
                "ROC-AUC":     mdata.get("roc_auc", 0),
                "Avg Prec":    mdata.get("avg_precision", 0),
            })
        metrics_df = pd.DataFrame(rows)

        fig_bar = px.bar(
            metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Model", barmode="group",
            template="plotly_dark", height=320,
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_range=[0, 1])
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(metrics_df.set_index("Model").style.format("{:.4f}"),
                     use_container_width=True)


# ──────────────────────────────────────────────────────────────
# ── PAGE: Single Reading Predictor ────────────────────────────
# ──────────────────────────────────────────────────────────────
elif "Single" in page:
    st.title("🔍 Single Reading Predictor")
    st.markdown("Enter real-time sensor values for one zone to get an instant anomaly prediction.")

    detector = load_detector()
    if detector is None or detector.pipeline is None:
        st.warning("No trained model found. Train one first.")
        st.stop()

    with st.form("single_predict_form"):
        st.subheader("Zone Telemetry Inputs")

        col1, col2, col3 = st.columns(3)
        with col1:
            crowd_density  = st.slider("Crowd Density (0–1)",     0.0, 1.0, 0.35, 0.01)
            flow_in        = st.number_input("Flow Rate In (ppm)", 0.0, 500.0, 40.0, 1.0)
            flow_out       = st.number_input("Flow Rate Out (ppm)",0.0, 500.0, 38.0, 1.0)
            net_flow       = flow_in - flow_out

        with col2:
            density_delta  = st.slider("Density Delta (change)",  -0.5, 0.5, 0.0, 0.01)
            adj_pressure   = st.slider("Adjacent Zone Pressure",   0.0, 1.0, 0.3, 0.01)
            exit_prox      = st.slider("Exit Proximity Score",      0.0, 1.0, 0.5, 0.01)

        with col3:
            temperature    = st.slider("Temperature (°C)",        10.0, 45.0, 24.0, 0.5)
            noise_db       = st.slider("Noise Level (dB)",        40.0, 110.0, 65.0, 1.0)
            hour_of_day    = st.slider("Hour of Day",              0.0, 23.9, 18.0, 0.1)
            min_elapsed    = st.slider("Minutes Since Event Start",0.0, 360.0, 90.0, 1.0)

        # Derived feature estimates
        density_abs    = abs(density_delta)
        roll_mean5     = crowd_density - density_delta * 2.5
        roll_std5      = max(0.0, abs(density_delta) * 1.5)
        roll_mean10    = crowd_density - density_delta * 5
        roll_std10     = max(0.0, abs(density_delta) * 2)
        z_score        = (density_delta / max(roll_std10, 1e-6)) if roll_std10 else 0.0
        denom          = max(flow_in + flow_out, 1e-6)
        flow_imb       = (flow_in - flow_out) / denom
        crowding_p     = crowd_density * adj_pressure * (1 - exit_prox)
        perc_overload  = crowd_density * noise_db / 100
        dwell_t        = min(120, max(5, flow_out and (crowd_density * 5000 / flow_out) or 30))
        is_peak        = 1 if 17 <= int(hour_of_day) <= 21 else 0
        cap_util       = crowd_density

        reading = {
            "crowd_density":              crowd_density,
            "capacity_utilization":       cap_util,
            "density_delta":              density_delta,
            "density_delta_abs":          density_abs,
            "density_rolling_mean_5":     roll_mean5,
            "density_rolling_std_5":      roll_std5,
            "density_rolling_mean_10":    roll_mean10,
            "density_rolling_std_10":     roll_std10,
            "density_z_score":            z_score,
            "flow_rate_in":               flow_in,
            "flow_rate_out":              flow_out,
            "net_flow":                   net_flow,
            "flow_imbalance_ratio":       flow_imb,
            "adjacent_zone_pressure":     adj_pressure,
            "exit_proximity_score":       exit_prox,
            "dwell_time_minutes":         dwell_t,
            "crowding_pressure":          crowding_p,
            "perceptual_overload":        perc_overload,
            "temperature_celsius":        temperature,
            "noise_level_db":             noise_db,
            "hour_of_day":                hour_of_day,
            "minutes_since_event_start":  min_elapsed,
            "is_peak_hour":               is_peak,
        }

        submitted = st.form_submit_button("🔍 Predict", type="primary", use_container_width=True)

    if submitted:
        result = detector.predict_single(reading)
        sev = result.severity
        color = SEVERITY_COLORS.get(sev, "#95a5a6")

        st.divider()
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            st.metric("Anomaly Detected", "⚠️ YES" if result.is_anomaly else "✅ NO")
        with r_col2:
            st.metric("Anomaly Score", f"{result.anomaly_score:.4f}")
        with r_col3:
            st.metric("Severity", sev)
        with r_col4:
            st.metric("Type", result.anomaly_type)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result.anomaly_score * 100,
            title={"text": "Anomaly Risk Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0,  50], "color": "#1d3a1d"},
                    {"range": [50, 75], "color": "#3d3300"},
                    {"range": [75, 90], "color": "#4a1a00"},
                    {"range": [90, 100],"color": "#3d0007"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 75},
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig_gauge.update_layout(
            template="plotly_dark", height=300,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if result.triggered_rules:
            st.markdown("**🔴 Triggered Rules:**")
            for r in result.triggered_rules:
                st.markdown(f"- `{r}`")

        if result.recommendations:
            st.markdown("**📋 Recommendations:**")
            for rec in result.recommendations:
                st.markdown(f"- {rec}")


# ──────────────────────────────────────────────────────────────
# ── PAGE: Batch Prediction ────────────────────────────────────
# ──────────────────────────────────────────────────────────────
elif "Batch" in page:
    st.title("📂 Batch Prediction")
    st.markdown(
        "Upload a CSV file matching the feature schema, "
        "or run predictions on the existing generated dataset."
    )

    detector = load_detector()
    if detector is None or detector.pipeline is None:
        st.warning("No trained model found. Train one first.")
        st.stop()

    tab_upload, tab_existing = st.tabs(["📤 Upload CSV", "📋 Use Generated Dataset"])

    with tab_upload:
        uploaded = st.file_uploader("Upload crowd telemetry CSV", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.markdown(f"Uploaded: **{len(df_up):,}** rows, **{df_up.shape[1]}** columns")
            st.dataframe(df_up.head(), use_container_width=True)

            if st.button("Run Prediction on Upload", type="primary"):
                with st.spinner("Running batch prediction…"):
                    result_df = detector.predict_batch(df_up)
                anom_n = result_df["is_anomaly_pred"].sum()
                st.success(f"Done — {anom_n:,} anomalies detected")

                fig_score = px.histogram(
                    result_df, x="anomaly_score", color="severity",
                    color_discrete_map=SEVERITY_COLORS,
                    template="plotly_dark", height=280,
                    title="Anomaly Score Distribution",
                )
                st.plotly_chart(fig_score, use_container_width=True)

                st.dataframe(result_df.head(200), use_container_width=True, height=300)
                csv_bytes = result_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download Predictions", csv_bytes,
                                   "predictions.csv", "text/csv")

    with tab_existing:
        feat_df = load_csv(DATA_PATH)
        if feat_df is None:
            st.info("No dataset. Generate it via the sidebar.")
        else:
            n_sample = st.slider("Rows to score", 500, min(50_000, len(feat_df)), 5000, 500)
            sample_df = feat_df.sample(n_sample, random_state=42)

            if st.button("🏃 Score Sample", type="primary"):
                with st.spinner("Scoring…"):
                    result_df = detector.predict_batch(sample_df.copy())

                anom_n  = result_df["is_anomaly_pred"].sum()
                true_n  = result_df["is_anomaly"].sum() if "is_anomaly" in result_df else "N/A"
                st.success(f"Predicted: **{anom_n:,}** anomalies  |  Ground truth: **{true_n}**")

                # Confusion-style comparison
                if "is_anomaly" in result_df.columns:
                    from sklearn.metrics import classification_report as cr
                    rep = cr(result_df["is_anomaly"], result_df["is_anomaly_pred"],
                             output_dict=True, zero_division=0)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Precision", f"{rep.get('1', {}).get('precision', 0):.4f}")
                    c2.metric("Recall",    f"{rep.get('1', {}).get('recall', 0):.4f}")
                    c3.metric("F1",        f"{rep.get('1', {}).get('f1-score', 0):.4f}")

                # Severity breakdown pie
                sev_counts = result_df["severity"].value_counts().reset_index()
                sev_counts.columns = ["Severity", "Count"]
                fig_pie = px.pie(
                    sev_counts, names="Severity", values="Count",
                    color="Severity", color_discrete_map=SEVERITY_COLORS,
                    template="plotly_dark", height=300,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                st.dataframe(
                    result_df[["event_id","zone_id","timestamp","crowd_density",
                               "is_anomaly","is_anomaly_pred","anomaly_score",
                               "severity","anomaly_type"]].head(300),
                    use_container_width=True, height=300,
                )
                csv_bytes = result_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download Scored Data", csv_bytes,
                                   "scored_predictions.csv", "text/csv")


# ──────────────────────────────────────────────────────────────
# ── PAGE: Model Performance ───────────────────────────────────
# ──────────────────────────────────────────────────────────────
elif "Performance" in page:
    st.title("📈 Model Performance Report")
    report = load_report()
    if report is None:
        st.warning("No training report found. Train models first.")
        st.stop()

    st.markdown(
        f"**Trained:** {report.get('trained_at','N/A')}   "
        f"| **Best Model:** `{report.get('best_model','N/A')}`   "
        f"| **Features:** {report.get('n_features', 0)}   "
        f"| **Dataset Rows:** {report.get('dataset_rows', 0):,}"
    )

    models = report.get("models", {})

    # ── Summary table ──────────────────────────────────────────
    summary_rows = []
    for mname, mdata in models.items():
        cm = mdata.get("confusion_matrix", {})
        summary_rows.append({
            "Model":     mname.replace("_", " ").title(),
            "Precision": mdata["precision"],
            "Recall":    mdata["recall"],
            "F1":        mdata["f1_score"],
            "ROC-AUC":   mdata["roc_auc"],
            "Avg Prec":  mdata["avg_precision"],
            "TP": cm.get("TP", "-"),
            "FP": cm.get("FP", "-"),
            "FN": cm.get("FN", "-"),
            "TN": cm.get("TN", "-"),
        })
    summ_df = pd.DataFrame(summary_rows).set_index("Model")
    st.subheader("Metrics Summary")
    st.dataframe(summ_df.style.format({
        "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}",
        "ROC-AUC": "{:.4f}", "Avg Prec": "{:.4f}",
    }), use_container_width=True)

    # ── Radar chart ────────────────────────────────────────────
    st.subheader("Model Comparison — Radar")
    radar_metrics = ["Precision", "Recall", "F1", "ROC-AUC", "Avg Prec"]
    fig_radar = go.Figure()
    for _, row_r in summ_df.iterrows():
        vals = [row_r[m] for m in radar_metrics] + [row_r[radar_metrics[0]]]
        cats = radar_metrics + [radar_metrics[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", name=row_r.name,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_dark", height=380,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Per-type breakdown ─────────────────────────────────────
    st.subheader("Per Anomaly Type Recall (Isolation Forest)")
    if_breakdown = models.get("isolation_forest", {}).get("per_type", {})
    type_rows = []
    for atype, tdata in if_breakdown.items():
        if atype == "NORMAL":
            type_rows.append({
                "Type": atype, "Total": tdata.get("total", 0),
                "Caught / FPR": f"FPR={tdata.get('false_pos_rate',0):.4f}", "Recall": "-"
            })
        else:
            type_rows.append({
                "Type": atype, "Total": tdata.get("total", 0),
                "Caught / FPR": str(tdata.get("caught", 0)), "Recall": tdata.get("recall", 0)
            })
    type_df = pd.DataFrame(type_rows)
    st.dataframe(type_df, use_container_width=True)

    # Full feature list
    with st.expander("Feature Columns Used"):
        st.code("\n".join(report.get("feature_cols", [])))


# ──────────────────────────────────────────────────────────────
# ── PAGE: Zone Simulation ─────────────────────────────────────
# ──────────────────────────────────────────────────────────────
elif "Simulation" in page:
    st.title("🗺️ Zone Live Simulation")
    st.markdown("Watch the anomaly detector process a live stream of synthetic zone readings.")

    detector = load_detector()
    if detector is None or detector.pipeline is None:
        st.warning("No trained model found. Train one first.")
        st.stop()

    ZONES_VIZ = ["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D", "ZONE_E", "ZONE_F",
                 "ENTRY_GATE_NORTH", "ENTRY_GATE_SOUTH", "CONCOURSE_MAIN"]

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        sim_speed = st.slider("Speed (readings/sec)", 1, 10, 3)
    with col_ctrl2:
        inject_at = st.selectbox("Inject anomaly in zone", ["(none)"] + ZONES_VIZ)

    # State
    if "sim_history" not in st.session_state:
        st.session_state.sim_history = []
    if "sim_step" not in st.session_state:
        st.session_state.sim_step = 0

    col_run1, col_run2 = st.columns(2)
    run_sim   = col_run1.button("▶ Start 30-step Simulation", type="primary")
    clear_sim = col_run2.button("🗑 Clear History")
    if clear_sim:
        st.session_state.sim_history = []
        st.session_state.sim_step = 0

    # Zone grid placeholder
    grid_placeholder  = st.empty()
    chart_placeholder = st.empty()
    log_placeholder   = st.empty()

    def _make_reading(zone: str, step: int, inject: bool) -> dict:
        rng = np.random.default_rng(step * 17 + hash(zone) % 100)
        t_norm = min(1.0, step / 100)
        base = 0.2 + 0.5 * np.sin(np.pi * t_norm) + rng.normal(0, 0.04)
        density = float(np.clip(base, 0.0, 1.0))
        if inject:
            density = float(np.clip(density + rng.uniform(0.35, 0.55), 0, 1))
        fi = density * 5000 * rng.uniform(0.03, 0.07)
        fo = density * 5000 * rng.uniform(0.02, 0.06)
        return {
            "crowd_density": density,
            "capacity_utilization": density,
            "density_delta": rng.uniform(-0.05, 0.05) if not inject else 0.35,
            "density_delta_abs": abs(rng.uniform(-0.05, 0.05)) if not inject else 0.35,
            "density_rolling_mean_5":  density * 0.95,
            "density_rolling_std_5":   0.02,
            "density_rolling_mean_10": density * 0.92,
            "density_rolling_std_10":  0.04,
            "density_z_score":         (density - 0.5) / 0.15 if not inject else 4.5,
            "flow_rate_in": float(fi),
            "flow_rate_out": float(fo),
            "net_flow": float(fi - fo),
            "flow_imbalance_ratio": (fi - fo) / max(fi + fo, 1),
            "adjacent_zone_pressure": density * 0.8 + rng.normal(0, 0.05),
            "exit_proximity_score": 0.4 + rng.uniform(-0.1, 0.1),
            "dwell_time_minutes": float(np.clip(density * 60, 5, 120)),
            "crowding_pressure": density * 0.7 * 0.6,
            "perceptual_overload": density * 70 / 100,
            "temperature_celsius": 22 + density * 8 + rng.normal(0, 0.5),
            "noise_level_db": 55 + density * 45 + rng.normal(0, 2),
            "hour_of_day": 18.0 + step / 60,
            "minutes_since_event_start": float(step),
            "is_peak_hour": 1,
        }

    if run_sim:
        for step_i in range(30):
            step = st.session_state.sim_step + step_i
            zone_results = {}
            for zone in ZONES_VIZ:
                inject = (inject_at != "(none)" and zone == inject_at and step_i == 10)
                reading = _make_reading(zone, step, inject)
                res     = detector.predict_single(reading)
                zone_results[zone] = {
                    "density":   reading["crowd_density"],
                    "score":     res.anomaly_score,
                    "severity":  res.severity,
                    "is_anomaly": res.is_anomaly,
                    "type":      res.anomaly_type,
                }
                if res.is_anomaly:
                    st.session_state.sim_history.append({
                        "step": step, "zone": zone,
                        "density": round(reading["crowd_density"], 3),
                        "score": round(res.anomaly_score, 4),
                        "severity": res.severity,
                        "type": res.anomaly_type,
                    })

            # Zone grid
            zone_cols = grid_placeholder.columns(len(ZONES_VIZ))
            for ci, zone in enumerate(ZONES_VIZ):
                zr = zone_results[zone]
                sc = SEVERITY_COLORS.get(zr["severity"], "#95a5a6")
                anom_icon = "⚠️" if zr["is_anomaly"] else "✅"
                zone_cols[ci].markdown(
                    f"<div style='background:{sc};border-radius:8px;padding:8px;text-align:center;"
                    f"min-height:80px'>"
                    f"<b style='font-size:.75rem'>{zone.replace('_',' ')}</b><br>"
                    f"<span style='font-size:1.2rem'>{anom_icon}</span><br>"
                    f"<span style='font-size:.8rem'>{zr['density']:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # History chart
            if st.session_state.sim_history:
                hist_df = pd.DataFrame(st.session_state.sim_history)
                fig_sim = px.scatter(
                    hist_df, x="step", y="score", color="severity",
                    color_discrete_map=SEVERITY_COLORS,
                    symbol="zone", template="plotly_dark",
                    height=250, title="Anomaly Score History",
                    hover_data=["zone","density","type"],
                )
                fig_sim.update_layout(margin=dict(t=30, b=10, l=10, r=10))
                chart_placeholder.plotly_chart(fig_sim, use_container_width=True)

                log_placeholder.dataframe(
                    hist_df.tail(15).sort_values("step", ascending=False),
                    use_container_width=True,
                )

            time.sleep(1.0 / sim_speed)
            st.session_state.sim_step += 30


# ──────────────────────────────────────────────────────────────
# ── PAGE: Incident Log ────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
elif "Incident" in page:
    st.title("📋 Incident Log")
    st.markdown("Anomalies from the generated dataset formatted as DrishtiX incident records.")

    feat_df = load_csv(DATA_PATH)
    if feat_df is None:
        st.warning("No dataset found. Generate it first.")
        st.stop()

    incidents = feat_df[feat_df["is_anomaly"] == 1].copy()
    incidents = incidents.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # Severity assignment (based on anomaly type heuristic)
    sev_map = {
        "CROWD_SURGE":        "HIGH",
        "STAMPEDE_PRECURSOR": "CRITICAL",
        "BOTTLENECK":         "HIGH",
        "FLOW_REVERSAL":      "MEDIUM",
        "ISOLATION_ZONE":     "MEDIUM",
        "SUSTAINED_OVERLOAD": "CRITICAL",
    }
    incidents["severity_level"] = incidents["anomaly_label"].map(
        sev_map).fillna("LOW")

    # Filter
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        sel_type = st.multiselect("Anomaly Type",
                                  incidents["anomaly_label"].unique(),
                                  default=list(incidents["anomaly_label"].unique()))
    with cf2:
        sel_sev  = st.multiselect("Severity",
                                  ["CRITICAL","HIGH","MEDIUM","LOW"],
                                  default=["CRITICAL","HIGH","MEDIUM","LOW"])
    with cf3:
        sel_ev   = st.multiselect("Event", incidents["event_id"].unique(),
                                  default=list(incidents["event_id"].unique()[:5]))

    filt = incidents[
        incidents["anomaly_label"].isin(sel_type) &
        incidents["severity_level"].isin(sel_sev) &
        incidents["event_id"].isin(sel_ev)
    ]

    st.markdown(f"**{len(filt):,}** incidents matching filters")

    # Summary stats
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total", len(filt))
    sc2.metric("Critical", (filt["severity_level"] == "CRITICAL").sum())
    sc3.metric("High",     (filt["severity_level"] == "HIGH").sum())
    sc4.metric("Unique Zones", filt["zone_id"].nunique())

    # Timeline
    filt_copy = filt.copy()
    filt_copy["timestamp"] = pd.to_datetime(filt_copy["timestamp"])
    fig_tl = px.scatter(
        filt_copy.sort_values("timestamp"),
        x="timestamp", y="zone_id",
        color="severity_level",
        color_discrete_map=SEVERITY_COLORS,
        symbol="anomaly_label",
        size="crowd_density",
        template="plotly_dark",
        height=350, title="Incident Timeline by Zone",
        hover_data=["anomaly_label","crowd_density","flow_rate_in","net_flow"],
    )
    fig_tl.update_layout(margin=dict(t=30, b=10), yaxis_title="Zone")
    st.plotly_chart(fig_tl, use_container_width=True)

    # Incident table
    display = ["event_id","zone_id","timestamp","anomaly_label","severity_level",
               "crowd_density","flow_rate_in","flow_rate_out","net_flow","temperature_celsius"]
    st.dataframe(filt[display].head(500), use_container_width=True, height=350)

    csv_bytes = filt.to_csv(index=False).encode()
    st.download_button("⬇️ Export Incident Log (CSV)", csv_bytes,
                       "incident_log.csv", "text/csv")


# ──────────────────────────────────────────────────────────────
# ── PAGE: Visual Detection ────────────────────────────────────
# ──────────────────────────────────────────────────────────────
elif "Visual" in page:
    st.title("🎥 Visual Crowd Anomaly Detection")
    st.markdown(
        "Upload an **image** or **video** to classify crowd scenes with CLIP. "
        "Results include severity, recommended actions, and per-frame timelines for video."
    )

    @st.cache_resource(show_spinner=False)
    def _load_visual():
        try:
            sys.path.insert(0, str(APP_DIR))
            from visual_detector import VisualCrowdDetector
            det = VisualCrowdDetector(settings_path=APP_DIR / "settings.yaml")
            return det if det.is_ready else None
        except Exception:
            return None

    vis_det = _load_visual()

    if vis_det is None:
        st.warning(
            "**Visual detector not available.** Install PyTorch and CLIP:\n"
            "```\n"
            "pip install torch torchvision\n"
            "pip install git+https://github.com/openai/CLIP.git\n"
            "```"
        )
        st.stop()

    st.success(f"✅ CLIP model loaded — {len(vis_det._labels)} crowd-safety labels")

    tab_img, tab_video = st.tabs(["🖼️ Image", "🎬 Video"])

    # ── Image tab ──────────────────────────────────────────────
    with tab_img:
        uploaded_img = st.file_uploader(
            "Upload crowd image", type=["jpg", "jpeg", "png", "webp"],
            key="vis_img_upload"
        )

        if uploaded_img:
            from PIL import Image as _PIL
            import cv2 as _cv2
            pil_img = _PIL.open(uploaded_img).convert("RGB")
            rgb_arr = np.array(pil_img)

            st.image(pil_img, caption="Uploaded image", use_column_width=True)

            with st.spinner("Running CLIP inference…"):
                vres = vis_det.predict(rgb_arr, use_smoother=False)

            sev_color = SEVERITY_COLORS.get(vres.severity, "#95a5a6")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Anomaly",    "⚠️ YES" if vres.is_anomaly else "✅ NO")
            c2.metric("Category",   vres.category)
            c3.metric("Severity",   vres.severity)
            c4.metric("Confidence", f"{vres.confidence:.3f}")

            st.markdown(f"**Matched label:** `{vres.label}`")

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=vres.confidence * 100,
                title={"text": "Prediction Confidence (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": sev_color},
                    "steps": [
                        {"range": [0,  30], "color": "#1d3a1d"},
                        {"range": [30, 60], "color": "#3d3300"},
                        {"range": [60, 80], "color": "#4a1a00"},
                        {"range": [80,100], "color": "#3d0007"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": 50},
                },
            ))
            fig_g.update_layout(
                template="plotly_dark", height=280,
                margin=dict(t=20, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_g, use_container_width=True)

            if vres.recommendations:
                st.markdown("**📋 Recommended Actions:**")
                for rec in vres.recommendations:
                    st.markdown(f"- {rec}")

    # ── Video tab ──────────────────────────────────────────────
    with tab_video:
        uploaded_vid = st.file_uploader(
            "Upload crowd video", type=["mp4", "avi", "mov", "mkv"],
            key="vis_vid_upload"
        )
        stride = st.slider("Frame stride (1=every frame, 3=every 3rd)", 1, 10, 3)

        if uploaded_vid:
            import cv2 as _cv2
            import tempfile as _tempfile
            with _tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name

            with st.spinner("Decoding video frames…"):
                cap = _cv2.VideoCapture(tmp_path)
                frames_rgb = []
                while True:
                    ok, frm = cap.read()
                    if not ok:
                        break
                    frames_rgb.append(_cv2.cvtColor(frm, _cv2.COLOR_BGR2RGB))
                cap.release()

            st.markdown(f"Decoded **{len(frames_rgb)}** frames.")

            if st.button("🔍 Analyse Video", type="primary"):
                with st.spinner(f"Running CLIP on every {stride}th frame…"):
                    summary = vis_det.summarise_video(frames_rgb, frame_stride=stride)

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Total Frames",   summary["total_frames"])
                sc2.metric("Anomaly Frames",  summary["anomaly_frames"])
                sc3.metric("Anomaly Rate",    f"{summary['anomaly_rate']*100:.1f}%")
                sc4.metric("Peak Confidence", f"{summary['peak_confidence']:.3f}")

                sev     = summary["severity"]
                dom     = summary["dominant_category"]
                sev_col = SEVERITY_COLORS.get(sev, "#95a5a6")
                st.markdown(
                    f"<div style='background:{sev_col};border-radius:8px;"
                    f"padding:14px 20px;color:white;font-size:1.1rem;font-weight:700'>"
                    f"Dominant: {dom} &nbsp;|&nbsp; Overall Severity: {sev}</div>",
                    unsafe_allow_html=True,
                )

                if summary["recommendations"]:
                    st.markdown("**📋 Recommended Actions:**")
                    for rec in summary["recommendations"]:
                        st.markdown(f"- {rec}")

                # Per-frame timeline
                frame_results = vis_det.predict_video(
                    frames_rgb, frame_stride=stride, use_smoother=True
                )
                frame_df = pd.DataFrame([
                    {
                        "frame":      i * stride,
                        "category":   r.category,
                        "confidence": round(r.confidence, 4),
                        "is_anomaly": r.is_anomaly,
                        "smoothed":   round(r.smoothed_score, 4),
                    }
                    for i, r in enumerate(frame_results)
                ])

                fig_vid = go.Figure()
                fig_vid.add_scatter(
                    x=frame_df["frame"], y=frame_df["smoothed"],
                    mode="lines", name="Smoothed Score",
                    line=dict(color="#3498db", width=2),
                )
                anom_frames = frame_df[frame_df["is_anomaly"]]
                if not anom_frames.empty:
                    fig_vid.add_scatter(
                        x=anom_frames["frame"], y=anom_frames["smoothed"],
                        mode="markers", name="Anomaly Frame",
                        marker=dict(color="#e74c3c", size=8, symbol="x"),
                    )
                fig_vid.update_layout(
                    template="plotly_dark", height=280,
                    margin=dict(t=10, b=10, l=10, r=10),
                    xaxis_title="Frame", yaxis_title="Anomaly Score",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_vid, use_container_width=True)

                st.dataframe(
                    frame_df[["frame", "category", "confidence", "smoothed", "is_anomaly"]],
                    use_container_width=True, height=280,
                )
