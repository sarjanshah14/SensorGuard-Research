"""
SKAB Research Explorer — Deep data understanding + future model workspace.
Run:  streamlit run skab.py
"""

import os
import glob
import json
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Config
# ─────────────────────────────────────────────────────────────────────────────

SKAB_ROOT = "SKAB"

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

# Plain-English notes shown beneath each paper figure in the Model Results tab.
RESULT_FIGURES = [
    ("model_comparison.png", "How the 4 models compare",
     "Each cluster is one score. Higher is better (max 1.0). The tall red bar "
     "(IsolationForest) leading on recall and F1 is the paper's headline result."),
    ("confusion_matrices.png", "What each model got right vs. wrong",
     "Top-left = correctly called normal, bottom-right = correctly caught a fault. "
     "Top-right = false alarms, bottom-left = missed faults. IsolationForest has the "
     "fullest bottom-right (catches the most real faults)."),
    ("timeseries_anomalies.png", "Predictions on one example run",
     "Every sensor over time for one test run. Red shading = real fault periods, "
     "orange dots = where IsolationForest flagged a fault. Dots inside red = correct "
     "catches; dots outside = false alarms."),
    ("anomaly_closeup.png", "Zoom into the moment a fault begins",
     "The same run zoomed to ±30s around the fault onset (red dashed line). Watch "
     "the flow-rate panel jump right at the line — the physical fault signature."),
]

SENSOR_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS",
]

SENSOR_EXPLAIN = {
    "Accelerometer1RMS": {
        "what": "Vibration intensity at pump mount-point 1 (RMS of raw accelerometer).",
        "why":  "Vibration spikes signal mechanical imbalance, cavitation, or loose parts.",
        "normal": "Low, flat line with tiny jitter (~0.02–0.03 g).",
        "anomaly": "Sudden jumps or sustained higher baseline = something is shaking loose.",
        "unit": "g (RMS)",
    },
    "Accelerometer2RMS": {
        "what": "Vibration intensity at pump mount-point 2.",
        "why":  "Second location catches directional faults the first sensor might miss.",
        "normal": "Slightly higher than sensor 1 but equally steady (~0.04 g).",
        "anomaly": "Correlated spikes with sensor 1 confirm mechanical fault, uncorrelated = localized issue.",
        "unit": "g (RMS)",
    },
    "Current": {
        "what": "Electrical current drawn by the pump motor.",
        "why":  "Motor draws more current when fighting resistance (blocked valve) or less under no-load faults.",
        "normal": "Steady around 1–2 A with occasional startup spikes.",
        "anomaly": "Extreme spikes (>200 A) = motor stalling. Gradual rise = increasing load from blockage.",
        "unit": "Amps",
    },
    "Pressure": {
        "what": "Water pressure at the pipe measurement point.",
        "why":  "Directly reveals valve blockage — pressure builds upstream of a blocked valve.",
        "normal": "Stable low pressure when water flows freely (~0.05 bar).",
        "anomaly": "Rising pressure = valve closing. Negative pressure = cavitation / suction effects.",
        "unit": "Bar",
    },
    "Temperature": {
        "what": "Water temperature in the flow loop.",
        "why":  "Restricted flow → less cooling → temperature drift upward.",
        "normal": "Slowly drifts within a 5°C band (thermal inertia keeps it stable).",
        "anomaly": "Faster rise rate when flow is restricted. The lag makes this a late indicator.",
        "unit": "°C",
    },
    "Thermocouple": {
        "what": "Pump body (housing) surface temperature.",
        "why":  "Overheating housing = friction from low flow, bearing wear, or motor strain.",
        "normal": "Flat around 26°C with very small variance.",
        "anomaly": "Gradual rise = sustained fault. Responds slower than other sensors (thermal mass).",
        "unit": "°C",
    },
    "Voltage": {
        "what": "Supply voltage to the pump motor.",
        "why":  "Voltage drops during current spikes (V=IR). Useful as a confirming indicator.",
        "normal": "Stable 225–240 V from mains supply.",
        "anomaly": "Momentary dips to near-zero during motor stall events. Usually mirrors Current inversely.",
        "unit": "Volts",
    },
    "Volume Flow RateRMS": {
        "what": "Water volume flow through the pipe (RMS of flow meter signal).",
        "why":  "The most direct indicator of valve blockage — flow drops when valves close.",
        "normal": "High and constant (~32 L/min baseline for valve1, ~120 for anomaly-free).",
        "anomaly": "Sharp drop to near-zero = fully blocked. Partial drop = partial blockage.",
        "unit": "L/min (RMS)",
    },
}

FOLDER_EXPLAIN = {
    "anomaly-free": {
        "title": "Anomaly-Free Baseline",
        "what": "Pump running with no faults — 2.5 hours of clean normal operation.",
        "why":  "Your model's 'ground truth normal'. Train on this to learn what healthy looks like.",
        "how":  "Pump and valves fully open, steady state operation.",
    },
    "valve1": {
        "title": "Valve 1 Blockage",
        "what": "16 experiment runs where Valve 1 was gradually closed to restrict flow.",
        "why":  "Most common industrial fault — valve failure / partial blockage.",
        "how":  "Valve physically turned to reduce flow. Each file = one blockage-and-release cycle.",
    },
    "valve2": {
        "title": "Valve 2 Blockage",
        "what": "4 runs blocking a different valve in the flow loop.",
        "why":  "Tests if your model generalises across valve locations.",
        "how":  "Same blockage method, different valve position in the plumbing.",
    },
    "other": {
        "title": "Other Fault Types",
        "what": "14 runs with miscellaneous faults: cavitation, electrical, bearing wear.",
        "why":  "Real systems have diverse failure modes, not just valve problems.",
        "how":  "Various fault injection methods — more unpredictable than valve experiments.",
    },
}

ACCENT = "#EF4444"
ACCENT_LIGHT = "rgba(239,68,68,0.12)"
PALETTE = ["#6366F1", "#10B981", "#F59E0B", "#3B82F6",
           "#EC4899", "#14B8A6", "#8B5CF6", "#F97316"]
FOLDER_COLORS = {"anomaly-free": "#10B981", "valve1": "#6366F1",
                 "valve2": "#3B82F6", "other": "#F59E0B"}

# ─────────────────────────────────────────────────────────────────────────────
# Data layer — cached, loads once
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def discover_files():
    result = {}
    if not os.path.isdir(SKAB_ROOT):
        return result
    for folder in sorted(os.listdir(SKAB_ROOT)):
        fp = os.path.join(SKAB_ROOT, folder)
        if os.path.isdir(fp):
            csvs = sorted(glob.glob(os.path.join(fp, "*.csv")))
            if csvs:
                result[folder] = csvs
    return result


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"], index_col="datetime")
    for col in ("anomaly", "changepoint"):
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    return df


@st.cache_data
def load_folder(paths: tuple) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = load_csv(p)
        d["_file"] = os.path.splitext(os.path.basename(p))[0]
        frames.append(d)
    return pd.concat(frames).sort_index()


@st.cache_data
def load_all(file_map_frozen) -> pd.DataFrame:
    frames = []
    for folder, paths in file_map_frozen:
        for p in paths:
            d = load_csv(p)
            d["_folder"] = folder
            d["_file"] = os.path.splitext(os.path.basename(p))[0]
            frames.append(d)
    return pd.concat(frames).sort_index()


def sensors_in(df):
    return [c for c in SENSOR_COLS if c in df.columns]


def anomaly_intervals(df):
    if df["anomaly"].sum() == 0:
        return []
    mask = df["anomaly"].astype(int)
    diff = mask.diff().fillna(0)
    starts = df.index[diff == 1].tolist()
    ends = df.index[diff == -1].tolist()
    if mask.iloc[0] == 1:
        starts.insert(0, df.index[0])
    if mask.iloc[-1] == 1:
        ends.append(df.index[-1])
    return list(zip(starts, ends))


def add_shading(fig, intervals, **kw):
    for s, e in intervals:
        fig.add_vrect(x0=s, x1=e, fillcolor=ACCENT_LIGHT,
                      line_width=0, annotation_text="", **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Insight computation (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def compute_sensor_insights(df_pickled, scols):
    """Compare normal vs anomaly windows across all sensors."""
    df = df_pickled
    has_anom = df["anomaly"].sum() > 0
    if not has_anom:
        return None

    normal = df.loc[df["anomaly"] == 0, scols]
    anom   = df.loc[df["anomaly"] == 1, scols]

    rows = []
    for c in scols:
        n_mean, n_std = normal[c].mean(), normal[c].std()
        a_mean, a_std = anom[c].mean(), anom[c].std()
        mean_shift = abs(a_mean - n_mean) / n_std if n_std > 0 else 0
        std_ratio  = a_std / n_std if n_std > 0 else 1
        rows.append({
            "Sensor": c,
            "Normal mean": round(n_mean, 4),
            "Normal std": round(n_std, 4),
            "Anomaly mean": round(a_mean, 4),
            "Anomaly std": round(a_std, 4),
            "Mean shift (σ)": round(mean_shift, 2),
            "Volatility ratio": round(std_ratio, 2),
        })

    insight_df = pd.DataFrame(rows).sort_values("Mean shift (σ)", ascending=False)
    return insight_df


@st.cache_data
def compute_correlation_matrices(df_pickled, scols):
    df = df_pickled
    normal_corr = df.loc[df["anomaly"] == 0, scols].corr()
    anom_corr = df.loc[df["anomaly"] == 1, scols].corr()
    return normal_corr, anom_corr


@st.cache_data
def compute_data_quality(df_pickled, scols):
    df = df_pickled
    info = {}

    # Null counts
    info["nulls"] = df[scols].isnull().sum().to_dict()
    info["total_nulls"] = int(df[scols].isnull().sum().sum())

    # Sampling gaps
    diffs = df.index.to_series().diff().dropna().dt.total_seconds()
    info["sampling_mean"] = round(diffs.mean(), 2)
    info["sampling_median"] = round(diffs.median(), 2)
    info["sampling_max"] = round(diffs.max(), 2)
    info["gaps_over_2s"] = int((diffs > 2).sum())

    # Constant columns (zero variance)
    info["low_variance"] = [c for c in scols if df[c].std() < 1e-8]

    # Outliers per sensor (>4σ from mean)
    outliers = {}
    for c in scols:
        mean, std = df[c].mean(), df[c].std()
        if std > 0:
            outliers[c] = int(((df[c] - mean).abs() > 4 * std).sum())
        else:
            outliers[c] = 0
    info["outliers_4sigma"] = outliers

    return info


@st.cache_data
def build_file_summary(file_map_frozen):
    rows = []
    for folder, paths in file_map_frozen:
        for p in paths:
            d = load_csv(p)
            n = len(d)
            na = int(d["anomaly"].sum())
            nc = int(d["changepoint"].sum())
            intervals = anomaly_intervals(d)
            rows.append({
                "Folder": folder,
                "File": os.path.basename(p),
                "Rows": n,
                "Duration (min)": round(n / 60, 1),
                "Anomaly rows": na,
                "Anomaly %": round(na / n * 100, 1) if n else 0,
                "# Changepoints": nc,
                "# Anomaly episodes": len(intervals),
                "path": p,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly helpers
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=10, r=10, t=30, b=30),
    font=dict(size=11),
)


def make_sensor_timeseries(df, col, color, intervals, changepoints, height=200):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df[col], mode="lines",
        line=dict(color=color, width=1.3),
        hovertemplate="%{x|%H:%M:%S}<br>%{y:.4f}<extra>" + col + "</extra>",
    ))
    add_shading(fig, intervals)
    for cp in changepoints:
        fig.add_vline(x=cp, line=dict(color=ACCENT, dash="dash", width=1.5))
    fig.update_layout(height=height, showlegend=False,
                      xaxis=dict(showgrid=False),
                      yaxis=dict(gridcolor="#F3F4F6"),
                      **PLOTLY_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SKAB Research Explorer", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.block-container{padding-top:1.2rem}
h1{font-size:1.9rem!important}
div[data-testid="stMetric"]{background:#F9FAFB;border:1px solid #E5E7EB;
border-radius:8px;padding:12px 16px}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

file_map = discover_files()

with st.sidebar:
    st.title("🔬 SKAB Research")
    st.markdown("---")
    if not file_map:
        st.error(f"SKAB/ folder not found at `{SKAB_ROOT}/`.")
        st.stop()

    folder_opts = ["ALL"] + sorted(file_map.keys())
    sel_folder = st.selectbox("Experiment type", folder_opts)

    sel_path = None
    if sel_folder != "ALL":
        paths = file_map[sel_folder]
        labels = [os.path.basename(p) for p in paths]
        sel_label = st.selectbox("File", labels)
        sel_path = paths[labels.index(sel_label)]

        info = FOLDER_EXPLAIN.get(sel_folder, {})
        if info:
            st.markdown(f"**{info.get('title', sel_folder)}**")
            st.caption(f"**What:** {info.get('what','')}")
            st.caption(f"**Why it matters:** {info.get('why','')}")
            st.caption(f"**How it was created:** {info.get('how','')}")
    else:
        st.caption("Viewing **all experiments combined**.")

    st.markdown("---")
    st.caption("Skoltech Anomaly Benchmark")

# ─────────────────────────────────────────────────────────────────────────────
# Load current selection
# ─────────────────────────────────────────────────────────────────────────────

if sel_folder == "ALL":
    df = load_all(tuple(sorted(file_map.items())))
    display_name = "All Files Combined"
else:
    df = load_csv(sel_path)
    display_name = f"{sel_folder}/{os.path.basename(sel_path)}"

scols = sensors_in(df)
intervals = anomaly_intervals(df)
n_rows = len(df)
n_anom = int(df["anomaly"].sum())
anom_pct = n_anom / n_rows * 100 if n_rows else 0
dur_min = n_rows / 60
cps = df.index[df["changepoint"] == 1].tolist()

# ─────────────────────────────────────────────────────────────────────────────
# Title + live insight banner (replaces static SKAB description)
# ─────────────────────────────────────────────────────────────────────────────

st.title(f"🔬 {display_name}")

# Compute a dynamic one-liner based on data
if n_anom > 0 and intervals:
    longest_ep = max((e - s).total_seconds() for s, e in intervals)
    st.markdown(
        f"📊 **{n_rows:,} rows** · **{dur_min:.1f} min** of data · "
        f"**{n_anom:,} anomaly seconds** ({anom_pct:.1f}%) across "
        f"**{len(intervals)} episode(s)** · longest episode: **{int(longest_ep)}s**"
    )
else:
    st.markdown(
        f"📊 **{n_rows:,} rows** · **{dur_min:.1f} min** of clean baseline data · "
        f"**No anomalies** — this is what 'healthy' looks like."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_quality, tab_sensors, tab_insights, tab_closeup, tab_summary, tab_model, tab_results = st.tabs([
    "🩺 Data Quality",
    "📈 Sensor Graphs",
    "🧠 Insights",
    "🔍 Anomaly Close-up",
    "📊 All Files",
    "🤖 Model-Ready",
    "🏆 Model Results",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Quality
# ══════════════════════════════════════════════════════════════════════════════

with tab_quality:
    st.subheader("Data Quality Report")
    st.caption("Before modelling, you need to know: are there gaps? missing values? outliers? weird distributions?")

    quality = compute_data_quality(df, scols)

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total rows", f"{n_rows:,}")
    c2.metric("Missing values", quality["total_nulls"])
    c3.metric("Sampling interval", f"{quality['sampling_median']}s")
    c4.metric("Max gap", f"{quality['sampling_max']}s")
    c5.metric("Gaps > 2s", quality["gaps_over_2s"])

    if quality["total_nulls"] == 0:
        st.success("✅ **No missing values** in any sensor column. The data is complete.")
    else:
        st.warning(f"⚠️ Found {quality['total_nulls']} missing values — you'll need to handle these before modelling.")
        st.json(quality["nulls"])

    if quality["gaps_over_2s"] > 0:
        st.warning(
            f"⚠️ Found **{quality['gaps_over_2s']} sampling gaps >2s** (expected 1s intervals). "
            f"Longest gap: **{quality['sampling_max']}s**. This matters if your model assumes uniform time steps."
        )
    else:
        st.success("✅ **Uniform sampling** — all intervals ≤ 2s. Safe to treat as 1-second time series.")

    st.markdown("---")

    # Distributions
    st.markdown("#### Sensor Value Distributions")
    st.caption(
        "Box plots show the spread of each sensor. Whiskers = data range, box = 25–75th percentile. "
        "Look for sensors with extreme outliers (dots far from the box) — these can dominate your model if not normalized."
    )

    desc = df[scols].describe().T
    desc["IQR"] = desc["75%"] - desc["25%"]
    desc["skew"] = df[scols].skew()
    st.dataframe(desc[["mean", "std", "min", "25%", "50%", "75%", "max", "IQR", "skew"]].round(4),
                 use_container_width=True)

    fig_box = go.Figure()
    for i, c in enumerate(scols):
        fig_box.add_trace(go.Box(
            y=df[c], name=c, marker_color=PALETTE[i % len(PALETTE)],
            boxmean="sd",
        ))
    fig_box.update_layout(height=350, showlegend=False,
                          yaxis_title="Value", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_box, use_container_width=True)

    # Outlier report
    st.markdown("#### Outlier Count (> 4σ from mean)")
    st.caption(
        "Rows where a sensor reading is more than 4 standard deviations from the mean. "
        "These are rare events — often exactly the anomaly moments your model should learn."
    )
    outlier_df = pd.DataFrame.from_dict(quality["outliers_4sigma"], orient="index", columns=["Count"])
    outlier_df = outlier_df.sort_values("Count", ascending=False)
    st.dataframe(outlier_df, use_container_width=True)

    # Key insight
    top_outlier = outlier_df.index[0]
    st.info(
        f"💡 **{top_outlier}** has the most extreme outliers ({outlier_df.iloc[0, 0]} rows). "
        f"This sensor will have the highest feature importance in most anomaly detectors — "
        f"but also risks dominating simpler models. Consider robust scaling (RobustScaler) "
        f"or clipping before training."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Sensor Graphs
# ══════════════════════════════════════════════════════════════════════════════

with tab_sensors:
    st.subheader(f"Sensor Time-Series — {display_name}")

    if sel_folder == "ALL":
        st.warning("Combined view — select a specific file in the sidebar for cleaner per-experiment plots.")

    for i, col in enumerate(scols):
        info = SENSOR_EXPLAIN.get(col, {})
        with st.expander(f"**{info.get('what', col)}**", expanded=True):
            cols = st.columns([2, 1])
            with cols[1]:
                st.markdown(f"**Unit:** {info.get('unit', '—')}")
                st.markdown(f"**Why it matters:** {info.get('why', '—')}")
                st.markdown(f"🟢 **Normal:** {info.get('normal', '—')}")
                st.markdown(f"🔴 **Anomaly:** {info.get('anomaly', '—')}")

                # Live stats for this sensor
                mean_val = df[col].mean()
                std_val = df[col].std()
                st.caption(f"Mean: {mean_val:.4f} | Std: {std_val:.4f}")

            with cols[0]:
                fig = make_sensor_timeseries(df, col, PALETTE[i % len(PALETTE)],
                                             intervals, cps, height=200)
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Insights (the real "why")
# ══════════════════════════════════════════════════════════════════════════════

with tab_insights:
    st.subheader("🧠 What Actually Changes During Anomalies?")

    if n_anom == 0:
        st.info("This file has no anomalies. Select a fault file (valve1, valve2, other) to see insights.")
    else:
        st.caption(
            "These insights compare the **normal windows** vs **anomaly windows** in this data. "
            "They answer: which sensors change the most during faults, and how?"
        )

        insight_df = compute_sensor_insights(df, scols)

        if insight_df is not None:
            # Sensor ranking
            st.markdown("#### 🏆 Sensor Importance Ranking")
            st.caption(
                "**Mean shift (σ)** = how many standard deviations the sensor mean moves during anomalies. "
                "Higher = the sensor is more 'surprised' by the fault. "
                "**Volatility ratio** = how much noisier the sensor gets (>1 = more noisy during anomaly)."
            )
            st.dataframe(
                insight_df.style.background_gradient(subset=["Mean shift (σ)"], cmap="YlOrRd")
                               .background_gradient(subset=["Volatility ratio"], cmap="PuBu"),
                use_container_width=True,
            )

            top = insight_df.iloc[0]
            second = insight_df.iloc[1] if len(insight_df) > 1 else None
            msg = (
                f"💡 **{top['Sensor']}** is the strongest anomaly indicator — "
                f"its mean shifts by **{top['Mean shift (σ)']}σ** during faults"
            )
            if top["Volatility ratio"] > 2:
                msg += f" and gets **{top['Volatility ratio']}× more volatile**"
            msg += "."
            if second is not None:
                msg += f" Second most important: **{second['Sensor']}** ({second['Mean shift (σ)']}σ shift)."
            msg += (
                "\n\nFor your model: these top sensors should be your primary features. "
                "The bottom sensors (low shift) add noise without much signal — consider dropping them "
                "or using them only in ensemble models."
            )
            st.info(msg)

            st.markdown("---")

            # Normal vs Anomaly distributions side-by-side
            st.markdown("#### Normal vs Anomaly Distributions")
            st.caption("Each histogram shows how sensor values distribute during normal (green) vs anomaly (red) periods. "
                       "More separation = easier for a model to distinguish.")

            cols_per_row = 2
            for row_start in range(0, len(scols), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, col_obj in enumerate(row_cols):
                    idx = row_start + j
                    if idx >= len(scols):
                        break
                    sensor = scols[idx]
                    with col_obj:
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=df.loc[df["anomaly"] == 0, sensor], name="Normal",
                            marker_color="#10B981", opacity=0.7, nbinsx=50,
                        ))
                        fig_hist.add_trace(go.Histogram(
                            x=df.loc[df["anomaly"] == 1, sensor], name="Anomaly",
                            marker_color=ACCENT, opacity=0.7, nbinsx=50,
                        ))
                        fig_hist.update_layout(
                            barmode="overlay", height=220, title=sensor,
                            showlegend=(idx == 0),
                            xaxis_title=SENSOR_EXPLAIN.get(sensor, {}).get("unit", ""),
                            **PLOTLY_LAYOUT,
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")

            # Correlation shift
            st.markdown("#### Correlation Structure: Normal vs Anomaly")
            st.caption(
                "These heatmaps show how sensors correlate with each other. "
                "If correlations **break** during anomalies (values change dramatically), "
                "it means fault conditions disrupt the normal sensor relationships — "
                "this is exploitable by models like autoencoders or PCA."
            )

            normal_corr, anom_corr = compute_correlation_matrices(df, scols)
            corr_diff = (anom_corr - normal_corr).abs()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Normal Correlation**")
                fig_cn = px.imshow(normal_corr, color_continuous_scale="RdBu_r",
                                   zmin=-1, zmax=1, aspect="equal", text_auto=".2f")
                fig_cn.update_layout(height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_cn, use_container_width=True)
            with c2:
                st.markdown("**Anomaly Correlation**")
                fig_ca = px.imshow(anom_corr, color_continuous_scale="RdBu_r",
                                   zmin=-1, zmax=1, aspect="equal", text_auto=".2f")
                fig_ca.update_layout(height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_ca, use_container_width=True)
            with c3:
                st.markdown("**Absolute Difference**")
                fig_cd = px.imshow(corr_diff, color_continuous_scale="YlOrRd",
                                   zmin=0, zmax=1, aspect="equal", text_auto=".2f")
                fig_cd.update_layout(height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_cd, use_container_width=True)

            # Find the biggest correlation break
            np.fill_diagonal(corr_diff.values, 0)
            max_break = corr_diff.max().max()
            max_pair = corr_diff.stack().idxmax()
            st.info(
                f"💡 The biggest correlation break is between **{max_pair[0]}** and **{max_pair[1]}** "
                f"(Δ = {max_break:.2f}). These two sensors have a fundamentally different relationship "
                f"during faults — a great feature pair for your model."
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Anomaly Close-up
# ══════════════════════════════════════════════════════════════════════════════

with tab_closeup:
    st.subheader("🔍 Anomaly Close-up")

    if sel_folder == "ALL":
        st.info("Select a specific folder and file in the sidebar to use this view.")
    elif n_anom == 0:
        st.info("This file has no anomalies — it's baseline data. Pick a fault file.")
    else:
        BUFFER = 30

        if len(intervals) > 1:
            labels = [f"Episode {i+1}: {s.strftime('%H:%M:%S')} → {e.strftime('%H:%M:%S')} ({int((e-s).total_seconds())}s)"
                      for i, (s, e) in enumerate(intervals)]
            chosen = st.selectbox("Anomaly episode", range(len(labels)),
                                  format_func=lambda x: labels[x])
        else:
            chosen = 0

        a_start, a_end = intervals[chosen]
        w_start = max(a_start - pd.Timedelta(seconds=BUFFER), df.index.min())
        w_end = min(a_end + pd.Timedelta(seconds=BUFFER), df.index.max())
        dfw = df.loc[w_start:w_end]
        ep_dur = int((a_end - a_start).total_seconds())
        w_intervals = anomaly_intervals(dfw)

        st.markdown(
            f"Zoomed to **{BUFFER}s before → {BUFFER}s after** the fault. "
            f"Episode lasted **{ep_dur} seconds**."
        )

        # Sensor reaction analysis
        st.markdown("#### Which sensors reacted first?")
        st.caption(
            "For each sensor, we measure the time between the changepoint (fault injection) "
            "and the first significant deviation from the pre-fault baseline. "
            "Earlier reaction = better early-warning feature."
        )

        cp_in_window = [cp for cp in cps if w_start <= cp <= w_end]

        # Compute reaction times from changepoint
        if cp_in_window:
            cp_time = cp_in_window[0]
            pre_fault = dfw.loc[:cp_time]
            post_fault = dfw.loc[cp_time:]

            reaction_rows = []
            for c in scols:
                baseline_mean = pre_fault[c].mean()
                baseline_std = pre_fault[c].std()
                if baseline_std > 0 and len(post_fault) > 0:
                    # First time post-fault exceeds 3σ from pre-fault mean
                    deviated = (post_fault[c] - baseline_mean).abs() > 3 * baseline_std
                    if deviated.any():
                        first_dev = deviated.idxmax()
                        delay_s = (first_dev - cp_time).total_seconds()
                    else:
                        delay_s = None
                else:
                    delay_s = None
                reaction_rows.append({"Sensor": c, "Reaction delay (s)": delay_s})

            react_df = pd.DataFrame(reaction_rows).sort_values("Reaction delay (s)")
            fast_sensors = react_df.dropna()

            if len(fast_sensors) > 0:
                st.dataframe(react_df.style.format({"Reaction delay (s)": "{:.0f}"}),
                             use_container_width=True)
                fastest = fast_sensors.iloc[0]
                st.success(
                    f"⚡ **{fastest['Sensor']}** reacted first — only "
                    f"**{int(fastest['Reaction delay (s)'])}s** after fault injection. "
                    f"Use this as your primary early-warning feature."
                )
            else:
                st.caption("No sensor crossed the 3σ threshold in this window.")

        st.markdown("---")

        # Multi-sensor subplot
        n = len(scols)
        fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                            vertical_spacing=0.015,
                            subplot_titles=[SENSOR_EXPLAIN.get(c, {}).get("what", c)[:50]
                                            for c in scols])

        for i, col in enumerate(scols, 1):
            fig.add_trace(go.Scatter(
                x=dfw.index, y=dfw[col], mode="lines",
                line=dict(color=PALETTE[(i-1) % len(PALETTE)], width=1.5),
                hovertemplate="%{x|%H:%M:%S}<br>%{y:.4f}<extra>" + col + "</extra>",
            ), row=i, col=1)

        for i in range(1, n + 1):
            add_shading(fig, w_intervals, row=i, col=1)
            for cp in cp_in_window:
                fig.add_vline(x=cp, line=dict(color=ACCENT, dash="dash", width=2),
                              row=i, col=1)
            fig.update_xaxes(showgrid=False, row=i, col=1)
            fig.update_yaxes(gridcolor="#F3F4F6", row=i, col=1)

        fig.update_layout(height=170 * n, showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — All Files Summary
# ══════════════════════════════════════════════════════════════════════════════

with tab_summary:
    st.subheader("📊 All Files Overview")

    summary = build_file_summary(tuple(sorted(file_map.items())))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files", len(summary))
    c2.metric("Total rows", f"{summary['Rows'].sum():,}")
    c3.metric("Anomaly rows", f"{summary['Anomaly rows'].sum():,}")
    total_pct = summary['Anomaly rows'].sum() / summary['Rows'].sum() * 100 if summary['Rows'].sum() else 0
    c4.metric("Overall anomaly %", f"{total_pct:.1f}%")

    # Class imbalance warning
    if total_pct < 40:
        st.warning(
            f"⚠️ **Class imbalance**: only {total_pct:.1f}% of rows are anomalous. "
            f"Accuracy will be misleading. Use **F1-score** or **AUC-ROC** as your primary metric. "
            f"Consider SMOTE, class weights, or anomaly detection (one-class SVM, isolation forest) "
            f"instead of binary classification."
        )

    st.markdown("---")

    # Detailed file table
    display_cols = [c for c in summary.columns if c != "path"]
    st.dataframe(summary[display_cols], use_container_width=True, height=400)

    st.markdown("---")

    # Bar chart by file
    st.markdown("#### Anomaly % per File")
    fig_bar = go.Figure()
    for folder in summary["Folder"].unique():
        sub = summary[summary["Folder"] == folder]
        fig_bar.add_trace(go.Bar(
            x=sub["Folder"] + "/" + sub["File"],
            y=sub["Anomaly %"],
            name=folder,
            marker_color=FOLDER_COLORS.get(folder, "#9CA3AF"),
            hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
        ))
    fig_bar.update_layout(barmode="group", height=380,
                          xaxis=dict(tickangle=-45, showgrid=False),
                          yaxis=dict(title="Anomaly %", gridcolor="#F3F4F6"),
                          legend=dict(orientation="h", y=1.08),
                          **PLOTLY_LAYOUT)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Folder-level
    st.markdown("#### Folder-level Breakdown")
    folder_agg = summary.groupby("Folder").agg(
        Files=("File", "count"),
        Total_rows=("Rows", "sum"),
        Anomaly_rows=("Anomaly rows", "sum"),
        Episodes=("# Anomaly episodes", "sum"),
        Avg_dur=("Duration (min)", "mean"),
    ).reset_index()
    folder_agg["Anomaly %"] = (folder_agg["Anomaly_rows"] / folder_agg["Total_rows"] * 100).round(1)
    folder_agg.columns = ["Folder", "Files", "Total rows", "Anomaly rows",
                          "Total episodes", "Avg duration (min)", "Anomaly %"]
    st.dataframe(folder_agg, use_container_width=True)

    # Insight
    most_episodes = folder_agg.sort_values("Total episodes", ascending=False).iloc[0]
    st.info(
        f"💡 **{most_episodes['Folder']}** has the most anomaly episodes "
        f"({int(most_episodes['Total episodes'])}). "
        f"More episodes = more fault transitions for your model to learn from."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Model-Ready
# ══════════════════════════════════════════════════════════════════════════════

with tab_model:
    st.subheader("🤖 Model-Ready Data")
    st.caption(
        "Prepare your data for training. This tab will grow as you build your model."
    )

    st.markdown("#### Feature Engineering Suggestions")
    st.markdown("""
Based on the data characteristics, here are recommended features for your model:

| Feature Type | Columns | Rationale |
|---|---|---|
| **Raw sensor values** | All 8 sensors | Direct signal — start here |
| **Rolling mean (10s)** | All sensors | Smooths noise, captures trends |
| **Rolling std (10s)** | All sensors | Captures volatility spikes during faults |
| **First derivative** | Pressure, Flow, Current | Rate of change reveals sudden shifts |
| **Sensor ratios** | Current/Voltage, Pressure/Flow | Physical relationships break during faults |
| **Lag features (1-5s)** | Top sensors from Insights tab | Captures temporal patterns |
    """)

    st.markdown("---")
    st.markdown("#### Preview: Engineered Features")
    st.caption("A sample of what the training data looks like with basic feature engineering applied.")

    @st.cache_data
    def engineer_features(df_input, scols_list):
        out = df_input[scols_list + ["anomaly"]].copy()
        for c in scols_list:
            out[f"{c}_rollmean10"] = out[c].rolling(10, min_periods=1).mean()
            out[f"{c}_rollstd10"] = out[c].rolling(10, min_periods=1).std().fillna(0)
            out[f"{c}_diff"] = out[c].diff().fillna(0)
        return out

    feat_df = engineer_features(df, scols)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Original features", len(scols))
    with col_b:
        feat_count = len([c for c in feat_df.columns if c != "anomaly"])
        st.metric("Engineered features", feat_count)

    st.dataframe(feat_df.head(100).round(4), use_container_width=True, height=300)

    st.markdown("---")
    st.markdown("#### Train / Test Split Strategy")

    if sel_folder == "ALL" and "_folder" in df.columns:
        normal_rows = int((df["anomaly"] == 0).sum())
        anom_rows_count = int((df["anomaly"] == 1).sum())
        st.markdown(f"""
**The split actually used by the research pipeline (`main.py`):**

| Split | Source | Rows | Purpose |
|---|---|---|---|
| **Train** | `anomaly-free` + the *normal-labeled* rows from the first 80% (by time) of each fault file | ~28,400 | Learn what "normal" looks like (0% anomalies — see DEVLOG session 3) |
| **Validation** | whole held-out fault files (~20% of files) | ~5,600 | Tune hyperparameters only (see DEVLOG session 4) |
| **Test** | the last 20% (by time) of each fault file | ~7,500 | Final evaluation — never touched during training/tuning |

**Why this split works:**
- These models learn "normal" and flag deviations, so the training set is kept 100% normal (mixing in faults hurt every model — see the contamination fix in DEVLOG session 3).
- Splitting each file chronologically (never shuffled) respects that this is time-series data.
- Validation holds out *whole experiments* (not a time-slice) so its anomaly mix stays representative.
- You have **{normal_rows:,} normal** and **{anom_rows_count:,} anomalous** rows total.

See the **🏆 Model Results** tab for the actual scores.
        """)
    else:
        st.caption("Select **ALL** in the sidebar to see the recommended train/test split.")

    st.markdown("---")
    st.markdown("#### Export Combined Dataset")
    st.caption("Download the full combined dataset with engineered features as CSV for use in notebooks.")

    if st.button("Prepare CSV Export"):
        csv_data = feat_df.to_csv()
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name="skab_engineered_features.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Model Results (the actual 4-model comparison from main.py)
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    st.subheader("🏆 Model Results — 4-Model Comparison")
    st.caption(
        "The real trained-model results from the research pipeline. "
        "Run `python main.py` to (re)generate `results/metrics.json` and the figures, "
        "then reload this page to see updated numbers."
    )

    if not os.path.exists(METRICS_PATH):
        st.warning(
            "No results found yet. Run **`python main.py`** from the project root "
            "to train the 4 models and generate the metrics + figures, then reload."
        )
    else:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        mdf = pd.DataFrame(metrics)

        best_row = mdf.loc[mdf["f1"].idxmax()]
        best_model = best_row["model"]

        # Headline metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model (by F1)", best_model)
        c2.metric("Best F1", f"{best_row['f1']:.3f}")
        c3.metric("Its precision", f"{best_row['precision']:.3f}")
        c4.metric("Its recall", f"{best_row['recall']:.3f}")

        st.info(
            f"💡 In plain English: **{best_model}** is the best of the 4 at catching faults "
            f"without too many false alarms. **Recall {best_row['recall']:.2f}** means it caught "
            f"about {best_row['recall']*100:.0f}% of the real faults; **precision {best_row['precision']:.2f}** "
            f"means about {best_row['precision']*100:.0f}% of its fault alerts were correct. "
            f"**F1 ({best_row['f1']:.2f})** balances those two into one number."
        )

        st.markdown("#### Comparison table")
        st.caption(
            "Precision = of the rows it flagged, how many were truly faults. "
            "Recall = of the true faults, how many it caught. F1 = balance of the two. "
            "Times show how long each model took to train and to predict the whole test set."
        )
        display = mdf[["model", "precision", "recall", "f1",
                       "train_time_s", "inference_time_s"]].rename(columns={
            "model": "Model", "precision": "Precision", "recall": "Recall", "f1": "F1",
            "train_time_s": "Train time (s)", "inference_time_s": "Inference time (s)",
        })
        styler = (display.style
                  .format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}",
                           "Train time (s)": "{:.3f}", "Inference time (s)": "{:.3f}"})
                  .highlight_max(subset=["F1"], color="#FEF3C7"))
        st.dataframe(styler, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Figures (the same PNGs that go into the paper)")

        for fname, title, explanation in RESULT_FIGURES:
            fpath = os.path.join(FIGURES_DIR, fname)
            st.markdown(f"**{title}**")
            if os.path.exists(fpath):
                st.image(fpath, use_container_width=True)
                st.caption(explanation)
            else:
                st.caption(f"_`{fname}` not found — run `python main.py` to generate it._")
            st.markdown("")
