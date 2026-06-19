"""Paper-quality figures for the 4-model comparison (ROADMAP C1).

All figures save as 300 DPI PNGs to results/figures/ so they can drop
straight into the paper. The Agg backend is forced so this runs headless
(no GUI windows) inside main.py.

Two families of figure:
  - aggregate (model_comparison, confusion_matrices): driven by the
    results list from evaluate.run_all_models — the whole test set.
  - per-file (time_series_with_anomalies, anomaly_closeup): driven by a
    single SKAB test run's rows plus that run's predicted labels, for a
    concrete "here's what detection looks like" illustration.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.features import SENSOR_COLS

FIGURES_DIR = "results/figures"
DPI = 300
MODEL_ORDER = ["Z-score", "One-Class SVM", "LOF", "PCA-recon",
               "EllipticEnvelope", "IsolationForest"]
TRUE_SHADE = "#EF4444"      # red — ground-truth anomaly regions
PRED_COLOR = "#F97316"      # orange — predicted anomaly points


def _intervals(series: pd.Series):
    """Contiguous (start, end) index ranges where an int 0/1 series == 1."""
    mask = series.astype(int)
    if mask.sum() == 0:
        return []
    diff = mask.diff().fillna(0)
    starts = list(series.index[diff == 1])
    ends = list(series.index[diff == -1])
    if mask.iloc[0] == 1:
        starts.insert(0, series.index[0])
    if mask.iloc[-1] == 1:
        ends.append(series.index[-1])
    return list(zip(starts, ends))


def plot_model_comparison(results, out_path=None):
    """Grouped bar chart: precision/recall/F1 on the x-axis, one bar per
    model in each group."""
    out_path = out_path or os.path.join(FIGURES_DIR, "model_comparison.png")
    df = pd.DataFrame(results)[["model", "precision", "recall", "f1"]]
    long = df.melt(id_vars="model", var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x="metric", y="score", hue="model",
                hue_order=MODEL_ORDER, ax=ax)
    ax.set_title("Model comparison on SKAB test set")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_confusion_matrices(results, out_path=None):
    """2x2 grid of confusion-matrix heatmaps, one per model."""
    out_path = out_path or os.path.join(FIGURES_DIR, "confusion_matrices.png")
    by_name = {r["model"]: r for r in results}

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        cm = by_name[name]["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"], ax=ax)
        ax.set_title(f"{name} (F1={by_name[name]['f1']:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices on SKAB test set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def _plot_sensor_panels(df, pred_col, title, out_path, changepoint=None):
    """One stacked subplot per sensor: raw trace, true-anomaly shading,
    predicted-anomaly dots, optional changepoint vertical line."""
    sensors = [c for c in SENSOR_COLS if c in df.columns]
    true_intervals = _intervals(df["anomaly"]) if "anomaly" in df.columns else []
    pred_points = df.index[df[pred_col] == 1]

    fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 1.6 * len(sensors)),
                             sharex=True)
    for ax, col in zip(axes, sensors):
        ax.plot(df.index, df[col], linewidth=0.8, color="#374151")
        for start, end in true_intervals:
            ax.axvspan(start, end, color=TRUE_SHADE, alpha=0.12)
        if len(pred_points):
            ax.scatter(pred_points, df.loc[pred_points, col], s=6,
                       color=PRED_COLOR, alpha=0.6, zorder=3)
        if changepoint is not None:
            ax.axvline(changepoint, color=TRUE_SHADE, linestyle="--", linewidth=1.2)
        ax.set_ylabel(col, fontsize=7)

    # one shared legend explaining the overlays
    handles = [
        plt.Line2D([0], [0], color="#374151", lw=1, label="sensor signal"),
        plt.Rectangle((0, 0), 1, 1, color=TRUE_SHADE, alpha=0.12, label="true anomaly"),
        plt.Line2D([0], [0], marker="o", color=PRED_COLOR, lw=0, label="predicted anomaly"),
    ]
    if changepoint is not None:
        handles.append(plt.Line2D([0], [0], color=TRUE_SHADE, linestyle="--",
                                  lw=1.2, label="changepoint"))
    axes[0].legend(handles=handles, fontsize=7, loc="upper right", ncol=len(handles))
    axes[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_time_series_with_anomalies(df, pred_col, model_name, out_path=None):
    out_path = out_path or os.path.join(FIGURES_DIR, "timeseries_anomalies.png")
    fname = df["source_file"].iloc[0] if "source_file" in df.columns else ""
    title = f"{model_name} predictions vs. ground truth — {fname}"
    return _plot_sensor_panels(df, pred_col, title, out_path)


def plot_anomaly_closeup(df, pred_col, model_name, out_path=None, window_seconds=30):
    """Zoom to ±window_seconds around the first changepoint (fall back to the
    first anomaly onset if a file has no marked changepoint)."""
    out_path = out_path or os.path.join(FIGURES_DIR, "anomaly_closeup.png")

    cp_times = df.index[df["changepoint"] == 1] if "changepoint" in df.columns else []
    if len(cp_times):
        center = cp_times[0]
    else:
        onsets = _intervals(df["anomaly"])
        if not onsets:
            return None
        center = onsets[0][0]

    window = pd.Timedelta(seconds=window_seconds)
    sub = df[(df.index >= center - window) & (df.index <= center + window)]
    fname = df["source_file"].iloc[0] if "source_file" in df.columns else ""
    title = f"{model_name} — anomaly close-up (±{window_seconds}s) — {fname}"
    return _plot_sensor_panels(sub, pred_col, title, out_path, changepoint=center)
