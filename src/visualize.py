"""Paper-quality figures for the 4-model comparison (ROADMAP C1).

All figures save as 300 DPI PNGs to results/initial_analysis/figures/ so they can drop
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

FIGURES_DIR = "results/initial_analysis/figures"
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


STATIC_COLOR = "#6B7280"    # grey — the never-updated baseline
ADAPTIVE_COLOR = "#2563EB"  # blue — the drift-adaptive model
RETRAIN_COLOR = "#16A34A"   # green — retrain events


def plot_drift_adaptation(static_records, adaptive_records, out_path=None,
                          metric="balanced_accuracy"):
    """Headline figure for the concept-drift experiment: a chosen skill
    metric over the stream, static vs adaptive, with the test files ordered
    by fault family (the regime shift). balanced_accuracy is the default
    because its 0.5 line is an interpretable "no better than chance" floor:
    the static model sits on that floor across the unseen valve regimes
    (it can only flag-everything there), while the adaptive model rises
    above it after drift-triggered retrains. Driven by the per-batch records
    in results/initial_analysis/concept_drift.json, so it regenerates independently of
    main.py's 6-model figures.
    """
    out_path = out_path or os.path.join(FIGURES_DIR, "drift_adaptation.png")
    s = pd.DataFrame(static_records)
    a = pd.DataFrame(adaptive_records)
    x = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(12, 5))

    # shade contiguous fault-family bands so the regime shift is visible
    families = list(s["family"])
    band_colors = {"other": "#F3F4F6", "valve1": "#FEF3C7", "valve2": "#FFE4E6"}
    start = 0
    for i in range(1, len(families) + 1):
        if i == len(families) or families[i] != families[start]:
            fam = families[start]
            ax.axvspan(start - 0.5, i - 0.5, color=band_colors.get(fam, "#F3F4F6"),
                       alpha=0.7, zorder=0)
            ax.text((start + i - 1) / 2, 0.02, fam, ha="center", va="bottom",
                    fontsize=9, color="#6B7280")
            start = i

    ax.axhline(0.5, color="#9CA3AF", linestyle=":", linewidth=1,
               label="chance (0.5)")
    ax.plot(x, s[metric], "-o", color=STATIC_COLOR, markersize=4,
            label="static (never retrains)")
    ax.plot(x, a[metric], "-o", color=ADAPTIVE_COLOR, markersize=4,
            label="adaptive (retrains on drift)")

    for xi in x[a["retrained"].to_numpy()]:
        ax.axvline(xi, color=RETRAIN_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    # one proxy handle for the retrain markers in the legend
    ax.plot([], [], color=RETRAIN_COLOR, linestyle="--", linewidth=1,
            label="retrain event")

    ax.set_xticks(x)
    ax.set_xticklabels(s["source_file"], rotation=90, fontsize=6)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1)
    ax.set_title("Concept-drift adaptation: skill over the regime-shifting stream")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


SKAB_COLOR = "#0EA5E9"      # sky — SKAB dataset
SWAT_COLOR = "#9333EA"      # violet — SWaT dataset


def plot_cross_dataset_comparison(results_by_dataset, metric="mcc", out_path=None,
                                  model_order=None):
    """Grouped bars: each model's `metric` on SKAB vs SWaT, side by side. This
    is the "the best detector is dataset-dependent" figure -- Isolation Forest
    leads on SKAB, Elliptic Envelope on SWaT. results_by_dataset maps a dataset
    label to its list of per-model result dicts (each with 'model' + metric).
    model_order defaults to the six classical models; pass an extended list to
    include the deep baseline.
    """
    out_path = out_path or os.path.join(FIGURES_DIR, "cross_dataset_comparison.png")
    order = model_order or MODEL_ORDER
    labels, colors = list(results_by_dataset.keys()), [SKAB_COLOR, SWAT_COLOR]
    lookup = {ds: {r["model"]: r.get(metric, 0.0) for r in rows}
              for ds, rows in results_by_dataset.items()}

    x = np.arange(len(order))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, ds in enumerate(labels):
        vals = [lookup[ds].get(m, 0.0) for m in order]
        bars = ax.bar(x + i * width, vals, width, label=ds,
                      color=colors[i % len(colors)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.03),
                    f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(0, color="#9CA3AF", linewidth=0.8)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"The best detector is dataset-dependent ({metric.upper()} per model)")
    ax.legend(title="Dataset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_edge_benchmark(skab_rows, out_path=None, ram_mb=None, cpu=None):
    """Two-panel deployability figure from the constrained-container run:
    per-model inference latency (ms/sample) and on-disk model size (KB), both
    log-scaled because they span orders of magnitude. Backs the "runs on cheap
    edge hardware" claim.
    """
    out_path = out_path or os.path.join(FIGURES_DIR, "edge_benchmark.png")
    by_name = {r["model"]: r for r in skab_rows}
    models = [m for m in MODEL_ORDER if m in by_name]
    latency = [by_name[m]["infer_ms_per_sample"] for m in models]
    size = [by_name[m]["size_KB"] for m in models]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, vals, title, ylab, fmt in [
        (a1, latency, "Inference latency (lower = faster)", "ms per reading (log)", "{:.4f}"),
        (a2, size, "Model size on disk (lower = lighter)", "KB (log)", "{:.0f}"),
    ]:
        bars = ax.bar(models, vals, color="#2563EB")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xticklabels(models, rotation=20, ha="right")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=7)
    cap = "Constrained ARM container"
    if cpu is not None:
        cap += f": {cpu} CPU"
    if ram_mb is not None:
        cap += f", peak {ram_mb:.0f} MB of 1024 MB"
    fig.suptitle(f"Edge deployability — {cap}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_swat_drift_motivation(positions_h, fp_rate, ks_drift, out_path=None):
    """Why drift matters on SWaT: a static detector trained on early normal
    data has its false-alarm rate on *later normal* data climb toward 100% as
    the sensor baselines drift, tracked by the rising KS drift score. Dual
    y-axis over the timeline (hours since training window).
    """
    out_path = out_path or os.path.join(FIGURES_DIR, "swat_drift_motivation.png")
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(positions_h, np.array(fp_rate) * 100, "-o", color=TRUE_SHADE,
             label="false-alarm rate on normal")
    ax1.set_xlabel("hours into the run (past the training window)")
    ax1.set_ylabel("false alarms on NORMAL data (%)", color=TRUE_SHADE)
    ax1.tick_params(axis="y", labelcolor=TRUE_SHADE)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(positions_h, ks_drift, "-s", color="#2563EB", label="KS drift score")
    ax2.set_ylabel("KS drift score vs. training data", color="#2563EB")
    ax2.tick_params(axis="y", labelcolor="#2563EB")
    ax2.set_ylim(0, max(0.6, max(ks_drift) * 1.1))

    ax1.set_title("SWaT normal operation drifts: a static detector decays into "
                  "flagging everything")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_training_loss(losses_by_dataset, out_path=None):
    """LSTM-autoencoder training convergence: reconstruction MSE per epoch for
    each dataset. Shows the model was trained to convergence (rebuts an
    ``under-trained'' reading of the deep result)."""
    out_path = out_path or os.path.join(FIGURES_DIR, "deep_training_loss.png")
    colors = {"SKAB": SKAB_COLOR, "SWaT": SWAT_COLOR}
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds, losses in losses_by_dataset.items():
        ax.plot(np.arange(1, len(losses) + 1), losses, "-o", ms=3,
                color=colors.get(ds), label=ds)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training reconstruction MSE (log)")
    ax.set_title("LSTM autoencoder training convergence")
    ax.legend(title="Dataset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_recon_error_hist(panels, out_path=None):
    """Why the deep model separates on one dataset and not the other: per-panel
    box plots (log scale) of the reconstruction error for normal vs. anomaly
    rows, with the decision threshold. panels is a list of
    (title, scores, labels, threshold). Box plots are used rather than
    histograms because the anomaly errors span several orders of magnitude,
    which a density histogram smears into invisibility. A clear gap between the
    boxes (SWaT) means the error separates the classes; overlapping boxes
    (SKAB) mean it does not. Fliers are hidden so the boxes stay readable."""
    out_path = out_path or os.path.join(FIGURES_DIR, "deep_recon_error.png")
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, scores, labels, thr) in zip(axes, panels):
        s = np.clip(np.asarray(scores, dtype=float), 1e-5, None)
        y = np.asarray(labels)
        bp = ax.boxplot([s[y == 0], s[y == 1]], positions=[1, 2], widths=0.6,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], ["#374151", TRUE_SHADE]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.axhline(thr, color="#2563EB", linestyle="--", linewidth=1.2,
                   label="threshold")
        ax.set_yscale("log")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["normal", "anomaly"])
        ax.set_ylabel("reconstruction error (log)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Deep model reconstruction error by class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


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
