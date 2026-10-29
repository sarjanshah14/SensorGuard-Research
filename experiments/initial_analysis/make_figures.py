"""Generate the cross-cutting paper figures that aren't produced by a single
experiment run: the cross-dataset winner comparison, the edge-deployability
chart, and the SWaT drift-motivation chart. Reads the saved result JSONs
(and, for drift motivation, recomputes a small trajectory from SWaT). All
figures are our own, generated from our own results.

Run: python -m experiments.initial_analysis.make_figures
"""

import json

import numpy as np

from src.visualize import (
    plot_cross_dataset_comparison, plot_edge_benchmark, plot_swat_drift_motivation,
    plot_training_loss, plot_recon_error_hist,
)


def cross_dataset():
    skab = json.load(open("results/initial_analysis/metrics.json"))
    swat = json.load(open("results/initial_analysis/swat_metrics.json"))["results"]
    order = ["Z-score", "One-Class SVM", "LOF", "PCA-recon",
             "EllipticEnvelope", "IsolationForest"]
    # append the LSTM-AE deep baseline if it has been run
    try:
        deep = json.load(open("results/initial_analysis/deep_baseline.json"))
        skab = skab + [deep["SKAB"]]
        swat = swat + [deep["SWaT"]]
        order = order + ["LSTM-AE"]
    except FileNotFoundError:
        pass
    p = plot_cross_dataset_comparison({"SKAB": skab, "SWaT": swat},
                                      metric="mcc", model_order=order)
    print(f"  {p}")


def edge():
    e = json.load(open("results/initial_analysis/edge_benchmark.json"))
    p = plot_edge_benchmark(e["skab"], ram_mb=e.get("peak_ram_mb"),
                            cpu=e.get("env", {}).get("cpu_limit_cgroup"))
    print(f"  {p}")


def swat_drift_motivation():
    """Recompute the false-alarm-climbs-as-normal-drifts trajectory and plot it."""
    from src.swat import load_swat, CONTINUOUS_SENSORS
    from src.features import build_features
    from src.models import scale_features, train_isolation_forest, predict_isolation_forest
    from src.drift import drift_score

    normal, _ = load_swat()
    normal = normal.assign(segment="n")
    X, _ = build_features(normal, group_col="segment", sensor_cols=CONTINUOUS_SENSORS)
    ref = X[:50000]
    ref_s, _, scaler = scale_features(ref, ref)
    model = train_isolation_forest(ref_s, contamination=0.05)

    pos_h, fp, ks = [], [], []
    CH = 25000
    for s in range(0, len(X) - CH, 50000):
        chunk = scaler.transform(X[s:s + CH])
        pos_h.append(round(s / 3600.0, 1))          # rows at 1 Hz -> hours
        fp.append(float(predict_isolation_forest(model, chunk).mean()))
        ks.append(float(drift_score(ref_s, chunk)))

    json.dump({"hours": pos_h, "fp_rate": fp, "ks_drift": ks},
              open("results/initial_analysis/swat_drift_motivation.json", "w"), indent=2)
    p = plot_swat_drift_motivation(pos_h, fp, ks)
    print(f"  {p}")


def deep_figures():
    """Training-loss curves and reconstruction-error distributions for the
    LSTM-AE, from results/initial_analysis/deep_baseline_arrays.npz (skip if not run yet)."""
    try:
        a = np.load("results/initial_analysis/deep_baseline_arrays.npz")
    except FileNotFoundError:
        print("  (deep_baseline_arrays.npz not found; run experiments.deep_baseline first)")
        return
    deep = json.load(open("results/initial_analysis/deep_baseline.json"))
    losses = {ds: a[f"{ds}_losses"] for ds in ("SKAB", "SWaT")}
    print(f"  {plot_training_loss(losses)}")
    panels = []
    for ds in ("SKAB", "SWaT"):
        title = f"{ds}  (MCC {deep[ds]['mcc']:.2f})"
        panels.append((title, a[f"{ds}_scores"], a[f"{ds}_labels"], float(a[f"{ds}_thr"][0])))
    print(f"  {plot_recon_error_hist(panels)}")


def main():
    print("cross-dataset winner comparison:")
    cross_dataset()
    print("edge deployability:")
    edge()
    print("SWaT drift motivation (recomputing trajectory):")
    swat_drift_motivation()
    print("deep baseline figures:")
    deep_figures()


if __name__ == "__main__":
    main()
