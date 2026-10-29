"""Stage 1 of the second-dataset study: run the SAME 6-model comparison on
SWaT that main.py runs on SKAB, to test whether the SKAB findings generalise.

Protocol mirrors main.py / hyperparameter_sweep.py:
  - train on NORMAL rows only (unsupervised, 0% contamination);
  - tune the shared contamination on a VALIDATION split (never the test set),
    selecting by MCC and rejecting degenerate "flag-everything" winners
    (recall > 0.99) -- the same anti-degenerate discipline as SKAB's B4;
  - report on a held-out TEST split containing both normal and attack rows,
    so both false alarms and detections are measured;
  - identical feature recipe (src/features.build_features) and identical 6
    models (src/models).

Splits, all leakage-safe and chronological:
  - normal run -> 70% train / 15% validation-normal / 15% test-normal;
  - attack run -> 50% validation-attack / 50% test-attack.
Features are built AFTER splitting, per segment, so a rolling window never
crosses a split or a normal/attack boundary (as main.py does per source_file).

DOWNSAMPLE thins the modelled rows by an integer step; features are computed
at full 1 Hz first so the 5 s / 30 s windows keep their SKAB meaning. The
canonical run uses DOWNSAMPLE=1 (every row, the complete dataset). The step
is reported and saved -- nothing hidden, and all models see identical data.
"""

import json
import time

import pandas as pd

from src.swat import CONTINUOUS_SENSORS, load_swat
from src.features import build_features
from src.models import (
    scale_features, train_zscore, predict_zscore,
    train_isolation_forest, predict_isolation_forest,
    train_one_class_svm, predict_one_class_svm,
    train_lof, predict_lof, train_pca_recon, predict_pca_recon,
    train_elliptic_envelope, predict_elliptic_envelope,
)
from src.evaluate import evaluate
from src.visualize import plot_model_comparison, plot_confusion_matrices

RESULTS_PATH = "results/initial_analysis/swat_metrics.json"
DOWNSAMPLE = 1
CONTAMINATION_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# train_fn, predict_fn, and whether the model takes the shared contamination
# (Z-score uses a fixed z-threshold instead, exactly as on SKAB).
MODEL_SPECS = {
    "Z-score":          (train_zscore, predict_zscore, False),
    "One-Class SVM":    (train_one_class_svm, predict_one_class_svm, True),
    "LOF":              (train_lof, predict_lof, True),
    "PCA-recon":        (train_pca_recon, predict_pca_recon, True),
    "EllipticEnvelope": (train_elliptic_envelope, predict_elliptic_envelope, True),
    "IsolationForest":  (train_isolation_forest, predict_isolation_forest, True),
}


def _feats(df):
    X, names = build_features(df, group_col="segment", sensor_cols=CONTINUOUS_SENSORS)
    return X[::DOWNSAMPLE], df["anomaly"].to_numpy()[::DOWNSAMPLE], names


def build_swat_splits():
    normal, attack = load_swat()
    n = len(normal)
    tr_end, va_end = int(n * 0.70), int(n * 0.85)
    normal_train = normal.iloc[:tr_end].assign(segment="normal_train")
    normal_val = normal.iloc[tr_end:va_end].assign(segment="normal_val")
    normal_test = normal.iloc[va_end:].assign(segment="normal_test")

    a = len(attack)
    attack_val = attack.iloc[:a // 2].assign(segment="attack_val")
    attack_test = attack.iloc[a // 2:].assign(segment="attack_test")

    X_train, _, names = _feats(normal_train)
    X_val, y_val, _ = _feats(pd.concat([normal_val, attack_val]))
    X_test, y_test, _ = _feats(pd.concat([normal_test, attack_test]))

    info = {
        "normal_rows_total": int(n), "attack_rows_total": int(a),
        "train_rows": int(len(X_train)), "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)), "test_anomaly_rate": float(y_test.mean()),
        "downsample": DOWNSAMPLE, "n_features": len(names),
    }
    return X_train, X_val, y_val, X_test, y_test, info


def tune_contamination(X_train_s, X_val_s, y_val):
    """Shared contamination from IsolationForest's validation MCC peak,
    rejecting degenerate flag-everything winners (recall > 0.99)."""
    print("contamination sweep (IsolationForest on validation, by MCC):")
    best_c, best_mcc = None, -2.0
    for c in CONTAMINATION_GRID:
        m = train_isolation_forest(X_train_s, contamination=c)
        r = evaluate(y_val, predict_isolation_forest(m, X_val_s), "IF")
        degen = r["recall"] > 0.99
        flag = "  (degenerate, rejected)" if degen else ""
        print(f"  c={c:.2f}  P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} MCC={r['mcc']:+.3f}{flag}")
        if not degen and r["mcc"] > best_mcc:
            best_c, best_mcc = c, r["mcc"]
    print(f"chosen contamination = {best_c} (val MCC={best_mcc:+.3f})")
    return best_c


def main():
    t0 = time.perf_counter()
    X_train, X_val, y_val, X_test, y_test, info = build_swat_splits()
    print("SWaT splits prepared:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    X_train_s, X_val_s, scaler = scale_features(X_train, X_val)
    X_test_s = scaler.transform(X_test)

    contamination = tune_contamination(X_train_s, X_val_s, y_val)

    print("\ntraining + scoring 6 models on the held-out test set...")
    results = []
    for name, (train_fn, predict_fn, uses_c) in MODEL_SPECS.items():
        tt = time.perf_counter()
        state = train_fn(X_train_s, contamination) if uses_c else train_fn(X_train_s)
        train_t = time.perf_counter() - tt
        tt = time.perf_counter()
        y_pred = predict_fn(state, X_test_s)
        infer_t = time.perf_counter() - tt
        r = evaluate(y_test, y_pred, name)
        r["train_time_s"], r["inference_time_s"] = train_t, infer_t
        results.append(r)
        print(f"  {name:17s} MCC={r['mcc']:+.3f} bAcc={r['balanced_accuracy']:.3f} "
              f"F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"({train_t:.1f}s train)")

    table = pd.DataFrame(results)[
        ["model", "precision", "recall", "f1", "mcc", "balanced_accuracy"]
    ].sort_values("mcc", ascending=False)
    print("\n=== SWaT 6-model comparison (sorted by MCC) ===")
    print(table.to_string(index=False))

    with open(RESULTS_PATH, "w") as f:
        json.dump({"info": info, "contamination": contamination, "results": results}, f, indent=2)
    plot_model_comparison(results, out_path="results/initial_analysis/figures/swat_model_comparison.png")
    plot_confusion_matrices(results, out_path="results/initial_analysis/figures/swat_confusion_matrices.png")
    print(f"\nsaved metrics: {RESULTS_PATH}")
    print(f"total wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
