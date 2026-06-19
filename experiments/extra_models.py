"""Can a higher-ranked, cost-free model beat Isolation Forest? (ROADMAP B6.)

Tests two scikit-learn models that rank above Isolation Forest in theory
because they model the correlation structure between sensors:
  - PCA reconstruction error
  - Elliptic Envelope (robust Mahalanobis distance)

Both get the same fair treatment the original 4 received: their contamination
is tuned on the validation files (whole held-out experiments, never the test
set), then they're evaluated once on the locked test set. The existing 4
models are run with their committed settings for a full 6-way comparison.
Bar to beat: Isolation Forest test F1 = 0.466.
"""

import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from src.preprocessing import chronological_split, file_holdout_split, load_anomaly_free, load_skab_all
from src.features import build_features
from src.models import (
    scale_features, MODELS,
    train_pca_recon, predict_pca_recon,
    train_elliptic_envelope, predict_elliptic_envelope,
)
from src.evaluate import evaluate

CONTAM_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
NEW_MODELS = {
    "PCA-recon": (train_pca_recon, predict_pca_recon),
    "EllipticEnvelope": (train_elliptic_envelope, predict_elliptic_envelope),
}


def make_anomaly_free():
    af = load_anomaly_free()
    af["experiment_type"] = "anomaly-free"
    af["source_file"] = "anomaly-free/anomaly-free"
    af["anomaly"] = 0
    af["changepoint"] = 0
    return af


def tune_contamination(name, train_fn, predict_fn, X_fit, X_val, y_val):
    best_f1, best_c = -1.0, CONTAM_GRID[0]
    for c in CONTAM_GRID:
        state = train_fn(X_fit, contamination=c)
        f1 = evaluate(y_val, predict_fn(state, X_val), name)["f1"]
        if f1 > best_f1:
            best_f1, best_c = f1, c
    print(f"{name:18s} best contamination={best_c} (val F1={best_f1:.3f})")
    return best_c


def main():
    af = make_anomaly_free()
    fault_df = load_skab_all()
    fault_train, fault_test = chronological_split(fault_df)
    fault_fit, fault_val = file_holdout_split(fault_train, val_frac=0.2)

    # Feature matrices (leakage-free: include_clock=False is the default)
    fit_df = pd.concat([af, fault_fit[fault_fit["anomaly"] == 0]])
    train_df = pd.concat([af, fault_train[fault_train["anomaly"] == 0]])

    X_fit, _ = build_features(fit_df, group_col="source_file")
    X_val, _ = build_features(fault_val, group_col="source_file")
    X_train, _ = build_features(train_df, group_col="source_file")
    X_test, _ = build_features(fault_test, group_col="source_file")
    y_val = fault_val["anomaly"].to_numpy()
    y_test = fault_test["anomaly"].to_numpy()

    Xf_s, Xv_s, _ = scale_features(X_fit, X_val)
    Xtr_s, Xte_s, _ = scale_features(X_train, X_test)

    print("=== tuning new models on validation ===")
    rows = []
    for name, (train_fn, predict_fn) in NEW_MODELS.items():
        best_c = tune_contamination(name, train_fn, predict_fn, Xf_s, Xv_s, y_val)
        state = train_fn(Xtr_s, contamination=best_c)
        r = evaluate(y_test, predict_fn(state, Xte_s), name)
        r["contamination"] = best_c
        rows.append(r)

    # Existing 4 with their committed settings
    for name, (train_fn, predict_fn) in MODELS.items():
        state = train_fn(Xtr_s)
        rows.append(evaluate(y_test, predict_fn(state, Xte_s), name))

    df = pd.DataFrame(rows)[["model", "precision", "recall", "f1"]].sort_values("f1", ascending=False)
    print("\n=== full test-set comparison (sorted by F1) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best = df.iloc[0]
    print(f"\nBest overall: {best['model']} (F1={best['f1']:.3f})")


if __name__ == "__main__":
    main()
