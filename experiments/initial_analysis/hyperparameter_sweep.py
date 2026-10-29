"""Hyperparameter sweep (ROADMAP.md checkpoint B4).

contamination (IsolationForest/LOF), nu (One-Class SVM), and LOF's
n_neighbors were all carried over from the old SensorGuard code or the
original brief as nominal defaults (0.1, 0.1, 20) — never tuned for SKAB.
Z-score's threshold (3.0 std devs) is in the same boat.

To tune these honestly we cannot just try values and check against
results/initial_analysis/metrics.json's test set — picking a hyperparameter based on test
performance leaks the test set into the "final" numbers, the same way
checking the answer key while studying would. So fault_train (the 80%
chronological slice already held out from fault_test) is split *again*,
this time via file_holdout_split (whole experiments held out, not another
time-slice) into:
  - fault_fit  (~80% of fault_train's files) -> filtered to normal-only,
    paired with anomaly-free.csv, exactly like main.py's training set.
  - fault_val  (~20% of fault_train's files, kept whole) -> kept with real
    labels, used only to score candidate hyperparameters. Never trained
    on, never the final reported number.
fault_test (the original 20%, by time, per file) stays completely
untouched until the very end, once, with whichever hyperparameters win on
fault_val.

First attempt used a second *time-based* slice (last 20% of fault_train)
for fault_val instead of file_holdout_split, and it produced a validation
set that was 98% anomalous — because that time slice happened to land
inside the sustained-fault block most SKAB runs have in their back half.
Switched to holding out whole files instead, which keeps each held-out
run's natural mix of normal/anomalous rows intact.
"""

import pandas as pd

from src.preprocessing import chronological_split, file_holdout_split, load_anomaly_free, load_skab_all
from src.features import build_features
from src.models import (
    scale_features,
    train_isolation_forest, predict_isolation_forest,
    train_one_class_svm, predict_one_class_svm,
    train_lof, predict_lof,
    train_zscore, predict_zscore,
)
from src.evaluate import evaluate


def make_anomaly_free():
    anomaly_free = load_anomaly_free()
    anomaly_free["experiment_type"] = "anomaly-free"
    anomaly_free["source_file"] = "anomaly-free/anomaly-free"
    anomaly_free["anomaly"] = 0
    anomaly_free["changepoint"] = 0
    return anomaly_free


def sweep(name, candidates, train_fn, predict_fn, X_fit, X_val, y_val, param_name):
    rows = []
    for value in candidates:
        model = train_fn(X_fit, **{param_name: value})
        y_pred = predict_fn(model, X_val)
        result = evaluate(y_val, y_pred, f"{name}({param_name}={value})")
        result[param_name] = value
        rows.append(result)
    df = pd.DataFrame(rows)[[param_name, "precision", "recall", "f1"]]
    best = df.loc[df["f1"].idxmax()]
    print(f"\n=== {name} ===")
    print(df.to_string(index=False))
    print(f"best {param_name} = {best[param_name]} (f1={best['f1']:.3f})")
    return best[param_name]


def main():
    anomaly_free = make_anomaly_free()
    fault_df = load_skab_all()
    fault_train, fault_test = chronological_split(fault_df, test_size=0.2)
    fault_fit, fault_val = file_holdout_split(fault_train, val_frac=0.2)

    fit_train_df = pd.concat([anomaly_free, fault_fit[fault_fit["anomaly"] == 0]])

    X_fit, _ = build_features(fit_train_df, group_col="source_file")
    X_val, _ = build_features(fault_val, group_col="source_file")
    y_val = fault_val["anomaly"].to_numpy()

    X_fit_scaled, X_val_scaled, _ = scale_features(X_fit, X_val)

    print(f"fit rows: {len(fit_train_df)} | val rows: {len(fault_val)} (anomaly rate {y_val.mean():.3f}) | test rows (untouched): {len(fault_test)}")

    # 0.5 is the max sklearn allows for contamination/nu (it's a fraction,
    # so anything past "half the data is abnormal" stops making sense).
    # Covering the full valid range so a "best" pick is never just an
    # artifact of where the grid happened to stop.
    grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    best_if = sweep("IsolationForest", grid, train_isolation_forest, predict_isolation_forest,
                     X_fit_scaled, X_val_scaled, y_val, "contamination")
    best_svm = sweep("One-Class SVM", grid, train_one_class_svm, predict_one_class_svm,
                      X_fit_scaled, X_val_scaled, y_val, "nu")
    best_lof_contam = sweep("LOF (contamination, n_neighbors=20 fixed)", grid,
                             lambda X, contamination: train_lof(X, contamination=contamination, n_neighbors=20),
                             predict_lof, X_fit_scaled, X_val_scaled, y_val, "contamination")
    best_lof_k = sweep("LOF (n_neighbors, contamination=0.1 fixed)", [5, 10, 20, 30, 50, 75, 100],
                        lambda X, n_neighbors: train_lof(X, contamination=0.1, n_neighbors=n_neighbors),
                        predict_lof, X_fit_scaled, X_val_scaled, y_val, "n_neighbors")
    best_z = sweep("Z-score", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], train_zscore, predict_zscore,
                    X_fit_scaled, X_val_scaled, y_val, "threshold")

    print("\n=== chosen hyperparameters ===")
    print(f"IsolationForest contamination = {best_if}")
    print(f"One-Class SVM nu              = {best_svm}")
    print(f"LOF contamination             = {best_lof_contam}")
    print(f"LOF n_neighbors                = {best_lof_k}")
    print(f"Z-score threshold              = {best_z}")


if __name__ == "__main__":
    main()
