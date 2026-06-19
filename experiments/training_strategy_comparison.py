"""Ablation: does reducing how 'dirty' the training data is improve the 4
models? (ROADMAP.md checkpoint B3.)

"Dirty" here means: what fraction of the training rows are actually labeled
anomalous. Z-score/OCSVM/LOF/IsolationForest all assume training data is
*mostly* normal — they learn "normal" and flag deviations from it, so if a
big chunk of "normal" is itself anomalous, the learned baseline is wrong.

Three variants, same test set, same features, same models:
  - pooled            : current main.py behaviour (anomaly-free + ALL of the
                        80% fault-train slice, anomalies included) — baseline.
  - anomaly_free_only : train on anomaly-free.csv alone (0% anomalous).
  - filtered          : anomaly-free.csv + only the *normal-labeled* rows
                        from the fault-train slice (keeps more realistic
                        "normal under real operating conditions" data than
                        anomaly_free_only, without the contamination).
"""

import pandas as pd

from src.preprocessing import chronological_split, load_anomaly_free, load_skab_all
from src.features import build_features
from src.models import scale_features
from src.evaluate import run_all_models


def make_anomaly_free():
    anomaly_free = load_anomaly_free()
    anomaly_free["experiment_type"] = "anomaly-free"
    anomaly_free["source_file"] = "anomaly-free/anomaly-free"
    anomaly_free["anomaly"] = 0
    anomaly_free["changepoint"] = 0
    return anomaly_free


def build_variant(anomaly_free, fault_train, strategy):
    if strategy == "pooled":
        return pd.concat([anomaly_free, fault_train])
    if strategy == "anomaly_free_only":
        return anomaly_free
    if strategy == "filtered":
        normal_rows = fault_train[fault_train["anomaly"] == 0]
        return pd.concat([anomaly_free, normal_rows])
    raise ValueError(strategy)


def main():
    anomaly_free = make_anomaly_free()
    fault_df = load_skab_all()
    fault_train, fault_test = chronological_split(fault_df)

    X_test, _ = build_features(fault_test, group_col="source_file")
    y_test = fault_test["anomaly"].to_numpy()

    for strategy in ["pooled", "anomaly_free_only", "filtered"]:
        train_df = build_variant(anomaly_free, fault_train, strategy)
        X_train, _ = build_features(train_df, group_col="source_file")
        X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

        results = run_all_models(X_train_scaled, X_test_scaled, y_test)
        summary = pd.DataFrame(results)[["model", "precision", "recall", "f1"]]

        print(f"\n=== {strategy} ===")
        print(f"train rows: {len(train_df)}, anomaly rate: {train_df['anomaly'].mean():.3f}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
