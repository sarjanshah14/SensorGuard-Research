"""Full pipeline: load SKAB -> engineer features -> train/eval all 4 models.

Train set = anomaly-free.csv (assumed 100% normal, per its filename) + the
chronological 80% of each fault CSV. Test set = the remaining chronological
20% of each fault CSV only — anomaly-free.csv is never evaluated against,
since it carries no real ground-truth labels (we only assign anomaly=0 to
it ourselves, for feature-building purposes, not as a dataset-provided
label to score against).
"""

import pandas as pd

from src.preprocessing import chronological_split, load_anomaly_free, load_skab_all
from src.features import build_features
from src.models import scale_features
from src.evaluate import compare_models, run_all_models


def main():
    anomaly_free = load_anomaly_free()
    anomaly_free["experiment_type"] = "anomaly-free"
    anomaly_free["source_file"] = "anomaly-free/anomaly-free"
    anomaly_free["anomaly"] = 0
    anomaly_free["changepoint"] = 0

    fault_df = load_skab_all()
    fault_train, fault_test = chronological_split(fault_df)

    train_df = pd.concat([anomaly_free, fault_train])
    test_df = fault_test

    print(f"train rows: {len(train_df)} (anomaly rate {train_df['anomaly'].mean():.3f})")
    print(f"test rows:  {len(test_df)} (anomaly rate {test_df['anomaly'].mean():.3f})")

    X_train, feature_names = build_features(train_df, group_col="source_file")
    X_test, _ = build_features(test_df, group_col="source_file")
    y_test = test_df["anomaly"].to_numpy()

    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    results = run_all_models(X_train_scaled, X_test_scaled, y_test)
    comparison = compare_models(results)

    print(f"\n{len(feature_names)} features: {feature_names}\n")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
