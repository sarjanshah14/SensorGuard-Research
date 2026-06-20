"""Full pipeline: load SKAB -> engineer features -> train/eval all 4 models.

Train set = anomaly-free.csv (assumed 100% normal, per its filename) + only
the *normal-labeled* rows from the chronological 80% slice of each fault
CSV. We deliberately drop the labeled-anomalous rows from training (see
experiments/training_strategy_comparison.py) — these are unsupervised
novelty detectors that assume "mostly normal" training data, and training
on anomaly-free.csv alone over-fits to that one run's specific operating
baseline, causing the models to flag almost everything in the fault
experiments as anomalous (precision ~0.30, recall ~1.0 — useless).

Test set = the remaining chronological 20% of each fault CSV (all rows,
anomalous and normal) — anomaly-free.csv is never evaluated against, since
it carries no real ground-truth labels (we only assign anomaly=0 to it
ourselves, for feature-building purposes, not as a dataset-provided label
to score against).
"""

import pandas as pd

from src.preprocessing import chronological_split, load_anomaly_free, load_skab_all
from src.features import build_features
from src.models import scale_features
from src.evaluate import compare_models, run_all_models
from src.persist import save_artifacts
from src.visualize import (
    plot_anomaly_closeup,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_time_series_with_anomalies,
)

SHOWCASE_MODEL = "IsolationForest"  # the paper's proposed model — used for the per-file figures


def select_showcase_file(test_df: pd.DataFrame) -> str:
    """Pick one test run for the illustrative per-file figures: prefer a run
    whose test slice contains a marked changepoint (so the close-up has a
    real transition to draw), and among those the most balanced normal/
    anomaly mix (anomaly rate nearest 0.5) so the plot shows both.
    """
    stats = []
    for fname, g in test_df.groupby("source_file", sort=False):
        stats.append((fname, g["changepoint"].sum() > 0, g["anomaly"].mean()))
    with_cp = [s for s in stats if s[1]]
    pool = with_cp or stats
    return min(pool, key=lambda s: abs(s[2] - 0.5))[0]


def main():
    anomaly_free = load_anomaly_free()
    anomaly_free["experiment_type"] = "anomaly-free"
    anomaly_free["source_file"] = "anomaly-free/anomaly-free"
    anomaly_free["anomaly"] = 0
    anomaly_free["changepoint"] = 0

    fault_df = load_skab_all()
    fault_train, fault_test = chronological_split(fault_df)
    fault_train_normal = fault_train[fault_train["anomaly"] == 0]

    train_df = pd.concat([anomaly_free, fault_train_normal])
    test_df = fault_test

    print(f"train rows: {len(train_df)} (anomaly rate {train_df['anomaly'].mean():.3f})")
    print(f"test rows:  {len(test_df)} (anomaly rate {test_df['anomaly'].mean():.3f})")

    X_train, feature_names = build_features(train_df, group_col="source_file")
    X_test, _ = build_features(test_df, group_col="source_file")
    y_test = test_df["anomaly"].to_numpy()

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    results, predictions, states = run_all_models(
        X_train_scaled, X_test_scaled, y_test,
        return_predictions=True, return_models=True,
    )
    comparison = compare_models(results)

    print(f"\n{len(feature_names)} features: {feature_names}\n")
    print(comparison.to_string(index=False))

    # --- persist fitted models for reproducibility (ROADMAP C3) ---
    artifacts_path = save_artifacts(states, scaler, feature_names)
    print(f"\nsaved models bundle: {artifacts_path}")

    # --- figures (ROADMAP C1) ---
    # build_features(test_df, group_col=...) preserves test_df's row order
    # (test_df already has contiguous per-source_file blocks from
    # chronological_split, and groupby(sort=False) keeps that order), so the
    # prediction arrays are row-aligned with test_df and can be attached as
    # a column. The assert guards that invariant against future refactors.
    assert len(predictions[SHOWCASE_MODEL]) == len(test_df)
    test_df = test_df.copy()
    test_df["pred"] = predictions[SHOWCASE_MODEL]

    showcase = select_showcase_file(test_df)
    showcase_df = test_df[test_df["source_file"] == showcase]

    paths = [
        plot_model_comparison(results),
        plot_confusion_matrices(results),
        plot_time_series_with_anomalies(showcase_df, "pred", SHOWCASE_MODEL),
        plot_anomaly_closeup(showcase_df, "pred", SHOWCASE_MODEL),
    ]
    print(f"\nshowcase file: {showcase}")
    print("figures written:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
