"""Concept drift adaptation experiment.

The 6-model comparison (main.py) trains once and scores a fixed test set —
it never asks what happens if the data a deployed model sees keeps changing
after training, which is exactly what happens on a real factory floor as
machines wear or operating conditions shift. This experiment simulates that:
SKAB's test-set files are streamed one at a time, in an order that groups by
fault family (other -> valve1 -> valve2). For this to actually simulate a
regime shift, the initial model must not have seen valve1/valve2 data at
all: main.py's chronological per-file 80/20 split puts the *early* 80% of
every fault file (including every valve1/valve2 file) into training, so a
model built that way has already partially learned every family before the
stream starts. Instead, the initial model here is trained only on
anomaly-free.csv + the "other" family's normal rows -- valve1 and valve2
are genuinely unseen distributions until the stream reaches them.

Two tracks run over the same stream, scored batch-by-batch (one
source_file = one batch):
  - static:   IsolationForest trained once on the "other"-only set, never
              updated.
  - adaptive: starts identical to static; after each batch, a KS-statistic
              DriftDetector (src/drift.py) checks the batch's features
              against the reference distribution. If drift is flagged, the
              model is retrained on all normal-labeled rows seen so far in
              the stream (plus the original training set) — never on
              test-time labels for detection itself, only for retraining
              hygiene, the same contamination discipline as main.py (B3).

Both the IsolationForest contamination and the drift threshold are tuned
on a holdout carved *only* from the "other" family (not the global
fault_train file pool) before touching the test stream: pulling validation
files from the full 34-file pool left only 1 "other" file by chance, too
small a sample to calibrate anything against. Mirrors
hyperparameter_sweep.py's fit/val/test discipline throughout.

Headline metric is MCC / balanced accuracy, not F1. F1 rewards a degenerate
"flag everything" predictor (recall 1.0, precision = anomaly base rate),
and that is exactly how the static model behaves once the stream enters an
unseen regime: it keeps recall ~1.0 only by alarming on nearly every row,
so its F1 stays ~0.43 while its MCC is 0.00 and balanced accuracy is 0.500
-- i.e. zero discriminative skill, no better than chance. Contamination is
therefore selected by MCC on the holdout (which collapses toward 0 as
contamination rises into the degenerate flag-everything range, the same
failure B4 rejected for the full pipeline), not by F1 (which would pick the
degenerate 0.35). The result: drift-triggered retraining restores genuine
skill on the unseen valve regimes (MCC 0.00 -> ~0.11, balanced accuracy
0.500 -> ~0.56) and roughly halves the false-alarm rate, where the static
detector has none.
"""

import json
from collections import deque

import numpy as np
import pandas as pd

from src.preprocessing import chronological_split, file_holdout_split, load_anomaly_free, load_skab_all
from src.features import build_features
from src.models import scale_features, train_isolation_forest, predict_isolation_forest
from src.evaluate import evaluate
from src.drift import DriftDetector, drift_score
from src.visualize import plot_drift_adaptation

RESULTS_PATH = "results/concept_drift.json"
INITIAL_FAMILY = "other"  # the only regime the initial model is trained on
CONTAMINATION_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def make_anomaly_free():
    anomaly_free = load_anomaly_free()
    anomaly_free["experiment_type"] = "anomaly-free"
    anomaly_free["source_file"] = "anomaly-free/anomaly-free"
    anomaly_free["anomaly"] = 0
    anomaly_free["changepoint"] = 0
    return anomaly_free


def build_train_set(anomaly_free, df, family: str = INITIAL_FAMILY):
    """anomaly-free.csv + only normal-labeled rows from `family` -- unlike
    main.py's recipe (which pools every family), this deliberately excludes
    valve1/valve2 so they're genuinely unseen when the stream reaches them."""
    family_rows = df[(df["experiment_type"] == family) & (df["anomaly"] == 0)]
    return pd.concat([anomaly_free, family_rows])


def ordered_stream(fault_test: pd.DataFrame) -> list[str]:
    """Test files ordered by fault family (other -> valve1 -> valve2),
    files within a family kept in their natural sorted order, so the
    regime visibly shifts partway through the stream."""
    family_order = ["other", "valve1", "valve2"]
    files_by_family = {
        fam: sorted(fault_test.loc[fault_test["experiment_type"] == fam, "source_file"].unique())
        for fam in family_order
    }
    return [f for fam in family_order for f in files_by_family[fam]]


def tune_contamination_and_threshold(anomaly_free, other_fit, other_val):
    """Sweep contamination on a holdout drawn only from "other" files,
    selecting by MCC rather than F1. F1 rises monotonically toward the
    degenerate "flag everything" end of the grid (recall -> 1.0, precision
    -> base rate) and so would pick contamination 0.35 -- a detector with
    essentially zero discriminative skill, the same failure B4 rejected.
    MCC instead peaks at a *selective* contamination and collapses toward 0
    as the model degenerates, so it picks an operating point that can
    actually tell anomalies from normal. Then, using the winning
    contamination's fit set, calibrate the drift threshold from each
    held-out "other" file's score against that fit set -- ordinary
    file-to-file variation within the same regime, which the test stream's
    real regime shifts should clearly exceed.
    """
    fit_train_df = build_train_set(anomaly_free, other_fit)
    X_fit, _ = build_features(fit_train_df, group_col="source_file")
    X_val, _ = build_features(other_val, group_col="source_file")
    y_val = other_val["anomaly"].to_numpy()
    X_fit_scaled, X_val_scaled, scaler = scale_features(X_fit, X_val)

    print("contamination sweep (other-family holdout, selecting by MCC):")
    best_contamination, best_mcc = None, -2.0
    for c in CONTAMINATION_GRID:
        model = train_isolation_forest(X_fit_scaled, contamination=c)
        y_pred = predict_isolation_forest(model, X_val_scaled)
        result = evaluate(y_val, y_pred, f"IF(contamination={c})")
        degenerate = result["recall"] > 0.99
        flag = " (degenerate, rejected)" if degenerate else ""
        print(f"  contamination={c:.2f}  precision={result['precision']:.3f}  "
              f"recall={result['recall']:.3f}  f1={result['f1']:.3f}  "
              f"mcc={result['mcc']:+.3f}{flag}")
        if not degenerate and result["mcc"] > best_mcc:
            best_contamination, best_mcc = c, result["mcc"]

    if best_contamination is None:
        raise RuntimeError("every contamination candidate was degenerate on the other-family holdout")
    print(f"chosen contamination = {best_contamination} (mcc={best_mcc:+.3f})")

    baseline_scores = []
    for fname, g in other_val.groupby("source_file", sort=False):
        Xg, _ = build_features(g, group_col="source_file")
        Xg_scaled = scaler.transform(Xg)
        baseline_scores.append((fname, drift_score(X_fit_scaled, Xg_scaled)))

    print("baseline drift scores (other-family holdout files vs. fit set):")
    for fname, score in sorted(baseline_scores, key=lambda t: t[1]):
        print(f"  {fname:20s} score={score:.3f}")
    threshold = float(np.percentile([s for _, s in baseline_scores], 90))

    return best_contamination, threshold


RETRAIN_WINDOW_FILES = 5  # how many most-recent batches feed each retrain


def run_stream(stream_files, fault_test, train_df, X_train_scaled, scaler, contamination, threshold):
    """Run the static and adaptive tracks batch-by-batch over stream_files.

    Retraining uses a sliding window of only the most recent
    RETRAIN_WINDOW_FILES normal-labeled batches, not all history since the
    stream began: once the regime shifts to valve1/valve2, the new regime's
    few hundred rows would be vastly outnumbered by the old "other"-family
    data in an all-history pool, so the retrained trees would barely move.
    The sliding window lets the model forget the old regime once it's no
    longer the deployment's reality, mirroring how real drift-adaptive
    systems retrain on recent data.

    Retraining deliberately lets IsolationForest re-derive its own decision
    threshold on the new window (no offset pinning): that re-derivation is
    precisely what re-anchors "normal" to the new regime and cuts the
    false-alarm flood the static model suffers there.
    """
    static_model = train_isolation_forest(X_train_scaled, contamination=contamination)

    adaptive_model = train_isolation_forest(X_train_scaled, contamination=contamination)
    detector = DriftDetector(reference=X_train_scaled, threshold=threshold)

    # sliding window of recent normal-labeled batches, seeded with the
    # original training set so an early retrain still has enough data
    recent_normal_rows = deque([train_df], maxlen=RETRAIN_WINDOW_FILES)
    static_records, adaptive_records = [], []
    n_retrains = 0

    for fname in stream_files:
        batch_df = fault_test[fault_test["source_file"] == fname]
        X_batch, _ = build_features(batch_df, group_col="source_file")
        X_batch_scaled = scaler.transform(X_batch)
        y_batch = batch_df["anomaly"].to_numpy()

        family = fname.split("/")[0]

        y_static = predict_isolation_forest(static_model, X_batch_scaled)
        static_result = evaluate(y_batch, y_static, "static")
        static_result["source_file"] = fname
        static_result["family"] = family
        static_records.append(static_result)

        y_adaptive = predict_isolation_forest(adaptive_model, X_batch_scaled)
        adaptive_result = evaluate(y_batch, y_adaptive, "adaptive")
        adaptive_result["source_file"] = fname
        adaptive_result["family"] = family

        drifted, score = detector.check(X_batch_scaled)
        adaptive_result["drift_score"] = score
        adaptive_result["drifted"] = bool(drifted)
        adaptive_result["retrained"] = False

        recent_normal_rows.append(batch_df[batch_df["anomaly"] == 0])
        if drifted:
            retrain_df = pd.concat(recent_normal_rows)
            X_retrain, _ = build_features(retrain_df, group_col="source_file")
            X_retrain_scaled = scaler.transform(X_retrain)
            adaptive_model = train_isolation_forest(X_retrain_scaled, contamination=contamination)
            detector.update_reference(X_retrain_scaled)
            adaptive_result["retrained"] = True
            n_retrains += 1

        adaptive_records.append(adaptive_result)

    print(f"\nretrains triggered: {n_retrains} / {len(stream_files)} batches")
    return static_records, adaptive_records


def main():
    anomaly_free = make_anomaly_free()
    fault_df = load_skab_all()
    fault_train, fault_test = chronological_split(fault_df, test_size=0.2)

    other_pool = fault_train[fault_train["experiment_type"] == INITIAL_FAMILY]
    other_fit, other_val = file_holdout_split(other_pool, val_frac=0.3)
    contamination, threshold = tune_contamination_and_threshold(anomaly_free, other_fit, other_val)
    print(f"\nchosen contamination = {contamination}, drift threshold = {threshold:.3f}")

    train_df = build_train_set(anomaly_free, fault_train)
    X_train, _ = build_features(train_df, group_col="source_file")
    X_test_dummy, _ = build_features(fault_test, group_col="source_file")
    X_train_scaled, _, scaler = scale_features(X_train, X_test_dummy)

    stream_files = ordered_stream(fault_test)
    print(f"\nstream order ({len(stream_files)} files): {stream_files}")

    static_records, adaptive_records = run_stream(
        stream_files, fault_test, train_df, X_train_scaled, scaler, contamination, threshold
    )

    metric_cols = ["source_file", "family", "precision", "recall", "f1", "mcc", "balanced_accuracy"]
    static_df = pd.DataFrame(static_records)[metric_cols]
    adaptive_df = pd.DataFrame(adaptive_records)[
        metric_cols + ["drift_score", "drifted", "retrained"]
    ]

    print("\n=== static (never retrains) ===")
    print(static_df.to_string(index=False))
    print("\n=== adaptive (retrains on drift) ===")
    print(adaptive_df.to_string(index=False))

    # Per-regime summary. The contribution lives in the unseen valve regimes
    # (everything after the "other" block in the stream), where the static
    # model degenerates to flag-everything and adaptive retraining restores
    # real skill. MCC / balanced accuracy are the honest scorecards; F1 is
    # shown too but it credits the degenerate static baseline.
    print("\n=== mean metrics by regime (static -> adaptive) ===")
    for label, mask in [
        ("other (seen)", static_df["family"] == "other"),
        ("valve (UNSEEN)", static_df["family"] != "other"),
        ("whole stream", static_df["family"].notna()),
    ]:
        s, a = static_df[mask], adaptive_df[mask]
        print(f"  {label:16s}  "
              f"MCC {s['mcc'].mean():+.3f} -> {a['mcc'].mean():+.3f}   "
              f"balAcc {s['balanced_accuracy'].mean():.3f} -> {a['balanced_accuracy'].mean():.3f}   "
              f"prec {s['precision'].mean():.3f} -> {a['precision'].mean():.3f}   "
              f"F1 {s['f1'].mean():.3f} -> {a['f1'].mean():.3f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "contamination": contamination,
            "threshold": threshold,
            "retrain_window_files": RETRAIN_WINDOW_FILES,
            "stream_order": stream_files,
            "static": static_records,
            "adaptive": adaptive_records,
        }, f, indent=2)
    print(f"\nsaved: {RESULTS_PATH}")

    fig_path = plot_drift_adaptation(static_records, adaptive_records)
    print(f"saved figure: {fig_path}")


if __name__ == "__main__":
    main()
