"""Does temporal smoothing of predictions improve the scores? (ROADMAP B5.)

Our best model has decent recall but low precision — too many false alarms.
The hypothesis (see src/postprocess.py): many false alarms are isolated
1-2 second blips, while real faults are sustained, so requiring a predicted
anomaly to persist for >= min_run seconds should cut false alarms.

min_run is a hyperparameter, so it is tuned on a validation set (whole held
-out fault files, never the test set) exactly like contamination was, then
applied once to the test set. The validation tuning and final test eval use
the same train sets as main.py so the "before" column here matches the
committed metrics.
"""

import pandas as pd

from src.preprocessing import chronological_split, file_holdout_split, load_anomaly_free, load_skab_all
from src.features import build_features
from src.models import scale_features, MODELS
from src.evaluate import evaluate
from src.postprocess import smooth_min_run

MIN_RUN_GRID = [1, 2, 3, 5, 8, 10, 15, 20, 30]


def make_anomaly_free():
    af = load_anomaly_free()
    af["experiment_type"] = "anomaly-free"
    af["source_file"] = "anomaly-free/anomaly-free"
    af["anomaly"] = 0
    af["changepoint"] = 0
    return af


def train_predict(train_df, eval_df):
    """Train all 4 models on train_df, return {model: raw_pred} on eval_df
    plus the eval source_file groups and y_true (all row-aligned)."""
    X_train, _ = build_features(train_df, group_col="source_file")
    X_eval, _ = build_features(eval_df, group_col="source_file")
    X_train_s, X_eval_s, _ = scale_features(X_train, X_eval)

    preds = {}
    for name, (train_fn, predict_fn) in MODELS.items():
        state = train_fn(X_train_s)
        preds[name] = predict_fn(state, X_eval_s)
    groups = eval_df["source_file"].to_numpy()
    y_true = eval_df["anomaly"].to_numpy()
    assert all(len(p) == len(eval_df) for p in preds.values())
    return preds, groups, y_true


def main():
    af = make_anomaly_free()
    fault_df = load_skab_all()
    fault_train, fault_test = chronological_split(fault_df)
    fault_fit, fault_val = file_holdout_split(fault_train, val_frac=0.2)

    # 1) Tune min_run per model on the validation files.
    fit_df = pd.concat([af, fault_fit[fault_fit["anomaly"] == 0]])
    val_preds, val_groups, y_val = train_predict(fit_df, fault_val)

    best_k = {}
    print("=== validation sweep (F1 by min_run) ===")
    for name, raw in val_preds.items():
        scores = {}
        for k in MIN_RUN_GRID:
            sm = smooth_min_run(raw, val_groups, k)
            scores[k] = evaluate(y_val, sm, name)["f1"]
        best_k[name] = max(scores, key=scores.get)
        row = " ".join(f"k{k}:{scores[k]:.3f}" for k in MIN_RUN_GRID)
        print(f"{name:16s} {row}  -> best k={best_k[name]}")

    # 2) Final eval on the untouched test set, with main.py's full train set.
    train_df = pd.concat([af, fault_train[fault_train["anomaly"] == 0]])
    test_preds, test_groups, y_test = train_predict(train_df, fault_test)

    print("\n=== test set: before vs after smoothing ===")
    rows = []
    for name, raw in test_preds.items():
        k = best_k[name]
        before = evaluate(y_test, raw, name)
        after = evaluate(y_test, smooth_min_run(raw, test_groups, k), name)
        rows.append({
            "model": name, "min_run": k,
            "P_before": before["precision"], "P_after": after["precision"],
            "R_before": before["recall"], "R_after": after["recall"],
            "F1_before": before["f1"], "F1_after": after["f1"],
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
