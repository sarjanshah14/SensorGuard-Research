"""Do the wall-clock features (hour, day_of_week) help or hurt? (ROADMAP B5.)

Concern: across the pooled SKAB runs, hour/day_of_week are spurious. Each
experiment was recorded at some wall-clock time, and faults are concentrated
within experiments, so a model can learn "anomalies happen around hour H /
weekday D" — which reflects the lab's scheduling, not the physics of a fault.
That kind of feature can flatter the test score while being meaningless (and
actively misleading) on any new deployment.

This compares the full feature set (with clock) vs. the clock-free set, same
train/test split as main.py, all 4 models. We expect the clock-free model to
be no worse — and to be the honest choice regardless.
"""

import pandas as pd

from src.preprocessing import chronological_split, load_anomaly_free, load_skab_all
from src.features import build_features
from src.models import scale_features, MODELS
from src.evaluate import evaluate


def make_anomaly_free():
    af = load_anomaly_free()
    af["experiment_type"] = "anomaly-free"
    af["source_file"] = "anomaly-free/anomaly-free"
    af["anomaly"] = 0
    af["changepoint"] = 0
    return af


def eval_with(train_df, test_df, y_test, include_clock):
    X_train, names = build_features(train_df, group_col="source_file", include_clock=include_clock)
    X_test, _ = build_features(test_df, group_col="source_file", include_clock=include_clock)
    X_train_s, X_test_s, _ = scale_features(X_train, X_test)
    out = {}
    for name, (train_fn, predict_fn) in MODELS.items():
        state = train_fn(X_train_s)
        out[name] = evaluate(y_test, predict_fn(state, X_test_s), name)["f1"]
    return out, len(names)


def main():
    af = make_anomaly_free()
    fault_df = load_skab_all()
    fault_train, fault_test = chronological_split(fault_df)
    train_df = pd.concat([af, fault_train[fault_train["anomaly"] == 0]])
    y_test = fault_test["anomaly"].to_numpy()

    with_clock, n_with = eval_with(train_df, fault_test, y_test, include_clock=True)
    no_clock, n_no = eval_with(train_df, fault_test, y_test, include_clock=False)

    print(f"features: with clock = {n_with}, without = {n_no}\n")
    rows = [{"model": m, "F1_with_clock": with_clock[m], "F1_no_clock": no_clock[m],
             "delta": no_clock[m] - with_clock[m]} for m in MODELS]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
