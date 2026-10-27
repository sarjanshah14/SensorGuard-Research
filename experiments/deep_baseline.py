"""Run the LSTM-autoencoder deep baseline on SKAB and SWaT, on the SAME test
splits as the classical comparison, and print / save the results so the
classical-vs-deep gap can be read directly.

Run: python -m experiments.deep_baseline
Saves: results/deep_baseline.json
"""

import json

import numpy as np
import pandas as pd

from src.preprocessing import (
    chronological_split, file_holdout_split, load_anomaly_free, load_skab_all,
)
from src.features import SENSOR_COLS
from src.swat import load_swat, CONTINUOUS_SENSORS
from src.deep_baseline import run_deep_baseline


def skab_frames():
    anomaly_free = load_anomaly_free()
    anomaly_free["experiment_type"] = "anomaly-free"
    anomaly_free["source_file"] = "anomaly-free/anomaly-free"
    anomaly_free["anomaly"] = 0

    fault_df = load_skab_all()
    fault_train, fault_test = chronological_split(fault_df)
    fit_df, val_df = file_holdout_split(fault_train)   # val = whole held-out files (labelled, unseen)

    train_df = pd.concat([anomaly_free, fit_df[fit_df["anomaly"] == 0]])
    return SENSOR_COLS, train_df, val_df, fault_test, "source_file"


def swat_frames():
    normal, attack = load_swat()
    n = len(normal)
    tr, va = int(n * 0.70), int(n * 0.85)
    ntr = normal.iloc[:tr].assign(segment="normal_train")
    nva = normal.iloc[tr:va].assign(segment="normal_val")
    nte = normal.iloc[va:].assign(segment="normal_test")
    a = len(attack)
    ava = attack.iloc[:a // 2].assign(segment="attack_val")
    ate = attack.iloc[a // 2:].assign(segment="attack_test")

    train_df = ntr
    val_df = pd.concat([nva, ava])
    test_df = pd.concat([nte, ate])
    return CONTINUOUS_SENSORS, train_df, val_df, test_df, "segment"


def main():
    results = {}
    arrays = {}
    for name, frames_fn in [("SKAB", skab_frames), ("SWaT", swat_frames)]:
        sensor_cols, train_df, val_df, test_df, group_col = frames_fn()
        print(f"\n===== {name}: LSTM-AE deep baseline =====")
        m, scores, labels = run_deep_baseline(sensor_cols, train_df, val_df, test_df, group_col, name)
        results[name] = m
        arrays[f"{name}_scores"] = scores.astype(np.float32)
        arrays[f"{name}_labels"] = labels.astype(np.int8)
        arrays[f"{name}_losses"] = np.array(m["train_losses"], dtype=np.float32)
        arrays[f"{name}_thr"] = np.array([m["threshold"]], dtype=np.float32)
        print(f"[{name}] TEST  MCC={m['mcc']:.3f}  bAcc={m['balanced_accuracy']:.3f}  "
              f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
        print(f"[{name}] params={m['n_params']}  size={m['size_KB']}KB  "
              f"train={m['train_time_s']}s  infer={m['infer_ms_per_sample']}ms/sample")

    with open("results/deep_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez("results/deep_baseline_arrays.npz", **arrays)
    print("\nsaved: results/deep_baseline.json + results/deep_baseline_arrays.npz")

    # side-by-side reminder of the classical leaders (from the paper tables)
    print("\nfor reference, classical MCC on the same test splits:")
    print("  SKAB : IsolationForest 0.147 (best) | EllipticEnvelope 0.086")
    print("  SWaT : EllipticEnvelope 0.766 (best) | One-Class SVM 0.634 | IsolationForest 0.473")


if __name__ == "__main__":
    main()
