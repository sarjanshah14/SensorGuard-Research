"""Edge / resource-constrained deployment benchmark.

Run INSIDE a Docker container capped to Raspberry-Pi-class limits
(1 CPU core, 1 GB RAM, ARM64 -- the same CPU family as a real Pi) to test
the paper's deployability claim: that these lightweight classical detectors,
and the concept-drift adaptation built on them, are small and fast enough to
run on edge hardware, not just a laptop.

Measured on the real data, per model:
  - train time (s);
  - inference latency (ms per sample, amortised over the test batch) -- the
    real-time detection cost;
  - on-disk model size (KB);
  - F1 / MCC (a sanity check that constraining the box didn't change the maths).
Plus the concept-drift adaptation costs that run on-device:
  - drift-check latency (one KS test over a batch, in ms);
  - one retrain (s) -- what an on-device adaptation event costs.
And whole-run peak RAM (MB), which must stay under the 1 GB cap or Docker
kills the process -- so a clean finish is itself the proof it fits.

Honest scope: capping cores + RAM + using ARM matches a Pi's *constraints*
and instruction set; the M4 core is faster than a Pi core, so the latency
figures are an optimistic lower bound, reported as "constrained ARM
container", not "measured on a physical Raspberry Pi".
"""

import io
import json
import resource
import time

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import chronological_split, load_anomaly_free, load_skab_all
from src.features import build_features
from src.models import (
    scale_features, MODELS,
    train_isolation_forest, predict_isolation_forest,
    train_elliptic_envelope, predict_elliptic_envelope,
)
from src.evaluate import evaluate
from src.drift import drift_score

RESULTS_PATH = "results/edge_benchmark.json"


def peak_ram_mb():
    # ru_maxrss is KB on Linux (the container OS)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cpu_limit():
    """Effective CPU limit from the cgroup (respects docker --cpus), since
    os.cpu_count() reports the host's cores, not the container's cap."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:  # cgroup v2
            quota, period = f.read().split()
            if quota != "max":
                return round(float(quota) / float(period), 2)
    except (OSError, ValueError):
        pass
    import os
    return os.cpu_count()


def model_size_kb(state):
    buf = io.BytesIO()
    joblib.dump(state, buf)
    return len(buf.getvalue()) / 1024.0


def bench_skab():
    anomaly_free = load_anomaly_free()
    anomaly_free["experiment_type"] = "anomaly-free"
    anomaly_free["source_file"] = "anomaly-free/anomaly-free"
    anomaly_free["anomaly"] = 0
    anomaly_free["changepoint"] = 0

    fault_df = load_skab_all()
    fault_train, fault_test = chronological_split(fault_df)
    train_df = pd.concat([anomaly_free, fault_train[fault_train["anomaly"] == 0]])

    X_train, names = build_features(train_df, group_col="source_file")
    X_test, _ = build_features(fault_test, group_col="source_file")
    y_test = fault_test["anomaly"].to_numpy()
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    rows = []
    for name, (train_fn, predict_fn) in MODELS.items():
        t = time.perf_counter()
        state = train_fn(X_train_s)
        train_s = time.perf_counter() - t
        t = time.perf_counter()
        y_pred = predict_fn(state, X_test_s)
        infer_total = time.perf_counter() - t
        r = evaluate(y_test, y_pred, name)
        rows.append({
            "model": name,
            "train_s": round(train_s, 3),
            "infer_ms_per_sample": round(infer_total / len(X_test_s) * 1000, 4),
            "size_KB": round(model_size_kb(state), 1),
            "f1": round(r["f1"], 3), "mcc": round(r["mcc"], 3),
        })
    return rows, X_train_s, X_test_s, names


def bench_drift(X_train_s, X_test_s):
    ref = X_train_s
    batch = X_test_s[:2000]
    t = time.perf_counter()
    for _ in range(5):
        drift_score(ref, batch)
    drift_ms = (time.perf_counter() - t) / 5 * 1000

    t = time.perf_counter()
    train_isolation_forest(X_train_s, contamination=0.05)
    retrain_s = time.perf_counter() - t
    return {"drift_check_ms_per_batch": round(drift_ms, 2),
            "retrain_s": round(retrain_s, 3)}


def bench_swat_infer():
    """Bounded SWaT slice (kept small to fit the 1 GB cap): inference latency
    for the two detectors that lead across datasets (IsolationForest,
    EllipticEnvelope)."""
    from src.swat import load_swat_bounded, CONTINUOUS_SENSORS
    normal, attack = load_swat_bounded(n_normal=30000, n_attack=20000)
    normal = normal.assign(segment="n")
    attack = attack.assign(segment="a")
    Xn, _ = build_features(normal, group_col="segment", sensor_cols=CONTINUOUS_SENSORS)
    Xa, _ = build_features(attack, group_col="segment", sensor_cols=CONTINUOUS_SENSORS)
    y = np.concatenate([np.zeros(len(Xn)), np.ones(len(Xa))])
    X = np.vstack([Xn, Xa])
    Xtr_s, X_s, _ = scale_features(Xn, X)  # train on normal only

    out = {}
    for name, tr, pr in [("IsolationForest", train_isolation_forest, predict_isolation_forest),
                         ("EllipticEnvelope", train_elliptic_envelope, predict_elliptic_envelope)]:
        state = tr(Xtr_s, contamination=0.05)
        t = time.perf_counter()
        yp = pr(state, X_s)
        infer_total = time.perf_counter() - t
        r = evaluate(y, yp, name)
        out[name] = {"infer_ms_per_sample": round(infer_total / len(X_s) * 1000, 4),
                     "size_KB": round(model_size_kb(state), 1), "mcc": round(r["mcc"], 3)}
    return out


def main():
    import os
    t0 = time.perf_counter()
    env = {
        "cpu_limit_cgroup": cpu_limit(),
        "note": "run under docker --cpus=1 --memory=1g on ARM64 (Pi-class limits)",
    }
    print("=== EDGE BENCHMARK (constrained ARM container) ===")
    print(f"CPU limit (cgroup): {env['cpu_limit_cgroup']}  |  RAM cap: 1024 MB  |  arch: ARM64")

    skab_rows, Xtr, Xte, names = bench_skab()
    print(f"\nSKAB: {len(names)} features, train={len(Xtr)}, test={len(Xte)}")
    print(f"{'model':17s} {'train_s':>8} {'infer_ms/samp':>13} {'size_KB':>9} {'F1':>6} {'MCC':>6}")
    for r in skab_rows:
        print(f"{r['model']:17s} {r['train_s']:>8.3f} {r['infer_ms_per_sample']:>13.4f} "
              f"{r['size_KB']:>9.1f} {r['f1']:>6.3f} {r['mcc']:>6.3f}")

    drift = bench_drift(Xtr, Xte)
    print(f"\ndrift adaptation on-device: KS check {drift['drift_check_ms_per_batch']} ms/batch, "
          f"retrain {drift['retrain_s']} s")

    swat = {}
    try:
        swat = bench_swat_infer()
        print("\nSWaT (bounded) inference latency:")
        for m, d in swat.items():
            print(f"  {m:17s} {d['infer_ms_per_sample']:.4f} ms/sample  "
                  f"size {d['size_KB']} KB  MCC {d['mcc']}")
    except Exception as e:
        print(f"\nSWaT bench skipped ({type(e).__name__}: {e})")

    peak = peak_ram_mb()
    total = time.perf_counter() - t0
    print(f"\npeak RAM: {peak:.1f} MB (cap 1024 MB)   total wall: {total:.1f}s")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"env": env, "skab": skab_rows, "drift": drift, "swat": swat,
                   "peak_ram_mb": round(peak, 1), "total_wall_s": round(total, 1)}, f, indent=2)
    print(f"saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
