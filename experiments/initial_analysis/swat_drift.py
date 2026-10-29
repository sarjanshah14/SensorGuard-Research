"""Stage 2 of the second-dataset study: replicate the concept-drift result on
SWaT, using REAL temporal drift rather than SKAB's fault-family proxy.

SWaT's normal operation drifts strongly over its ~4.5-day run: a detector
trained on the first ~14 h of normal data flags 8% of that early normal as
anomalous (as designed) but, left static, climbs to flagging 60-100% of
*later* normal data as anomalous -- the sensor baselines simply move (a
direct, measured form of concept drift; see the FP-rate trajectory in
DEVLOG). This is exactly the failure a deployed detector suffers and could
not be shown on SKAB's single short runs.

Protocol:
  - train the initial IsolationForest on the first N_INIT normal rows;
  - stream the REMAINING normal data forward in time, in batches;
  - to score detection (not just false alarms) at each step, each batch is
    paired with a fixed, held-out sample of ATTACK rows -- identical across
    batches, so any change in skill over the stream is due to the drifting
    NORMAL baseline, which is what we are studying;
  - static = never updated; adaptive = when the KS-statistic DriftDetector
    fires on a batch's normal features, retrain on a sliding window of the
    most recent normal batches (so it tracks the moving baseline).

Headline metric is MCC / balanced accuracy (as in the SKAB drift study), with
the normal-only false-alarm rate reported alongside because it is the most
visceral symptom of the drift. contamination=0.05 is reused from the SWaT
validation tuning in swat_comparison.py (never tuned on this stream).
"""

import json
import time

import numpy as np
import pandas as pd
from collections import deque

from src.swat import CONTINUOUS_SENSORS, load_swat
from src.features import build_features
from src.models import scale_features, train_isolation_forest, predict_isolation_forest
from src.evaluate import evaluate
from src.drift import DriftDetector, drift_score
from src.visualize import plot_drift_adaptation

RESULTS_PATH = "results/initial_analysis/swat_drift.json"
CONTAMINATION = 0.05
N_INIT = 50000            # initial training window (~14 h of normal at 1 Hz)
BATCH = 10000            # streamed normal batch size (~2.8 h)
ATTACK_SAMPLE = 3000     # fixed held-out attacks paired with each batch
RETRAIN_WINDOW = 5       # sliding window of recent normal batches for retrain


def main():
    t0 = time.perf_counter()
    normal, attack = load_swat()
    normal = normal.assign(segment="n")
    attack = attack.assign(segment="a")

    Xn, names = build_features(normal, group_col="segment", sensor_cols=CONTINUOUS_SENSORS)
    Xa, _ = build_features(attack, group_col="segment", sensor_cols=CONTINUOUS_SENSORS)

    # scaler + initial model from the first N_INIT normal rows only
    X_init = Xn[:N_INIT]
    X_init_s, _, scaler = scale_features(X_init, X_init)
    Xa_s = scaler.transform(Xa)

    # fixed held-out attack sample (second half of attacks => disjoint from the
    # first-half validation used to tune contamination in swat_comparison.py)
    rng = np.random.RandomState(42)
    attack_pool = Xa_s[len(Xa_s) // 2:]
    idx = rng.choice(len(attack_pool), size=min(ATTACK_SAMPLE, len(attack_pool)), replace=False)
    attack_fixed = attack_pool[idx]

    # drift threshold = 95th percentile of within-initial-window KS variation
    init_scores = [drift_score(X_init_s, X_init_s[i:i + BATCH])
                   for i in range(0, N_INIT - BATCH, BATCH)]
    threshold = float(np.percentile(init_scores, 95)) if init_scores else 0.2
    print(f"within-init drift scores: {[round(s,3) for s in init_scores]}  -> threshold={threshold:.3f}")

    static_model = train_isolation_forest(X_init_s, contamination=CONTAMINATION)
    adaptive_model = train_isolation_forest(X_init_s, contamination=CONTAMINATION)
    detector = DriftDetector(reference=X_init_s, threshold=threshold)
    recent = deque([X_init_s], maxlen=RETRAIN_WINDOW)

    static_records, adaptive_records = [], []
    n_retrains = 0
    y_attack = np.ones(len(attack_fixed), dtype=int)

    starts = list(range(N_INIT, len(Xn) - BATCH, BATCH))
    for bi, s in enumerate(starts):
        normal_batch = scaler.transform(Xn[s:s + BATCH])
        # batch test set = drifting normal (label 0) + fixed attacks (label 1)
        X_batch = np.vstack([normal_batch, attack_fixed])
        y_batch = np.concatenate([np.zeros(len(normal_batch), dtype=int), y_attack])

        for model, recs in [(static_model, static_records), (adaptive_model, adaptive_records)]:
            yp = predict_isolation_forest(model, X_batch)
            r = evaluate(y_batch, yp, "x")
            r["source_file"] = f"t{bi:02d}"
            r["family"] = "normal"  # all batches are the same regime drifting in time
            r["normal_fp_rate"] = float(yp[:len(normal_batch)].mean())
            recs.append(r)

        drifted, score = detector.check(normal_batch)
        adaptive_records[-1]["drift_score"] = score
        adaptive_records[-1]["drifted"] = bool(drifted)
        adaptive_records[-1]["retrained"] = False
        recent.append(normal_batch)
        if drifted:
            X_re = np.vstack(list(recent))
            adaptive_model = train_isolation_forest(X_re, contamination=CONTAMINATION)
            detector.update_reference(X_re)
            adaptive_records[-1]["retrained"] = True
            n_retrains += 1

    sdf = pd.DataFrame(static_records)
    adf = pd.DataFrame(adaptive_records)
    print(f"\nstreamed {len(starts)} batches; retrains: {n_retrains}")
    print("\n=== mean over stream (static -> adaptive) ===")
    print(f"  MCC              {sdf['mcc'].mean():+.3f} -> {adf['mcc'].mean():+.3f}")
    print(f"  balanced acc     {sdf['balanced_accuracy'].mean():.3f} -> {adf['balanced_accuracy'].mean():.3f}")
    print(f"  precision        {sdf['precision'].mean():.3f} -> {adf['precision'].mean():.3f}")
    print(f"  normal FP rate   {sdf['normal_fp_rate'].mean():.3f} -> {adf['normal_fp_rate'].mean():.3f}")
    print(f"  recall(attacks)  {sdf['recall'].mean():.3f} -> {adf['recall'].mean():.3f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "contamination": CONTAMINATION, "threshold": threshold,
            "n_init": N_INIT, "batch": BATCH, "attack_sample": ATTACK_SAMPLE,
            "retrain_window": RETRAIN_WINDOW, "n_retrains": n_retrains,
            "static": static_records, "adaptive": adaptive_records,
        }, f, indent=2)
    fig = plot_drift_adaptation(static_records, adaptive_records,
                                out_path="results/initial_analysis/figures/swat_drift_adaptation.png")
    print(f"\nsaved: {RESULTS_PATH}\nsaved figure: {fig}")
    print(f"wall time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
