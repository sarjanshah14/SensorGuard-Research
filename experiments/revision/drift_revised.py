"""Concept-drift study redone on SWaT's genuine normal run (revision/AUDIT.md E4).

Run: python -m experiments.revision.drift_revised   -> results/revised/drift.json

Deployment simulation, strictly causal (each 1-hour batch is scored BEFORE it
can be used for retraining):
  - initial model: first 50,000 normal rows after start-up (~14 h);
  - stream: rest of the 7-day normal run, then the 5-day attack period;
  - policies: static | KS-triggered (mean KS D vs reference > 95th pct of
    within-initial-window variation) | periodic every P batches for
    P in PERIODS (P = 1 is "always"), giving a cost-performance curve against
    which the KS policy is compared;
  - a retrain *attempt* is a batch where the policy fires; it becomes a *refit*
    only if the selected window holds > 1,000 rows (otherwise the model is kept).
    Both counts are reported;
  - retraining data = last 5 batches, rows selected by
      oracle      : ground-truth normal rows (what the original paper did),
      pred_normal : rows the current model scored as normal (label-free),
      all         : every row (label-free, no filtering);
  - after every (re)fit the threshold is the 99th pct of the fit-window
    scores (label-free).
Scaler is fit once on the initial window and kept fixed.
"""

import json
import os
import time
from collections import deque

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.drift import drift_score
from src.features import build_features
from experiments.revision.common import event_metrics, make_models, point_metrics
from experiments.revision.swat_revised import SENSORS, SOURCE, STARTUP, load_standard

OUT = {"reupload": "results/revised/drift.json"}.get(SOURCE, "results/revised/drift_official.json")
N_INIT, BATCH, WINDOW = 50000, 3600, 5
MODELS = ["IsolationForest", "EllipticEnvelope"]
PERIODS = [1, 2, 3, 4, 6, 12, 24, 48]


def fit(name, X):
    m = make_models(seed=0)[name].fit(X)
    return m, float(np.quantile(m.score(X), 0.99))


def run(name, Xinit, batches, ybat, policy, select, period=None, ks_thr=None):
    m, thr = fit(name, Xinit)
    ref = Xinit
    recent = deque(maxlen=WINDOW)
    preds, retrains, attempts, retrain_s, ks_s = [], 0, 0, 0.0, 0.0
    for i, (Xb, yb) in enumerate(zip(batches, ybat)):
        p = m.score(Xb) > thr
        preds.append(p)
        keep = {"oracle": yb == 0, "pred_normal": ~p, "all": np.ones(len(p), bool)}[select]
        recent.append(Xb[keep])
        do = False
        if policy == "always":
            do = True
        elif policy == "periodic":
            do = (i + 1) % period == 0
        elif policy == "ks":
            t = time.perf_counter()
            do = drift_score(ref, Xb) > ks_thr
            ks_s += time.perf_counter() - t
        if do:
            attempts += 1
            Xr = np.vstack(list(recent))
            if len(Xr) > 1000:
                t = time.perf_counter()
                m, thr = fit(name, Xr)
                retrain_s += time.perf_counter() - t
                ref = Xr
                retrains += 1
    return np.concatenate(preds), (retrains, attempts), retrain_s, ks_s


def summarise(pred, y, seg, counts, retrain_s, ks_s):
    n, a = seg == 0, seg == 1
    return {"retrains": counts[0], "attempts": counts[1], "retrain_cpu_s": round(retrain_s, 2), "ks_check_cpu_s": round(ks_s, 2),
            "normal_week_fpr": float(pred[n].mean()),
            "attack_period": {**point_metrics(y[a], pred[a].astype(float), 0.5),
                              **event_metrics(y[a], pred[a].astype(float), 0.5)}}


def main():
    os.makedirs("results/revised", exist_ok=True)
    normal, ap = load_standard()
    Xn, _ = build_features(normal, sensor_cols=SENSORS, include_elapsed=False)
    Xa, _ = build_features(ap, sensor_cols=SENSORS, include_elapsed=False)
    init = slice(STARTUP, STARTUP + N_INIT)
    sc = StandardScaler().fit(Xn[init])
    Xinit = sc.transform(Xn[init])
    Xs = np.vstack([sc.transform(Xn[STARTUP + N_INIT:]), sc.transform(Xa)])
    y = np.r_[np.zeros(len(Xn) - STARTUP - N_INIT, int), ap["anomaly"].to_numpy()]
    seg = np.r_[np.zeros(len(Xn) - STARTUP - N_INIT, int), np.ones(len(Xa), int)]
    cuts = list(range(0, len(Xs), BATCH))
    batches = [Xs[c:c + BATCH] for c in cuts]
    ybat = [y[c:c + BATCH] for c in cuts]
    within = [drift_score(Xinit, Xinit[i:i + BATCH]) for i in range(0, N_INIT - BATCH, BATCH)]
    ks_thr = float(np.percentile(within, 95))
    print(f"{len(batches)} batches; KS threshold {ks_thr:.3f}", flush=True)

    # static model's false-alarm trajectory over the normal week (does genuine normal drift?)
    m, thr = fit("IsolationForest", Xinit)
    normal_batches = sum(1 for c in cuts if c < (seg == 0).sum())
    traj = [float((m.score(b) > thr).mean()) for b in batches[:normal_batches]]

    out = {"config": {"n_init": N_INIT, "batch": BATCH, "window_batches": WINDOW, "ks_threshold": ks_thr,
                      "n_batches": len(batches), "normal_week_batches": normal_batches},
           "static_if_normal_fpr_trajectory": traj, "runs": {}}
    for name in MODELS:
        p, r, rs, ks = run(name, Xinit, batches, ybat, "static", "all")
        out["runs"][f"{name}|static"] = summarise(p, y, seg, r, rs, ks)
        for sel in ["oracle", "pred_normal", "all"]:
            p, r, rs, ks = run(name, Xinit, batches, ybat, "ks", sel, ks_thr=ks_thr)
            out["runs"][f"{name}|ks|{sel}"] = summarise(p, y, seg, r, rs, ks)
            for P in PERIODS:
                p, r2, rs, ks = run(name, Xinit, batches, ybat, "periodic", sel, period=P)
                out["runs"][f"{name}|periodic{P}|{sel}"] = {**summarise(p, y, seg, r2, rs, ks), "period_batches": P}
            print(f"  {name} {sel} done", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("static IF normal-week FPR (first, mid, last):", traj[0], traj[len(traj) // 2], traj[-1])
    for k, v in out["runs"].items():
        ap_ = v["attack_period"]
        print(f"{k:34s} attempts {v['attempts']:3d} refits {v['retrains']:3d}  normalFPR {v['normal_week_fpr']:.3f} | attack MCC {ap_['mcc']:+.3f} "
              f"rec {ap_['recall']:.3f} FPR {ap_['fpr']:.3f} evRec {ap_['event_recall']:.2f}  cpu {v['retrain_cpu_s']}s")


if __name__ == "__main__":
    main()
