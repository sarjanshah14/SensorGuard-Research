"""HAI 21.03 as a third dataset (revision E9), same protocol as SWaT.

Run: python -m experiments.revision.hai_revised   -> results/revised/hai.json
Data: github.com/icsdataset/hai, folder hai-21.03 (expected in external_data/hai/hai-21.03).

HAI = HIL-augmented ICS testbed: a GE steam turbine (P1), Emerson boiler (P2),
FESTO water treatment (P3) coupled through a hardware-in-the-loop simulator (P4),
1 Hz. Protocol:
  - train = train1-3 (all normal) minus the last 15% of train3, which is held-out
    normal data for the label-free threshold;
  - test1 = labelled calibration for the validation-MCC threshold;
  - metrics on test1-5 ("full") and on test2-5 ("tail", unseen by both rules);
  - continuous points = columns with > 10 distinct values in the training files;
  - features computed per file; OCSVM/LOF on a 20,000-row and ECOD/COPOD on a
    100,000-row evenly spaced subsample (memory).
"""

import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src import deep_baseline as dbl
from src.features import build_features
from experiments.revision.common import (
    block_bootstrap, event_metrics, make_models, paired_diff, point_metrics,
    threshold_free, thr_label_free, thr_val_mcc,
)

DIR = "external_data/hai/hai-21.03"
OUT = "results/revised/hai.json"
CAP = 20000
CAP_ECDF = 100000  # ECOD/COPOD on all 850k x 275 training rows exhausted 16 GB RAM


def load():
    def rd(f):
        d = pd.read_csv(f, compression="gzip")
        d["time"] = pd.to_datetime(d["time"])
        return d.set_index("time")
    tr = [rd(f"{DIR}/train{i}.csv.gz") for i in (1, 2, 3)]
    te = [rd(f"{DIR}/test{i}.csv.gz") for i in (1, 2, 3, 4, 5)]
    labels = [c for c in tr[0].columns if c.startswith("attack")]
    allt = pd.concat(tr)
    sensors = [c for c in allt.columns if c not in labels and allt[c].nunique() > 10]
    return tr, te, sensors


def feats(dfs, sensors):
    return [build_features(d, sensor_cols=sensors, include_elapsed=False)[0] for d in dfs]


def main():
    os.makedirs("results/revised", exist_ok=True)
    t0 = time.perf_counter()
    tr, te, sensors = load()
    Ftr, Fte = feats(tr, sensors), feats(te, sensors)
    cut = int(len(Ftr[2]) * 0.85)
    Xtr_raw = np.vstack([Ftr[0], Ftr[1], Ftr[2][:cut]])
    Xvn_raw = Ftr[2][cut:]
    Xte_raw = np.vstack(Fte)
    y = np.concatenate([d["attack"].to_numpy() for d in te]).astype(int)
    file_id = np.concatenate([np.full(len(d), i) for i, d in enumerate(te)])
    cal, tail = file_id == 0, file_id > 0
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xvn, Xte = sc.transform(Xtr_raw), sc.transform(Xvn_raw), sc.transform(Xte_raw)
    del Xtr_raw, Xvn_raw, Xte_raw, Ftr, Fte
    print(f"train {len(Xtr)}, val-normal {len(Xvn)}, test {len(Xte)} ({y.mean():.4f} attacks), {Xtr.shape[1]} features", flush=True)

    scores, fit_s = {}, {}
    for name, m in make_models(seed=0, cap=CAP, cap_ecdf=CAP_ECDF).items():
        t = time.perf_counter(); m.fit(Xtr); fit_s[name] = round(time.perf_counter() - t, 1)
        # ponytail: chunked scoring bounds memory; PyOD's ECOD/COPOD score each chunk jointly with
        # their training sample, so their scores depend slightly on chunk size (50k rows here)
        sc_ = lambda X: np.concatenate([m.score(X[i:i + 50000]) for i in range(0, len(X), 50000)])
        scores[name] = (sc_(Xvn), sc_(Xte))
        print(f"  {name} {fit_s[name]}s", flush=True)
    if_auc = [threshold_free(y, make_models(seed=s)["IsolationForest"].fit(Xtr).score(Xte))["auroc"] for s in range(10)]

    # LSTM-AE on raw scaled points, windows within each file
    dbl.SEED = 0; torch.manual_seed(0); np.random.seed(0)
    rs = StandardScaler().fit(np.vstack([d[sensors].to_numpy() for d in tr]))
    R = lambda d: rs.transform(d[sensors].to_numpy()).astype(np.float32)
    w = np.concatenate([dbl.strided_windows(R(d)[: (cut if i == 2 else None)], dbl.WINDOW, dbl.TRAIN_STRIDE * 2)
                        for i, d in enumerate(tr)])
    t = time.perf_counter()
    model, losses = dbl.train_autoencoder(w, len(sensors), verbose=False)
    fit_s["LSTM-AE"] = round(time.perf_counter() - t, 1)
    s_vn = dbl.row_scores(model, R(tr[2])[cut:], np.zeros(len(tr[2]) - cut), dbl.WINDOW)
    s_te = np.concatenate([dbl.row_scores(model, R(d), np.zeros(len(d)), dbl.WINDOW) for d in te])
    scores["LSTM-AE"] = (s_vn, s_te)
    print(f"  LSTM-AE {fit_s['LSTM-AE']}s ({len(w)} windows)", flush=True)

    hours = file_id * 10000 + np.concatenate([np.arange(len(d)) // 3600 for d in te])
    res, lf = {}, {}
    for name, (sv, st) in scores.items():
        t_lf, t_mcc = thr_label_free(sv), thr_val_mcc(st[cal], y[cal])
        lf[name] = t_lf
        res[name] = {
            "fit_s": fit_s[name],
            "full": {**threshold_free(y, st), "label_free": {**point_metrics(y, st, t_lf), **event_metrics(y, st, t_lf, file_id)}},
            "tail": {**threshold_free(y[tail], st[tail]),
                     "label_free": {**point_metrics(y[tail], st[tail], t_lf), **event_metrics(y[tail], st[tail], t_lf, file_id[tail])},
                     "val_mcc": {**point_metrics(y[tail], st[tail], t_mcc), **event_metrics(y[tail], st[tail], t_mcc, file_id[tail])}},
        }
    ci, raw = block_bootstrap(y, {k: v[1] for k, v in scores.items()}, hours, lf, B=200)
    for k in res:
        res[k]["full"]["ci95"] = ci[k]
    ranked = sorted(res, key=lambda k: -res[k]["full"]["auroc"])
    out = {"rows": {"train": len(Xtr), "val_normal": len(Xvn), "test": len(Xte), "attack_rate": float(y.mean()),
                    "n_sensors": len(sensors), "n_features": int(Xtr.shape[1]), "ocsvm_lof_cap": CAP, "ecod_copod_cap": CAP_ECDF,
                    "lstm_windows": int(len(w))},
           "models": res, "isolation_forest_seeds_auroc": {"mean": float(np.mean(if_auc)), "sd": float(np.std(if_auc))},
           "paired_top2_auroc": {"a": ranked[0], "b": ranked[1], **paired_diff(raw, ranked[0], ranked[1])},
           "wall_s": round(time.perf_counter() - t0)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    for k in ranked:
        v = res[k]
        print(f"{k:16s} AUROC {v['full']['auroc']:.3f} {np.round(v['full']['ci95']['auroc'], 3)} AUPRC {v['full']['auprc']:.3f} | "
              f"LF MCC {v['full']['label_free']['mcc']:+.3f} FPR {v['full']['label_free']['fpr']:.3f} evR {v['full']['label_free']['event_recall']:.2f} | "
              f"tail VM MCC {v['tail']['val_mcc']['mcc']:+.3f}")
    print("IF seeds", out["isolation_forest_seeds_auroc"], "paired", out["paired_top2_auroc"], "wall", out["wall_s"])


if __name__ == "__main__":
    main()
