"""SWaT under the standard protocol (revision/AUDIT.md E1) plus the original
re-upload layout scored with the same code (protocol-sensitivity, RQ2).

Run: python -m experiments.revision.swat_revised   -> results/revised/swat.json

Standard protocol (as in Li et al. 2019 MAD-GAN; Deng & Hooi 2021):
  - normal run = 22-28 Dec 2015 (the 7-day block of the Kaggle normal.csv,
    de-duplicated by timestamp: 496,800 rows); first 21,600 rows (start-up
    stabilisation) dropped;
  - attack period = the 395,298 normal-labelled rows of normal.csv dated
    28 Dec-2 Jan merged in time with the 54,621 rows of attack.csv
    (= 449,919 rows, 12.1% attacks);
  - 24 continuous sensors (AIT201 is blank in this copy's normal run);
  - features computed on each continuous recording;
  - train = normal run up to 85%; validation-normal = last 15% (label-free
    threshold); first third of the attack period = labelled calibration for
    the val-MCC threshold; metrics reported on the full attack period and on
    its last two thirds (where both threshold rules are comparable).
OCSVM and LOF train on a 20,000-row stride subsample (stated in results).
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src import deep_baseline as dbl
from src.features import build_features
from src.swat import CONTINUOUS_SENSORS, _read, _swat_dir
from experiments.revision.common import (
    block_bootstrap, event_metrics, make_models, paired_diff, point_metrics,
    threshold_free, thr_label_free, thr_val_mcc,
)

OUT = "results/revised/swat.json"
SENSORS = [c for c in CONTINUOUS_SENSORS if c != "AIT201"]
CAP = 20000
STARTUP = 21600


def load_standard():
    d = _swat_dir()
    raw = _read(os.path.join(d, "normal.csv"))
    att = _read(os.path.join(d, "attack.csv"))
    raw["anomaly"] = 0
    att["anomaly"] = 1
    cut = pd.Timestamp("2015-12-28 10:00:00")
    normal = raw[raw.index < cut]
    normal = normal[~normal.index.duplicated(keep="first")].sort_index()
    attack_period = pd.concat([raw[raw.index >= cut], att]).sort_index()
    assert not normal[SENSORS].isna().any().any()
    return normal, attack_period


def deep_scores(X_tr_rows, X_all_by_segment, seed=0):
    dbl.SEED = seed
    torch.manual_seed(seed); np.random.seed(seed)
    w = dbl.strided_windows(X_tr_rows, dbl.WINDOW, dbl.TRAIN_STRIDE)
    t = time.perf_counter()
    model, losses = dbl.train_autoencoder(w, X_tr_rows.shape[1], verbose=False)
    train_s = time.perf_counter() - t
    out = {k: dbl.row_scores(model, v, np.zeros(len(v)), dbl.WINDOW) for k, v in X_all_by_segment.items()}
    return out, {"train_s": round(train_s, 1), "final_loss": losses[-1], "windows": len(w)}


def run_standard():
    normal, ap = load_standard()
    n = len(normal)
    Xn, names = build_features(normal, sensor_cols=SENSORS, include_elapsed=False)
    Xa, _ = build_features(ap, sensor_cols=SENSORS, include_elapsed=False)
    tr = np.arange(STARTUP, int(n * 0.85))
    vn = np.arange(int(n * 0.85), n)
    ya = ap["anomaly"].to_numpy()
    third = len(ap) // 3
    cal, tail = np.arange(third), np.arange(third, len(ap))
    sc = StandardScaler().fit(Xn[tr])
    Xtr, Xvn, Xas = sc.transform(Xn[tr]), sc.transform(Xn[vn]), sc.transform(Xa)

    scores, timing = {}, {}
    for name, m in make_models(seed=0, cap=CAP).items():
        t = time.perf_counter(); m.fit(Xtr); fit_s = time.perf_counter() - t
        scores[name] = (m.score(Xvn), m.score(Xas))
        timing[name] = round(fit_s, 1)
        print(f"  {name} fitted in {fit_s:.1f}s", flush=True)
    if_auc = [threshold_free(ya, make_models(seed=s)["IsolationForest"].fit(Xtr).score(Xas))["auroc"] for s in range(10)]

    # deep: raw scaled sensors
    rs = StandardScaler().fit(normal[SENSORS].to_numpy()[tr])
    Rn, Ra = rs.transform(normal[SENSORS].to_numpy()).astype(np.float32), rs.transform(ap[SENSORS].to_numpy()).astype(np.float32)
    ds, dinfo = deep_scores(Rn[tr], {"normal": Rn, "attack": Ra})
    scores["LSTM-AE"] = (ds["normal"][vn], ds["attack"])
    timing["LSTM-AE"] = dinfo["train_s"]

    hours = np.arange(len(ap)) // 3600
    res, thr_tail = {}, {}
    for name, (s_vn, s_a) in scores.items():
        t_lf = thr_label_free(s_vn)
        t_mcc = thr_val_mcc(s_a[cal], ya[cal])
        thr_tail[name] = t_mcc
        res[name] = {
            "fit_s": timing[name],
            "full": {**threshold_free(ya, s_a), "label_free": {**point_metrics(ya, s_a, t_lf), **event_metrics(ya, s_a, t_lf)}},
            "tail": {**threshold_free(ya[tail], s_a[tail]),
                     "label_free": {**point_metrics(ya[tail], s_a[tail], t_lf), **event_metrics(ya[tail], s_a[tail], t_lf)},
                     "val_mcc": {**point_metrics(ya[tail], s_a[tail], t_mcc), **event_metrics(ya[tail], s_a[tail], t_mcc)}},
            "val_normal_fpr_label_free": float((s_vn > t_lf).mean()),
        }
    lf_full = {k: thr_label_free(v[0]) for k, v in scores.items()}
    ci, raw = block_bootstrap(ya, {k: v[1] for k, v in scores.items()}, hours, lf_full, B=200)
    for k in res:
        res[k]["full"]["ci95"] = ci[k]
    ranked = sorted(res, key=lambda k: -res[k]["full"]["auroc"])
    return {
        "rows": {"normal_unique": int(n), "train": int(len(tr)), "val_normal": int(len(vn)),
                 "attack_period": int(len(ap)), "attack_rate": float(ya.mean()),
                 "tail_rows": int(len(tail)), "n_features": len(names), "sensors": SENSORS, "ocsvm_lof_cap": CAP},
        "models": res, "deep": dinfo,
        "isolation_forest_seeds_auroc": {"mean": float(np.mean(if_auc)), "sd": float(np.std(if_auc))},
        "paired_top2_auroc": {"a": ranked[0], "b": ranked[1], **paired_diff(raw, ranked[0], ranked[1])},
    }


def run_original_layout():
    """Same scorers on the paper's original layout (attack-period normal rows
    as 'normal', attack-only rows as anomalies), original 70/15/15 + halves split."""
    from src.swat import load_swat
    normal, attack = load_swat()
    n, a = len(normal), len(attack)
    seg = lambda df, s: df.assign(segment=s)
    parts = {"tr": seg(normal.iloc[:int(n * .7)], "a"), "vn": seg(normal.iloc[int(n * .7):int(n * .85)], "b"),
             "tn": seg(normal.iloc[int(n * .85):], "c"), "va": seg(attack.iloc[:a // 2], "d"), "ta": seg(attack.iloc[a // 2:], "e")}
    F = {k: build_features(v, group_col="segment", sensor_cols=CONTINUOUS_SENSORS)[0] for k, v in parts.items()}
    sc = StandardScaler().fit(F["tr"])
    Xtr = sc.transform(F["tr"])
    Xval, yval = sc.transform(np.vstack([F["vn"], F["va"]])), np.r_[np.zeros(len(F["vn"])), np.ones(len(F["va"]))]
    Xte, yte = sc.transform(np.vstack([F["tn"], F["ta"]])), np.r_[np.zeros(len(F["tn"])), np.ones(len(F["ta"]))]
    res = {}
    for name, m in make_models(seed=0, cap=CAP).items():
        m.fit(Xtr)
        sv, st = m.score(Xval), m.score(Xte)
        res[name] = {**threshold_free(yte, st), "val_mcc": point_metrics(yte, st, thr_val_mcc(sv, yval))}
        print(f"  [orig layout] {name} AUROC {res[name]['auroc']:.3f}", flush=True)
    return {"test_anomaly_rate": float(yte.mean()), "models": res}


def main():
    os.makedirs("results/revised", exist_ok=True)
    t0 = time.perf_counter()
    print("standard protocol...", flush=True)
    std = run_standard()
    print("original layout...", flush=True)
    orig = run_original_layout()
    out = {"standard": std, "original_layout_same_code": orig, "wall_s": round(time.perf_counter() - t0)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(std["rows"])
    for k, v in sorted(std["models"].items(), key=lambda kv: -kv[1]["full"]["auroc"]):
        fu, ta = v["full"], v["tail"]
        print(f"{k:17s} AUROC {fu['auroc']:.3f} {np.round(fu['ci95']['auroc'],3)} AUPRC {fu['auprc']:.3f} | LF MCC {fu['label_free']['mcc']:+.3f} "
              f"FPR {fu['label_free']['fpr']:.3f} evRec {fu['label_free']['event_recall']:.2f} | tail valMCC {ta['val_mcc']['mcc']:+.3f} | orig-layout AUROC {orig['models'].get(k, {}).get('auroc', float('nan')):.3f}")
    print("IF seeds", std["isolation_forest_seeds_auroc"], "paired", std["paired_top2_auroc"], "wall", out["wall_s"])


if __name__ == "__main__":
    main()
