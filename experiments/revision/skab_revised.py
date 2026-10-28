"""SKAB under the revised protocol (revision/AUDIT.md E2, E3, E6, E7, E8).

Run: python -m experiments.revision.skab_revised   -> results/revised/skab.json

Pooled protocol (same split structure as the original paper, fixed):
  - features computed on each full run BEFORE dropping anomalous rows (A3),
    without time_since_start (A2);
  - per-file chronological 80/20; 20% of files' train portions held out as
    labelled validation (same file_holdout_split seed as the original);
  - train = anomaly-free run + normal rows of the remaining train portions;
  - thresholds per model: label-free (99th pct of validation-normal scores),
    validation-MCC, validation-F1 (sensitivity only);
  - 95% CIs by bootstrap over the 34 test files.
SKAB-benchmark protocol (waico/SKAB core/utils.py: first 400 rows of each file
train, rest test) is run separately so results can be read against the
official leaderboard.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src import deep_baseline as dbl
from src.features import SENSOR_COLS, build_features
from src.preprocessing import file_holdout_split, load_anomaly_free, load_skab_all
from experiments.revision.common import (
    block_bootstrap, event_metrics, make_models, paired_diff, point_metrics,
    threshold_free, thr_label_free, thr_val_f1, thr_val_mcc,
)

OUT = "results/revised/skab.json"
SEEDS = list(range(10))
DEEP_SEEDS = [0, 1, 2]


def frames(include_elapsed=False):
    af = load_anomaly_free()
    af["source_file"], af["anomaly"], af["experiment_type"] = "anomaly-free/anomaly-free", 0, "anomaly-free"
    faults = load_skab_all()
    df = pd.concat([af, faults])
    X, names = build_features(df, group_col="source_file", include_elapsed=include_elapsed)
    df = df.reset_index()
    df["pos"] = df.groupby("source_file").cumcount()
    df["n"] = df.groupby("source_file")["pos"].transform("max") + 1
    df["is_test"] = (df["source_file"] != "anomaly-free/anomaly-free") & (df["pos"] >= (df["n"] * 0.8).astype(int))
    train_part = df[(df["source_file"] != "anomaly-free/anomaly-free") & ~df["is_test"]]
    _, val = file_holdout_split(train_part, val_frac=0.2)
    df["is_val"] = df["source_file"].isin(set(val["source_file"])) & ~df["is_test"]
    df["is_train"] = ~df["is_test"] & ~df["is_val"] & (df["anomaly"] == 0)
    return df, X, names


def deep_scores(df, seed):
    """LSTM-AE on raw scaled sensors; scores for val and test rows. Windows are
    causal within each file and may look back into the file's earlier rows."""
    dbl.SEED = seed
    torch.manual_seed(seed); np.random.seed(seed)
    sc = StandardScaler().fit(df.loc[df.is_train, SENSOR_COLS].to_numpy())
    Xs = sc.transform(df[SENSOR_COLS].to_numpy()).astype(np.float32)
    g = df["source_file"].to_numpy()
    tr_w = []
    for f in pd.unique(g[df.is_train]):
        m = (g == f) & df.is_train.to_numpy()
        tr_w.append(dbl.strided_windows(Xs[m], dbl.WINDOW, dbl.TRAIN_STRIDE))
    model, losses = dbl.train_autoencoder(np.concatenate([w for w in tr_w if len(w)]), len(SENSOR_COLS), verbose=False)
    s = dbl.row_scores(model, Xs, g, dbl.WINDOW)  # full runs, causal
    return s


def evaluate_pooled(df, X):
    tr, va, te = df.is_train.to_numpy(), df.is_val.to_numpy(), df.is_test.to_numpy()
    y = df["anomaly"].to_numpy()
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    va_norm = va & (y == 0)
    scores, models_out = {}, {}
    for name, m in make_models(seed=0).items():
        s = m.fit(Xs[tr]).score(Xs)
        scores[name] = s
    # IF seed spread
    if_auc = []
    for sd in SEEDS:
        s = make_models(seed=sd)["IsolationForest"].fit(Xs[tr]).score(Xs)
        if_auc.append(threshold_free(y[te], s[te])["auroc"])
    deep = [deep_scores(df, sd) for sd in DEEP_SEEDS]
    deep_auc = [threshold_free(y[te], s[te])["auroc"] for s in deep]
    scores["LSTM-AE"] = deep[0]

    groups = df["source_file"].to_numpy()
    thr = {}
    for name, s in scores.items():
        t_lf = thr_label_free(s[va_norm])
        t_mcc = thr_val_mcc(s[va], y[va])
        t_f1 = thr_val_f1(s[va], y[va])
        thr[name] = t_mcc
        models_out[name] = {
            **threshold_free(y[te], s[te]),
            "label_free": {**point_metrics(y[te], s[te], t_lf), **event_metrics(y[te], s[te], t_lf, groups[te])},
            "val_mcc": {**point_metrics(y[te], s[te], t_mcc), **event_metrics(y[te], s[te], t_mcc, groups[te])},
            "val_f1": point_metrics(y[te], s[te], t_f1),
        }
    ci, raw = block_bootstrap(y[te], {k: v[te] for k, v in scores.items()}, groups[te], thr, B=500)
    for k in models_out:
        models_out[k]["ci95"] = ci[k]
    ranked = sorted(models_out, key=lambda k: -models_out[k]["auroc"])
    return {
        "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                 "test_anomaly_rate": float(y[te].mean()), "n_features": int(X.shape[1])},
        "models": models_out,
        "isolation_forest_seeds_auroc": {"mean": float(np.mean(if_auc)), "sd": float(np.std(if_auc)), "values": if_auc},
        "lstm_ae_seeds_auroc": {"mean": float(np.mean(deep_auc)), "sd": float(np.std(deep_auc)), "values": deep_auc},
        "paired_top2_auroc": {"a": ranked[0], "b": ranked[1], **paired_diff(raw, ranked[0], ranked[1])},
    }


def evaluate_per_file(df, X):
    """Official SKAB split: first 400 rows of each fault file train, rest test.
    Label-free threshold = 99th pct of the file's own training scores."""
    faults = df[df["source_file"] != "anomaly-free/anomaly-free"]
    res = {}
    first400_anom = float(faults[faults["pos"] < 400]["anomaly"].mean())
    for name in make_models():
        ys, ps, aucs = [], [], []
        for f, g in faults.groupby("source_file", sort=False):
            idx = g.index.to_numpy()
            tr, te = idx[g["pos"].to_numpy() < 400], idx[g["pos"].to_numpy() >= 400]
            sc = StandardScaler().fit(X[tr])
            m = make_models()[name].fit(sc.transform(X[tr]))
            s_tr, s_te = m.score(sc.transform(X[tr])), m.score(sc.transform(X[te]))
            yt = df.loc[te, "anomaly"].to_numpy()
            ys.append(yt); ps.append(s_te > np.quantile(s_tr, 0.99))
            if 0 < yt.mean() < 1:
                aucs.append(threshold_free(yt, s_te)["auroc"])
        y, p = np.concatenate(ys), np.concatenate(ps)
        pm = point_metrics(y, p.astype(float), 0.5)
        res[name] = {**pm, "FAR": pm["fpr"], "MAR": 1 - pm["recall"], "mean_file_auroc": float(np.mean(aucs))}
    return {"first400_anomaly_rate": first400_anom, "models": res}


def ablation():
    """A2/A3: does time_since_start or feature-after-drop change results?"""
    out = {}
    for label, elapsed in [("revised_no_elapsed", False), ("full_run_with_elapsed", True)]:
        df, X, _ = frames(include_elapsed=elapsed)
        tr, te, y = df.is_train.to_numpy(), df.is_test.to_numpy(), df["anomaly"].to_numpy()
        Xs = StandardScaler().fit(X[tr]).transform(X)
        out[label] = {n: threshold_free(y[te], make_models()[n].fit(Xs[tr]).score(Xs)[te])["auroc"]
                      for n in ["IsolationForest", "EllipticEnvelope", "PCA-recon"]}
    # original pipeline: features built after dropping anomalous rows, with elapsed
    af = load_anomaly_free(); af["source_file"], af["anomaly"] = "anomaly-free/anomaly-free", 0
    from src.preprocessing import chronological_split
    ftr, fte = chronological_split(load_skab_all())
    trd = pd.concat([af, ftr[ftr.anomaly == 0]])
    Xtr, _ = build_features(trd, group_col="source_file"); Xte, _ = build_features(fte, group_col="source_file")
    sc = StandardScaler().fit(Xtr)
    out["original_pipeline"] = {n: threshold_free(fte.anomaly.to_numpy(), make_models()[n].fit(sc.transform(Xtr)).score(sc.transform(Xte)))["auroc"]
                                for n in ["IsolationForest", "EllipticEnvelope", "PCA-recon"]}
    return out


def main():
    os.makedirs("results/revised", exist_ok=True)
    df, X, names = frames()
    res = {"pooled": evaluate_pooled(df, X), "skab_benchmark_protocol": evaluate_per_file(df, X),
           "ablation_auroc": ablation(), "features": names}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    p = res["pooled"]
    print(p["rows"])
    for k, v in sorted(p["models"].items(), key=lambda kv: -kv[1]["auroc"]):
        print(f"{k:17s} AUROC {v['auroc']:.3f} {np.round(v['ci95']['auroc'],3)}  AUPRC {v['auprc']:.3f}  "
              f"MCC(valMCC) {v['val_mcc']['mcc']:+.3f}  MCC(labelfree) {v['label_free']['mcc']:+.3f} FPR_lf {v['label_free']['fpr']:.3f}  MCC(valF1) {v['val_f1']['mcc']:+.3f}")
    print("IF seeds", p["isolation_forest_seeds_auroc"]["mean"], p["isolation_forest_seeds_auroc"]["sd"])
    print("LSTM seeds", p["lstm_ae_seeds_auroc"])
    print("paired", p["paired_top2_auroc"])
    print("SKAB protocol", res["skab_benchmark_protocol"]["first400_anomaly_rate"])
    for k, v in res["skab_benchmark_protocol"]["models"].items():
        print(f"  {k:17s} F1 {v['f1']:.3f} FAR {v['FAR']:.3f} MAR {v['MAR']:.3f} MCC {v['mcc']:+.3f} fileAUROC {v['mean_file_auroc']:.3f}")
    print("ablation", json.dumps(res["ablation_auroc"], indent=1))


if __name__ == "__main__":
    main()
