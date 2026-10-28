"""Shared pieces of the revised protocol (revision/AUDIT.md, experiments E1-E8).

Differences from the original pipeline, each tied to an audit finding:
  - every detector returns a continuous anomaly score (higher = more anomalous),
    so threshold-free metrics (AUROC, AUPRC) can be reported (audit C);
  - thresholds are chosen per model by an explicit rule, the same rule for every
    model (audit B1): label-free (quantile of validation-normal scores) or
    validation-MCC (uses labelled validation anomalies, i.e. semi-supervised
    calibration, audit B2), plus validation-F1 only to show sensitivity;
  - uncertainty by block bootstrap over files / hour blocks, never i.i.d. rows
    (audit B3);
  - Elliptic Envelope uses sklearn's default (robust) MCD support fraction
    (audit D).
"""

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def _cap(X, cap):
    # ponytail: deterministic stride subsample for O(n^2) models; cap is reported in results
    if cap is None or len(X) <= cap:
        return X
    return X[np.linspace(0, len(X) - 1, cap).astype(int)]


class ZScore:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        return self

    def score(self, X):
        return np.abs((X - self.mu) / self.sd).max(1)


class SK:
    """Wraps an sklearn/pyod detector; sign flips so higher = more anomalous."""
    def __init__(self, make, how, cap=None):
        self.make, self.how, self.cap = make, how, cap

    def fit(self, X):
        self.m = self.make().fit(_cap(X, self.cap))
        self.make = None  # lambdas don't pickle; fitted wrappers must serialise for the size benchmark
        return self

    def score(self, X):
        if self.how == "neg_score_samples":
            return -self.m.score_samples(X)
        if self.how == "neg_decision":
            return -self.m.decision_function(X)
        return self.m.decision_function(X)  # pyod: higher = more anomalous


class PCARecon:
    def fit(self, X):
        self.p = PCA(n_components=0.95, random_state=0).fit(X)
        return self

    def score(self, X):
        return ((X - self.p.inverse_transform(self.p.transform(X))) ** 2).sum(1)


class Elliptic:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X):
        self.p = PCA(n_components=10, whiten=True, random_state=0).fit(X)
        # default support_fraction -> robust MCD (the original used 1.0 = plain Gaussian)
        self.m = EllipticEnvelope(random_state=self.seed).fit(self.p.transform(X))
        return self

    def score(self, X):
        return self.m.mahalanobis(self.p.transform(X))


def make_models(seed=0, cap=None, cap_ecdf=None):
    """All classical detectors. `cap` bounds training rows for OCSVM/LOF;
    `cap_ecdf` for ECOD/COPOD, which PyOD rescores against all training rows."""
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.hbos import HBOS
    return {
        "Z-score": ZScore(),
        "PCA-recon": PCARecon(),
        "EllipticEnvelope": Elliptic(seed),
        "IsolationForest": SK(lambda: IsolationForest(n_estimators=100, random_state=seed), "neg_score_samples"),
        "One-Class SVM": SK(lambda: OneClassSVM(kernel="rbf", nu=0.1), "neg_decision", cap),
        "LOF": SK(lambda: LocalOutlierFactor(n_neighbors=20, novelty=True), "neg_score_samples", cap),
        "ECOD": SK(lambda: ECOD(), "pyod", cap_ecdf),
        "COPOD": SK(lambda: COPOD(), "pyod", cap_ecdf),
        "HBOS": SK(lambda: HBOS(), "pyod"),
    }


# ---------------- thresholds ----------------

def thr_label_free(s_val_normal, q=0.99):
    """Target ~1% false alarms on held-out normal data; needs no anomaly labels."""
    return float(np.quantile(s_val_normal, q))


def _best(s, y, metric):
    cands = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 120)))
    best_t, best_v = cands[-1], -np.inf
    for t in cands:
        p = s > t
        if p.mean() > 0.99 * 1 and y.mean() < 0.99:  # never pick flag-everything
            continue
        v = metric(y, p)
        if v > best_v:
            best_t, best_v = t, v
    return float(best_t)


def thr_val_mcc(s_val, y_val):
    return _best(s_val, y_val, mcc)


def thr_val_f1(s_val, y_val):
    return _best(s_val, y_val, f1)


# ---------------- metrics ----------------

def _cm(y, p):
    y, p = y.astype(bool), p.astype(bool)
    tp = np.sum(y & p); tn = np.sum(~y & ~p); fp = np.sum(~y & p); fn = np.sum(y & ~p)
    return tp, tn, fp, fn


def mcc(y, p):
    tp, tn, fp, fn = map(float, _cm(y, p))
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if d == 0 else (tp * tn - fp * fn) / d


def f1(y, p):
    tp, _, fp, fn = _cm(y, p)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def events(y, groups=None):
    """(start, end) index pairs of contiguous anomaly runs, never crossing groups."""
    y = np.asarray(y).astype(int)
    g = np.zeros(len(y)) if groups is None else np.asarray(groups)
    cuts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, len(y)]
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        edge = np.diff(np.r_[0, y[lo:hi], 0])
        out += [(lo + a, lo + b) for a, b in zip(np.flatnonzero(edge[:-1] == 1), np.flatnonzero(edge[1:] == -1))]
    return out


def point_metrics(y, s, t):
    p = s > t
    tp, tn, fp, fn = _cm(y, p)
    return {
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(f1(y, p)), "mcc": float(mcc(y, p)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "bacc": float(0.5 * (tp / (tp + fn) + tn / (tn + fp))) if (tp + fn) and (tn + fp) else 0.5,
    }


def event_metrics(y, s, t, groups=None):
    p = s > t
    ev = events(y, groups)
    hit, delays = 0, []
    for a, b in ev:
        idx = np.flatnonzero(p[a:b + 1])
        if len(idx):
            hit += 1
            delays.append(int(idx[0]))
    return {"n_events": len(ev), "event_recall": hit / len(ev) if ev else 0.0,
            "median_delay_s": float(np.median(delays)) if delays else None}


def threshold_free(y, s):
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}


def block_bootstrap(y, scores_by_model, blocks, thr_by_model, B=300, seed=0):
    """95% CIs for AUROC, AUPRC and MCC by resampling whole blocks (files or
    hours) with replacement. Same resamples for every model, so paired
    differences are valid."""
    rng = np.random.default_rng(seed)
    ub, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ub))]
    out = {m: {"auroc": [], "auprc": [], "mcc": []} for m in scores_by_model}
    for _ in range(B):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(ub), len(ub))])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for m, s in scores_by_model.items():
            sb = s[idx]
            out[m]["auroc"].append(roc_auc_score(yb, sb))
            out[m]["auprc"].append(average_precision_score(yb, sb))
            out[m]["mcc"].append(mcc(yb, sb > thr_by_model[m]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res = {m: {k: ci(v) for k, v in d.items()} for m, d in out.items()}
    return res, out


def paired_diff(boot_raw, a, b, metric="auroc"):
    d = np.array(boot_raw[a][metric]) - np.array(boot_raw[b][metric])
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_b_ge_a": float((d <= 0).mean())}


if __name__ == "__main__":
    # self-check: metrics + event extraction
    y = np.array([0, 0, 1, 1, 0, 1, 0])
    assert events(y) == [(2, 3), (5, 5)]
    assert events(y, groups=np.array([0, 0, 0, 1, 1, 1, 1])) == [(2, 2), (3, 3), (5, 5)]
    s = np.array([0, 0, 2, 0, 0, 2, 0.0])
    assert event_metrics(y, s, 1)["event_recall"] == 1.0
    assert abs(mcc(y, y.astype(bool)) - 1) < 1e-12 and mcc(y, np.ones(7, bool)) == 0
    assert threshold_free(y, y.astype(float))["auroc"] == 1.0
    print("common.py self-check ok")
