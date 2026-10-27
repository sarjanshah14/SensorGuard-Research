"""Lightweight deep sequential baseline: an LSTM autoencoder.

Both external reviews asked for a deep baseline to test whether the classical
detectors' low compute is bought at a real accuracy cost. This module trains a
small LSTM autoencoder on windows of raw (scaled) sensor readings from normal
operation only, and flags a reading when its window reconstructs poorly. The
anomaly threshold is tuned on the validation split by MCC (rejecting degenerate
flag-everything cut-offs), exactly the discipline used for the classical
models' contamination, so the comparison is fair. Evaluation is per row, on the
same SKAB and SWaT test splits, with the same MCC / balanced-accuracy / F1.

The model is kept deliberately small (single-layer LSTM, 64 hidden units) so it
stays a *lightweight* deep baseline and its size/latency can be compared against
the classical detectors on the edge.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler

WINDOW = 30          # timesteps per window (30 s at 1 Hz, matches the 30 s feature horizon)
HIDDEN = 64
EPOCHS = 25          # enough for the loss to converge on both datasets (fairness)
BATCH = 256
LR = 1e-3
TRAIN_STRIDE = 2     # denser training windows so the small SKAB set is not starved
SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _unique_ordered(a):
    seen, out = set(), []
    for x in a:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def causal_windows(X_seg: np.ndarray, w: int) -> np.ndarray:
    """(n, f) -> (n, w, f); window ending at row k is rows [k-w+1 .. k],
    front-padded by repeating row 0 for the first w-1 rows (causal)."""
    n = len(X_seg)
    pad = np.repeat(X_seg[:1], w - 1, axis=0)
    Xp = np.concatenate([pad, X_seg], axis=0)
    idx = np.arange(n)[:, None] + np.arange(w)[None, :]
    return Xp[idx]


def strided_windows(X_seg: np.ndarray, w: int, stride: int) -> np.ndarray:
    """Full (non-padded) windows over a segment, taken every `stride` rows, for
    training. Builds only the sampled windows (no full-resolution materialise)."""
    if len(X_seg) < w:
        return np.empty((0, w, X_seg.shape[1]), dtype=X_seg.dtype)
    starts = np.arange(0, len(X_seg) - w + 1, stride)
    return X_seg[starts[:, None] + np.arange(w)[None, :]]


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)
        self.hidden = hidden

    def forward(self, x):                       # x: (B, W, F)
        _, (h, _) = self.enc(x)                 # h: (1, B, H)
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)  # (B, W, H)
        d, _ = self.dec(z)
        return self.out(d)                      # (B, W, F)


def train_autoencoder(train_windows: np.ndarray, n_features: int, verbose=True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = LSTMAutoencoder(n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    X = torch.tensor(train_windows, dtype=torch.float32)
    n = len(X)
    losses = []
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            xb = X[perm[i:i + BATCH]].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        losses.append(total / n)
        if verbose:
            print(f"  epoch {ep+1}/{EPOCHS}  train MSE {total/n:.5f}")
    return model, losses


def row_scores(model, X: np.ndarray, groups: np.ndarray, w: int) -> np.ndarray:
    """Per-row reconstruction error (causal window ending at each row), computed
    per segment so windows never cross a run/segment boundary."""
    model.eval()
    scores = np.empty(len(X), dtype=np.float64)
    pos = 0
    with torch.no_grad():
        for g in _unique_ordered(groups):
            seg = X[groups == g]
            win = causal_windows(seg, w)                 # (m, w, f)
            m = len(win)
            errs = np.empty(m)
            for i in range(0, m, 1024):
                xb = torch.tensor(win[i:i + 1024], dtype=torch.float32).to(DEVICE)
                rec = model(xb)
                e = ((rec - xb) ** 2).mean(dim=(1, 2)).cpu().numpy()
                errs[i:i + len(e)] = e
            scores[pos:pos + m] = errs
            pos += m
    return scores


def _metrics(y, pred):
    try:
        mcc = float(matthews_corrcoef(y, pred))
    except ValueError:
        mcc = 0.0
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": mcc,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def tune_threshold(scores_val, y_val):
    """Pick the reconstruction-error cut-off that maximises validation MCC,
    rejecting degenerate cut-offs that flag almost everything (recall > 0.99)."""
    cands = np.quantile(scores_val, np.linspace(0.50, 0.999, 80))
    best_t, best_mcc = None, -2.0
    for t in cands:
        pred = (scores_val > t).astype(int)
        r = recall_score(y_val, pred, zero_division=0)
        if r > 0.99:
            continue
        try:
            mcc = matthews_corrcoef(y_val, pred)
        except ValueError:
            mcc = 0.0
        if mcc > best_mcc:
            best_t, best_mcc = float(t), float(mcc)
    if best_t is None:                       # fallback: 95th percentile
        best_t = float(np.quantile(scores_val, 0.95))
    return best_t, best_mcc


def run_deep_baseline(sensor_cols, X_train_df, val_df, test_df, group_col, name):
    """Train + evaluate the LSTM AE. Each df must have the sensor columns, an
    `anomaly` column, and `group_col` marking contiguous segments. Training uses
    only X_train_df (normal). Returns a metrics dict for the test split plus
    footprint numbers."""
    scaler = StandardScaler().fit(X_train_df[sensor_cols].to_numpy())

    def prep(df):
        X = scaler.transform(df[sensor_cols].to_numpy()).astype(np.float32)
        return X, df[group_col].to_numpy(), df["anomaly"].to_numpy()

    Xtr, gtr, _ = prep(X_train_df)
    Xva, gva, yva = prep(val_df)
    Xte, gte, yte = prep(test_df)
    n_features = len(sensor_cols)

    # strided full windows over normal data (per segment), built directly so no
    # full-resolution window array is ever materialised
    tw = [strided_windows(Xtr[gtr == g], WINDOW, TRAIN_STRIDE) for g in _unique_ordered(gtr)]
    train_windows = np.concatenate([w for w in tw if len(w)], axis=0)
    print(f"[{name}] training windows: {train_windows.shape}, device={DEVICE.type}")

    t0 = time.perf_counter()
    model, losses = train_autoencoder(train_windows, n_features)
    train_s = time.perf_counter() - t0

    s_val = row_scores(model, Xva, gva, WINDOW)
    thr, val_mcc = tune_threshold(s_val, yva)

    t0 = time.perf_counter()
    s_test = row_scores(model, Xte, gte, WINDOW)
    infer_s = time.perf_counter() - t0
    pred = (s_test > thr).astype(int)
    m = _metrics(yte, pred)

    n_params = sum(p.numel() for p in model.parameters())
    size_kb = n_params * 4 / 1024.0
    m.update({
        "model": "LSTM-AE",
        "val_mcc": round(val_mcc, 3),
        "threshold": thr,
        "train_losses": [round(float(l), 6) for l in losses],
        "train_time_s": round(train_s, 1),
        "infer_ms_per_sample": round(infer_s / len(Xte) * 1000, 4),
        "n_params": int(n_params),
        "size_KB": round(size_kb, 1),
        "test_rows": int(len(yte)),
    })
    return m, s_test, yte
