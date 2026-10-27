"""Scoring + comparison table for the 4 anomaly detectors."""

import json
import time

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.models import MODELS


def evaluate(y_true, y_pred, model_name: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # MCC and balanced accuracy are reported alongside F1 because, unlike F1,
    # they collapse to ~0 / 0.5 for a degenerate "flag everything" predictor
    # (precision = base rate, recall = 1). On the concept-drift stream the
    # static model degenerates exactly that way on unseen regimes, so F1
    # alone would credit it with skill it doesn't have (see
    # experiments/concept_drift.py). matthews_corrcoef is undefined when a
    # batch has only one true class; guarded to 0.0 there.
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        mcc = 0.0
    return {
        "model": model_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": mcc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_all_models(X_train, X_test, y_test, return_predictions: bool = False,
                   return_models: bool = False):
    """Train+predict every model in MODELS on identical X, return one
    result dict per model with metrics + train/inference timing (seconds).

    Optional extras (appended to the return tuple in this order):
      - return_predictions=True -> a {model_name: y_pred} dict, row-aligned
        with X_test, so callers can plot without re-fitting.
      - return_models=True -> a {model_name: fitted_state} dict, so callers
        can persist the trained models (see src/persist.py).
    """
    results = []
    predictions = {}
    states = {}
    for name, (train_fn, predict_fn) in MODELS.items():
        t0 = time.perf_counter()
        state = train_fn(X_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = predict_fn(state, X_test)
        inference_time = time.perf_counter() - t0

        result = evaluate(y_test, y_pred, name)
        result["train_time_s"] = train_time
        result["inference_time_s"] = inference_time
        results.append(result)
        predictions[name] = y_pred
        states[name] = state

    extras = []
    if return_predictions:
        extras.append(predictions)
    if return_models:
        extras.append(states)
    return (results, *extras) if extras else results


def compare_models(results: list[dict], out_path: str = "results/metrics.json") -> pd.DataFrame:
    df = pd.DataFrame(results)[
        ["model", "precision", "recall", "f1", "train_time_s", "inference_time_s"]
    ]
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return df
