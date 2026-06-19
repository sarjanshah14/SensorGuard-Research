"""Scoring + comparison table for the 4 anomaly detectors."""

import json
import time

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.models import MODELS


def evaluate(y_true, y_pred, model_name: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_all_models(X_train, X_test, y_test) -> list[dict]:
    """Train+predict every model in MODELS on identical X, return one
    result dict per model with metrics + train/inference timing (seconds).
    """
    results = []
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
    return results


def compare_models(results: list[dict], out_path: str = "results/metrics.json") -> pd.DataFrame:
    df = pd.DataFrame(results)[
        ["model", "precision", "recall", "f1", "train_time_s", "inference_time_s"]
    ]
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return df
