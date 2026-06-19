"""Temporal post-processing of anomaly predictions (ROADMAP B5).

Real SKAB faults are sustained — they last many seconds in a row. So an
isolated one- or two-second spike of "anomaly" surrounded by normal
predictions is almost always a false alarm, not a real fault. This removes
predicted-anomaly runs shorter than `min_run`, done per source experiment
(never merging runs across different files).

In image terms this is a morphological "opening" of the prediction mask —
a standard, physically motivated cleanup, not a scoring trick. It trades a
little recall (a few genuine short detections get dropped) for precision
(many isolated false alarms disappear).
"""

import numpy as np


def _filter_short_runs(arr: np.ndarray, min_run: int) -> np.ndarray:
    """Set any run of consecutive 1s shorter than min_run back to 0."""
    arr = arr.copy()
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == 1:
            j = i
            while j < n and arr[j] == 1:
                j += 1
            if j - i < min_run:
                arr[i:j] = 0
            i = j
        else:
            i += 1
    return arr


def smooth_min_run(y_pred, groups, min_run: int) -> np.ndarray:
    """Apply _filter_short_runs within each contiguous group (source_file).

    groups must be row-aligned with y_pred (same length, same order). A
    min_run <= 1 is a no-op (returns the predictions unchanged), which makes
    min_run=1 a clean "no smoothing" baseline in sweeps.
    """
    y_pred = np.asarray(y_pred)
    if min_run <= 1:
        return y_pred
    groups = np.asarray(groups)
    out = y_pred.copy()
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            out[start:i] = _filter_short_runs(y_pred[start:i], min_run)
            start = i
    return out
