"""Concept-drift detection for the streaming adaptation experiment.

A two-sample KS test's *p-value* is the wrong statistic to threshold here:
with thousands of rows per batch, p-values collapse to ~0 for almost any
real-world distributional wobble, so a p<0.05 count saturates to "100% of
features differ" regardless of how different they actually are (confirmed
empirically -- every threshold from 0.1 to 0.7 flagged 100% of validation
files). The KS *statistic* D itself (the max distance between the two
empirical CDFs, bounded in [0, 1]) is sample-size-robust and measures
effect size directly, so it's used as the drift score instead.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp


def drift_score(reference: np.ndarray, window: np.ndarray) -> float:
    """Mean KS D-statistic across features between `reference` and
    `window` (both (n_rows, n_features) arrays of the same feature matrix).
    """
    n_features = reference.shape[1]
    d_stats = np.empty(n_features)
    for j in range(n_features):
        d_stats[j], _ = ks_2samp(reference[:, j], window[:, j])
    return float(d_stats.mean())


@dataclass
class DriftDetector:
    """Flags drift when drift_score(reference, window) exceeds threshold."""
    reference: np.ndarray
    threshold: float

    def check(self, window: np.ndarray) -> tuple[bool, float]:
        score = drift_score(self.reference, window)
        return score >= self.threshold, score

    def update_reference(self, new_reference: np.ndarray) -> None:
        """Call after a retrain so future drift checks compare against the
        distribution the model was just retrained on, not the stale original.
        """
        self.reference = new_reference
