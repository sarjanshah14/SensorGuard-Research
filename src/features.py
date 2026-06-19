"""Feature engineering for the SKAB anomaly-detection pipeline.

All stats are causal (computed only from current + past rows) and computed
per source experiment (group_col), never across experiment boundaries —
SKAB's CSVs are independent runs with their own clocks, so a rolling window
that crossed from the tail of one run into the head of another would mix
unrelated physical conditions into a single "feature."

Never pass anomaly/changepoint columns in here as features.
"""

import numpy as np
import pandas as pd

SENSOR_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS",
]


def build_features(df: pd.DataFrame, group_col: str | None = None, sensor_cols=None):
    """Build a causal feature matrix from raw SKAB sensor columns.

    Returns (X, feature_names): X is an (n_rows, n_features) float ndarray
    row-aligned with df (same order, same length). hour/day_of_week are
    included per the original spec but carry little signal here — each
    SKAB CSV is a single short run, so both are near-constant within a
    file; time_since_start is the temporal feature that actually varies
    within a run.
    """
    sensor_cols = sensor_cols or SENSOR_COLS
    groups = [g for _, g in df.groupby(group_col, sort=False)] if group_col else [df]

    blocks = []
    for g in groups:
        ts = g.index
        feats = pd.DataFrame(index=range(len(g)))
        feats["hour"] = ts.hour.values
        feats["day_of_week"] = ts.dayofweek.values
        feats["time_since_start"] = (ts - ts[0]).total_seconds().values

        for col in sensor_cols:
            s = g[col].reset_index(drop=True)
            feats[col] = s.values
            feats[f"{col}_roll_mean_5s"] = s.rolling(5, min_periods=1).mean().values
            feats[f"{col}_roll_std_5s"] = s.rolling(5, min_periods=1).std().fillna(0).values
            feats[f"{col}_roll_mean_30s"] = s.rolling(30, min_periods=1).mean().values
            feats[f"{col}_delta"] = s.diff().fillna(0).values

        blocks.append(feats)

    feature_df = pd.concat(blocks, ignore_index=True)
    feature_names = list(feature_df.columns)
    return feature_df.to_numpy(dtype=np.float64), feature_names
