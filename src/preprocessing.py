"""SKAB data loading for the research pipeline.

Mirrors the CSV-reading conventions already used in skab.py (";" delimiter,
datetime index) but keeps the anomaly-free baseline free of fabricated
anomaly/changepoint columns, since that file is for training only and is
never evaluated against labels.
"""

import os
import glob

import pandas as pd

SKAB_ROOT = "SKAB"
FAULT_FOLDERS = ("other", "valve1", "valve2")


def load_skab_single(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=";", parse_dates=["datetime"], index_col="datetime")
    df = df.ffill()
    return df


def load_anomaly_free(data_dir: str = SKAB_ROOT) -> pd.DataFrame:
    filepath = os.path.join(data_dir, "anomaly-free", "anomaly-free.csv")
    return load_skab_single(filepath)


def load_skab_all(data_dir: str = SKAB_ROOT) -> pd.DataFrame:
    frames = []
    for folder in FAULT_FOLDERS:
        for filepath in sorted(glob.glob(os.path.join(data_dir, folder, "*.csv"))):
            df = load_skab_single(filepath)
            name = os.path.splitext(os.path.basename(filepath))[0]
            df["experiment_type"] = folder
            # Filenames repeat across folders (e.g. valve1/0.csv, valve2/0.csv),
            # so source_file must include the folder to stay unique per run.
            df["source_file"] = f"{folder}/{name}"
            frames.append(df)
    return pd.concat(frames)


def chronological_split(fault_df: pd.DataFrame, test_size: float = 0.2):
    """Per-source-file chronological 80/20 split, then concatenated.

    A single global time cut across pooled files would put some whole
    experiments entirely in train and others entirely in test (each SKAB
    CSV is an independent run with its own clock), so the split is done
    within each source_file before pooling.
    """
    train_frames, test_frames = [], []
    for _, group in fault_df.groupby("source_file", sort=False):
        group = group.sort_index()
        cut = int(len(group) * (1 - test_size))
        train_frames.append(group.iloc[:cut])
        test_frames.append(group.iloc[cut:])
    train_df = pd.concat(train_frames)
    test_df = pd.concat(test_frames)
    return train_df, test_df
