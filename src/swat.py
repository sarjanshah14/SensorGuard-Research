"""SWaT (Secure Water Treatment) loading + feature setup — the paper's second
real dataset, used to test whether the SKAB findings (the 6-model comparison
and the concept-drift result) generalise beyond a single testbed.

SWaT is the iTrust/SUTD water-treatment testbed (Dec 2015 collection): 51
process variables sampled at 1 Hz, with a long normal-operation run and a
separate run containing 36 labelled attacks. We use it through the same
unsupervised pipeline as SKAB (train on normal only, score on a held-out
normal+anomaly mix), reusing src/models.py and src/evaluate.py unchanged.

Two data realities, handled honestly (not hidden):

1. The provided normal.csv is two concatenated runs. A 395,298-row block
   has all 51 sensors; a 991,800-row block has 7 sensors
   (MV101, AIT201, MV201, P201, P202, P204, MV303) recorded as entirely
   blank. We keep only the fully-recorded block for training rather than
   dropping those 7 columns (which are present and informative at test time)
   or imputing ~1M fabricated values. 395k rows (~4.5 days) is ample.

2. SWaT mixes 25 continuous sensors (flows, levels, pressures, analysers)
   with 26 discrete on/off actuators (valves, pumps). We engineer features
   on the 25 continuous sensors with the *same* rolling-statistics recipe
   used for SKAB (src/features.build_features), so the methodology is
   identical across both datasets and the comparison is apples-to-apples.
"""

import pandas as pd

# 25 continuous sensors (unique-value count > 10 in the normal run); the
# remaining 26 columns are discrete actuator states. Classification verified
# empirically against normal.csv.
CONTINUOUS_SENSORS = [
    "FIT101", "LIT101", "AIT201", "AIT202", "AIT203", "FIT201", "DPIT301",
    "FIT301", "LIT301", "AIT401", "AIT402", "FIT401", "LIT401", "AIT501",
    "AIT502", "AIT503", "AIT504", "FIT501", "FIT502", "FIT503", "FIT504",
    "PIT501", "PIT502", "PIT503", "FIT601",
]
LABEL_COL = "Normal/Attack"
TS_FORMAT = "%d/%m/%Y %I:%M:%S %p"


def _swat_dir() -> str:
    """Resolve the local SWaT path. Prefer the SWAT_DIR env var (used inside
    the resource-constrained Docker container, where the data is mounted and
    there is no network / Kaggle credentials); otherwise fall back to
    kagglehub's cached download."""
    import os
    env = os.environ.get("SWAT_DIR")
    if env:
        return env
    import kagglehub
    return kagglehub.dataset_download(
        "vishala28/swat-dataset-secure-water-treatment-system"
    )


def _read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["datetime"] = pd.to_datetime(df["Timestamp"], format=TS_FORMAT)
    df = df.set_index("datetime").sort_index()
    return df


def load_swat():
    """Return (normal_df, attack_df), both datetime-indexed with an integer
    `anomaly` column (0 normal / 1 attack). normal_df is restricted to the
    fully-recorded block (no blank-sensor rows). attack_df is all attacks.
    """
    import os
    d = _swat_dir()
    normal = _read(os.path.join(d, "normal.csv"))
    attack = _read(os.path.join(d, "attack.csv"))

    # keep only fully-recorded normal rows (drops the 991,800-row block where
    # 7 sensors are blank); see module docstring
    normal = normal.dropna(subset=CONTINUOUS_SENSORS)

    normal["anomaly"] = 0
    attack["anomaly"] = (attack[LABEL_COL].str.strip() == "Attack").astype(int)
    return normal, attack


def load_swat_bounded(n_normal: int, n_attack: int):
    """Read only the first n_normal / n_attack rows of each file, so a
    memory-constrained host (e.g. the 1 GB edge container) never has to hold
    the full ~1.4 M-row normal.csv in RAM. The first 395,298 normal rows are
    the fully-recorded block, so any n_normal <= that is clean."""
    import os
    d = _swat_dir()
    normal = pd.read_csv(os.path.join(d, "normal.csv"), skipinitialspace=True, nrows=n_normal)
    attack = pd.read_csv(os.path.join(d, "attack.csv"), skipinitialspace=True, nrows=n_attack)
    frames = []
    for df in (normal, attack):
        df.columns = df.columns.str.strip()
        df["datetime"] = pd.to_datetime(df["Timestamp"], format=TS_FORMAT)
        frames.append(df.set_index("datetime").sort_index())
    normal, attack = frames
    normal["anomaly"] = 0
    attack["anomaly"] = (attack[LABEL_COL].str.strip() == "Attack").astype(int)
    return normal, attack
