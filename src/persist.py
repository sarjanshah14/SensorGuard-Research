"""Save/load the trained pipeline so results are reproducible without
retraining (ROADMAP C3).

Everything needed to score new data is bundled into a single joblib file:
the fitted models, the StandardScaler that produced the feature scaling they
were trained on, and the feature names (so a caller knows what build_features
must produce). The `.joblib` itself is git-ignored — it's a regenerable
artifact of `python main.py`, not source — but the folder is tracked.
"""

import os

import joblib

MODELS_DIR = "results/models"
ARTIFACTS_PATH = os.path.join(MODELS_DIR, "anomaly_models.joblib")


def save_artifacts(states: dict, scaler, feature_names, path: str = ARTIFACTS_PATH) -> str:
    """Persist {model_name: fitted_state}, the scaler, and feature_names."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        "models": states,
        "scaler": scaler,
        "feature_names": list(feature_names),
        "model_order": list(states.keys()),
    }
    joblib.dump(bundle, path)
    return path


def load_artifacts(path: str = ARTIFACTS_PATH) -> dict:
    """Load the bundle saved by save_artifacts. Raises if it doesn't exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved models at {path}. Run `python main.py` first."
        )
    return joblib.load(path)
