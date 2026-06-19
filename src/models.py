"""The 4 unsupervised anomaly detectors compared in the paper.

Every model exposes a train_*(X_train) -> state and predict_*(state, X) ->
binary ndarray (1 = anomaly, 0 = normal) pair, so evaluate.py can loop over
them uniformly with an identical, pre-scaled feature matrix. None of the
train_* functions see labels — these are genuinely unsupervised fits, not
just "unsupervised at test time."

contamination/nu default to 0.1 as a literature-informed nominal anomaly
rate, not something fit from labels (that would quietly leak supervision
into an "unsupervised" method) — it's a named, overridable constant instead
of a magic number buried in a call site.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

DEFAULT_CONTAMINATION = 0.1
RANDOM_STATE = 42


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """Fit StandardScaler on train only, transform both. OCSVM/LOF/Z-score
    are distance- or scale-based; IsolationForest is scale-invariant but
    using the same scaled matrix keeps "identical feature matrix" literal.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


@dataclass
class ZScoreParams:
    mean: np.ndarray
    std: np.ndarray
    threshold: float


def train_zscore(X_train: np.ndarray, threshold: float = 3.0) -> ZScoreParams:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # constant columns would otherwise divide by zero
    return ZScoreParams(mean=mean, std=std, threshold=threshold)


def predict_zscore(params: ZScoreParams, X: np.ndarray) -> np.ndarray:
    z = np.abs((X - params.mean) / params.std)
    return (z.max(axis=1) > params.threshold).astype(int)


def train_isolation_forest(X_train: np.ndarray, contamination: float = DEFAULT_CONTAMINATION) -> IsolationForest:
    model = IsolationForest(
        n_estimators=100, contamination=contamination, random_state=RANDOM_STATE
    )
    model.fit(X_train)
    return model


def predict_isolation_forest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    return (model.predict(X) == -1).astype(int)


def train_one_class_svm(X_train: np.ndarray, nu: float = DEFAULT_CONTAMINATION) -> OneClassSVM:
    model = OneClassSVM(kernel="rbf", nu=nu)
    model.fit(X_train)
    return model


def predict_one_class_svm(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    return (model.predict(X) == -1).astype(int)


def train_lof(X_train: np.ndarray, contamination: float = DEFAULT_CONTAMINATION, n_neighbors: int = 20) -> LocalOutlierFactor:
    # novelty=True is required to call .predict() on held-out data later —
    # LOF's default mode (novelty=False) only supports fit_predict on the
    # same data it was fit on.
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    model.fit(X_train)
    return model


def predict_lof(model: LocalOutlierFactor, X: np.ndarray) -> np.ndarray:
    return (model.predict(X) == -1).astype(int)


MODELS = {
    "Z-score": (train_zscore, predict_zscore),
    "One-Class SVM": (train_one_class_svm, predict_one_class_svm),
    "LOF": (train_lof, predict_lof),
    "IsolationForest": (train_isolation_forest, predict_isolation_forest),
}
