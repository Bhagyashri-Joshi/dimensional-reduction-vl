"""Preprocessing helpers: scaling and reproducible stratified sampling (TRD section 4)."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on TRAINING data only."""
    return StandardScaler().fit(X_train)


def stratified_sample(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproducible class-balanced subset. Returns (X_sample, y_sample, indices).

    If ``n`` >= number of rows the full data is returned (in original order).
    """
    if n <= 0:
        raise ValueError("Sample size must be positive.")
    total = X.shape[0]
    if n >= total:
        idx = np.arange(total)
        return np.asarray(X), np.asarray(y), idx
    idx, _ = train_test_split(np.arange(total), train_size=n, stratify=y, random_state=seed)
    idx = np.sort(idx)
    return np.asarray(X)[idx], np.asarray(y)[idx], idx
