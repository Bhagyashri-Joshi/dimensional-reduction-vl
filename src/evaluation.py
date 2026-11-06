"""Leakage-free classification comparison (PRD 5.6, TRD 4.1, 6).

Protocol for every method
1. Use the predefined HAR train/test split (never re-split).
2. Fit scaler / reducer on TRAIN features only.
3. Transform TEST features with the already-fitted objects.
4. Fit Logistic Regression on transformed train data; evaluate on transformed test data.

t-SNE is intentionally absent: scikit-learn's TSNE has no out-of-sample ``transform``.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .data_loader import ACTIVITY_NAMES, HARData
from .dimensionality_reduction import ParameterError, make_umap
from .preprocessing import fit_scaler, stratified_sample

METHODS = ("Baseline", "PCA", "Truncated SVD", "UMAP")
METHOD_LABELS = {
    "Baseline": "Baseline (original features)",
    "PCA": "PCA + Logistic Regression",
    "Truncated SVD": "Truncated SVD + Logistic Regression",
    "UMAP": "UMAP + Logistic Regression",
}
LOGREG_CONFIG = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
CLASS_LABELS = sorted(ACTIVITY_NAMES)


@dataclass
class ClassificationResult:
    method: str
    n_dims: int
    is_original: bool
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    reduction_time_s: float | None  # None = no reduction step (baseline)
    classifier_time_s: float
    confusion: np.ndarray
    converged: bool
    settings: dict[str, Any]
    n_train: int
    n_test: int

    @property
    def label(self) -> str:
        return METHOD_LABELS[self.method]


def make_classifier(seed: int) -> LogisticRegression:
    """The single Logistic Regression configuration shared by all four pipelines."""
    return LogisticRegression(random_state=int(seed), **LOGREG_CONFIG)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    kw = {"average": "macro", "zero_division": 0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, **kw)),
        "recall_macro": float(recall_score(y_true, y_pred, **kw)),
        "f1_macro": float(f1_score(y_true, y_pred, **kw)),
        "confusion": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS),
    }


def validate_comparison_params(method: str, n_components: int, umap_params: dict, n_train: int, n_features: int) -> None:
    if method in ("PCA", "Truncated SVD"):
        limit = min(n_train, n_features) if method == "PCA" else n_features - 1
        if not 1 <= n_components <= limit:
            raise ParameterError(f"{method}: components must be between 1 and {limit} (got {n_components}).")
    if method == "UMAP":
        nn, md = int(umap_params["n_neighbors"]), float(umap_params["min_dist"])
        if not 2 <= nn < n_train:
            raise ParameterError(f"UMAP n_neighbors must be between 2 and {n_train - 1} (got {nn}).")
        if not 0.0 <= md <= 1.0:
            raise ParameterError(f"UMAP min_dist must be between 0.0 and 1.0 (got {md:g}).")
        if not 1 <= n_components < n_train - 1:
            raise ParameterError(f"UMAP components must be between 1 and {n_train - 2} (got {n_components}).")
        fs = int(umap_params.get("fit_subset", 0) or 0)
        if fs and fs < max(nn + 1, n_components + 2):
            raise ParameterError(
                f"UMAP fit subset must be 0 (full training set) or at least {max(nn + 1, n_components + 2)} (got {fs}). "
                "A value larger than the training set simply uses the whole training set."
            )


def fit_transform_split(
    method: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    seed: int,
    umap_params: dict | None = None,
    y_train: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit preprocessing + reducer on ``X_train`` only, then transform both splits.

    For UMAP, ``umap_params["fit_subset"] > 0`` fits the reducer on a stratified subset of the
    TRAINING data (faster); the full training set and the test set are then only transformed.

    Exposed separately so tests can prove that ``X_test`` never influences the fitted objects.
    """
    if method == "Baseline":
        scaler = fit_scaler(X_train)
        return scaler.transform(X_train), scaler.transform(X_test)
    if method == "Truncated SVD":
        # TruncatedSVD does not centre the data; raw features are used by design (see Theory page).
        svd = TruncatedSVD(n_components=int(n_components), random_state=seed).fit(X_train)
        return svd.transform(X_train), svd.transform(X_test)
    scaler = fit_scaler(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    if method == "PCA":
        pca = PCA(n_components=int(n_components), random_state=seed).fit(Xtr)
        return pca.transform(Xtr), pca.transform(Xte)
    if method == "UMAP":
        p = umap_params or {}
        reducer = make_umap(n_components, p.get("n_neighbors", 15), p.get("min_dist", 0.1), seed)
        subset = int(p.get("fit_subset", 0) or 0)
        if subset and y_train is not None and subset < Xtr.shape[0]:
            Xfit, _, _ = stratified_sample(Xtr, y_train, subset, seed)
            reducer.fit(Xfit)
            return reducer.transform(Xtr), reducer.transform(Xte)
        Z_tr = reducer.fit_transform(Xtr)
        return Z_tr, reducer.transform(Xte)
    raise ParameterError(f"Unknown method '{method}'.")


def run_method(
    method: str,
    data: HARData,
    n_components: int = 20,
    seed: int = 42,
    umap_params: dict | None = None,
) -> ClassificationResult:
    """Run one pipeline end-to-end on the predefined split and score it on the test set."""
    n_train, n_features = data.X_train.shape
    validate_comparison_params(method, n_components, umap_params or {}, n_train, n_features)

    t0 = time.perf_counter()
    Z_tr, Z_te = fit_transform_split(
        method, data.X_train, data.X_test, n_components, seed, umap_params, y_train=data.y_train
    )
    reduction_time = time.perf_counter() - t0

    clf = make_classifier(seed)
    t1 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(Z_tr, data.y_train)
        y_pred = clf.predict(Z_te)
    clf_time = time.perf_counter() - t1
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    m = compute_metrics(data.y_test, y_pred)
    settings: dict[str, Any] = {"seed": int(seed)}
    if method != "Baseline":
        settings["n_components"] = int(n_components)
    if method == "UMAP":
        settings.update(
            n_neighbors=int((umap_params or {}).get("n_neighbors", 15)),
            min_dist=float((umap_params or {}).get("min_dist", 0.1)),
            umap_fit_samples=int((umap_params or {}).get("fit_subset", 0) or 0) or "all train",
        )
    return ClassificationResult(
        method=method,
        n_dims=int(Z_tr.shape[1]),
        is_original=(method == "Baseline"),
        reduction_time_s=None if method == "Baseline" else float(reduction_time),
        classifier_time_s=float(clf_time),
        converged=converged,
        settings=settings,
        n_train=int(Z_tr.shape[0]),
        n_test=int(Z_te.shape[0]),
        **m,
    )


def run_comparison(
    data: HARData,
    n_components: int,
    seed: int,
    umap_params: dict,
    methods: tuple[str, ...] = METHODS,
    progress: Callable[[str], None] | None = None,
) -> list[ClassificationResult]:
    out = []
    for m in methods:
        if progress:
            progress(m)
        out.append(run_method(m, data, n_components, seed, umap_params))
    return out
