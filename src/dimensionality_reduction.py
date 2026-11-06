"""PCA, Truncated SVD, t-SNE and UMAP wrappers with validation (TRD section 5, 7).

Everything here is pure Python/NumPy (no Streamlit) so it can be unit-tested.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

ALGORITHMS = ("PCA", "Truncated SVD", "t-SNE", "UMAP")
DIM_OPTIONS = (2, 5, 10, 20, 50)

TSNE_MAX_SAMPLES = 3000
TSNE_MIN_ITER, TSNE_MAX_ITER = 250, 1000  # scikit-learn requires max_iter >= 250
TSNE_PERPLEXITY_RANGE = (5, 50)

DEFAULTS: dict[str, dict[str, Any]] = {
    "PCA": {"n_components": 2},
    "Truncated SVD": {"n_components": 2},
    "t-SNE": {"perplexity": 30, "learning_rate": 200, "max_iter": 500},
    "UMAP": {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1},
}


class ParameterError(ValueError):
    """Invalid user-supplied parameters; the message is shown directly to students."""


@dataclass
class EmbeddingResult:
    algorithm: str
    params: dict[str, Any]
    seed: int
    embedding: np.ndarray
    labels: np.ndarray
    n_samples: int
    n_original: int
    n_reduced: int
    runtime_s: float
    trustworthiness: float | None
    trust_k: int
    trust_note: str = ""
    trust_runtime_s: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- validation
def validate_trust_k(k: int, n_samples: int) -> None:
    if k < 1 or k >= n_samples / 2:
        raise ParameterError(
            f"Trustworthiness neighbourhood size k={k} is invalid: it must satisfy 1 <= k < n_samples/2 "
            f"(n_samples={n_samples}). Choose k below {int(np.ceil(n_samples / 2)) - 1}."
        )


def validate_params(algorithm: str, params: dict[str, Any], n_samples: int, n_features: int) -> None:
    """Raise :class:`ParameterError` with an actionable message for any invalid combination."""
    if algorithm not in ALGORITHMS:
        raise ParameterError(f"Unknown algorithm '{algorithm}'.")
    if n_samples < 10:
        raise ParameterError(f"Need at least 10 samples (got {n_samples}). Increase the sample size.")

    if algorithm == "PCA":
        n = int(params["n_components"])
        limit = min(n_samples, n_features)
        if not 1 <= n <= limit:
            raise ParameterError(f"PCA components must be between 1 and {limit} (got {n}).")
    elif algorithm == "Truncated SVD":
        n = int(params["n_components"])
        if not 1 <= n < n_features:
            raise ParameterError(
                f"Truncated SVD components must be between 1 and {n_features - 1} (got {n})."
            )
    elif algorithm == "t-SNE":
        if n_samples > TSNE_MAX_SAMPLES:
            raise ParameterError(
                f"t-SNE is limited to {TSNE_MAX_SAMPLES} samples to keep runtime reasonable (got {n_samples}). "
                "Reduce the sample size."
            )
        p = float(params["perplexity"])
        if p <= 0 or p >= n_samples:
            raise ParameterError(
                f"Perplexity must be positive and smaller than the number of samples ({n_samples}); got {p:g}."
            )
        if not TSNE_MIN_ITER <= int(params["max_iter"]) <= TSNE_MAX_ITER:
            raise ParameterError(f"t-SNE iterations must be between {TSNE_MIN_ITER} and {TSNE_MAX_ITER}.")
        if float(params["learning_rate"]) <= 0:
            raise ParameterError("t-SNE learning rate must be positive.")
    elif algorithm == "UMAP":
        nn, md, n = int(params["n_neighbors"]), float(params["min_dist"]), int(params["n_components"])
        if not 2 <= nn < n_samples:
            raise ParameterError(
                f"UMAP n_neighbors must be between 2 and {n_samples - 1} for {n_samples} samples (got {nn})."
            )
        if not 0.0 <= md <= 1.0:
            raise ParameterError(f"UMAP min_dist must be between 0.0 and 1.0 (got {md:g}).")
        if not 1 <= n < n_samples - 1:
            raise ParameterError(f"UMAP components must be between 1 and {n_samples - 2} (got {n}).")


# --------------------------------------------------------------------------- reducer factory
def make_umap(n_components: int, n_neighbors: int, min_dist: float, seed: int):
    import umap  # imported lazily: slow import, optional for unit tests of other modules

    return umap.UMAP(
        n_components=int(n_components),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        random_state=int(seed),
        n_jobs=1,  # explicit: a fixed seed forces single-threaded execution anyway
    )


def compute_trustworthiness(X_ref: np.ndarray, embedding: np.ndarray, k: int) -> tuple[float | None, str, float | None]:
    """Return (score, note, runtime). Score is None with an explanatory note if constraints fail."""
    n = X_ref.shape[0]
    try:
        validate_trust_k(k, n)
    except ParameterError as exc:
        return None, str(exc), None
    t0 = time.perf_counter()
    score = float(trustworthiness(X_ref, embedding, n_neighbors=int(k)))
    return score, "", time.perf_counter() - t0


# --------------------------------------------------------------------------- main entry point
def run_embedding(
    algorithm: str,
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    seed: int = 42,
    trust_k: int = 5,
) -> EmbeddingResult:
    """Fit one algorithm on a (sampled) feature matrix for exploratory visualisation.

    PCA, t-SNE and UMAP receive standardised features (scaler fitted on the sample). Truncated SVD
    receives the raw features because it does not centre the data (this is the point of comparing it
    with PCA). This function is for *visualisation*; classification uses ``evaluation.py`` with a
    strict train/test separation.
    """
    n_samples, n_features = X.shape
    validate_params(algorithm, params, n_samples, n_features)
    validate_trust_k(int(trust_k), n_samples)
    extras: dict[str, Any] = {}

    if algorithm == "Truncated SVD":
        X_in = X
        t0 = time.perf_counter()
        model = TruncatedSVD(n_components=int(params["n_components"]), random_state=seed)
        emb = model.fit_transform(X_in)
        runtime = time.perf_counter() - t0
        extras["singular_values"] = model.singular_values_.copy()
        extras["explained_variance_ratio"] = model.explained_variance_ratio_.copy()
        extras["input_note"] = "Raw (uncentred) features were used."
    else:
        X_in = StandardScaler().fit_transform(X)
        if algorithm == "PCA":
            t0 = time.perf_counter()
            model = PCA(n_components=int(params["n_components"]), random_state=seed)
            emb = model.fit_transform(X_in)
            runtime = time.perf_counter() - t0
            evr = model.explained_variance_ratio_.copy()
            extras["explained_variance_ratio"] = evr
            extras["cumulative_explained_variance"] = np.cumsum(evr)
            full = PCA(n_components=min(n_samples, n_features), svd_solver="full").fit(X_in)
            extras["full_cumulative"] = np.cumsum(full.explained_variance_ratio_)
        elif algorithm == "t-SNE":
            t0 = time.perf_counter()
            model = TSNE(
                n_components=2,
                perplexity=float(params["perplexity"]),
                learning_rate=float(params["learning_rate"]),
                max_iter=int(params["max_iter"]),
                init="pca",
                random_state=seed,
            )
            emb = model.fit_transform(X_in)
            runtime = time.perf_counter() - t0
            extras["kl_divergence"] = float(model.kl_divergence_)
            extras["n_iter"] = int(model.n_iter_)
        else:  # UMAP
            t0 = time.perf_counter()
            model = make_umap(params["n_components"], params["n_neighbors"], params["min_dist"], seed)
            emb = model.fit_transform(X_in)
            runtime = time.perf_counter() - t0

    emb = np.asarray(emb)
    score, note, t_trust = compute_trustworthiness(X_in, emb, int(trust_k))
    return EmbeddingResult(
        algorithm=algorithm,
        params=dict(params),
        seed=int(seed),
        embedding=emb,
        labels=np.asarray(y),
        n_samples=n_samples,
        n_original=n_features,
        n_reduced=int(emb.shape[1]),
        runtime_s=float(runtime),
        trustworthiness=score,
        trust_k=int(trust_k),
        trust_note=note,
        trust_runtime_s=t_trust,
        extras=extras,
    )


def components_for_variance(cumulative: np.ndarray, threshold: float) -> int | None:
    """Smallest number of components whose cumulative explained variance >= threshold."""
    hits = np.flatnonzero(np.asarray(cumulative) >= threshold)
    return int(hits[0]) + 1 if hits.size else None
