import numpy as np
import pytest

from src.dimensionality_reduction import (
    ParameterError,
    compute_trustworthiness,
    run_embedding,
    validate_params,
)
from src.preprocessing import stratified_sample


@pytest.fixture(scope="module")
def sample(har):
    X, y, _ = stratified_sample(har.X_train, har.y_train, 300, seed=1)
    return X, y


def test_pca_dims_and_variance(sample):
    X, y = sample
    r = run_embedding("PCA", X, y, {"n_components": 5}, seed=0)
    assert r.embedding.shape == (300, 5)
    evr = r.extras["explained_variance_ratio"]
    assert np.all(np.diff(evr) <= 1e-12) and 0 < evr.sum() <= 1
    assert np.allclose(r.extras["cumulative_explained_variance"], np.cumsum(evr))
    assert r.extras["full_cumulative"][-1] == pytest.approx(1.0, abs=1e-6)
    assert 0 <= r.trustworthiness <= 1


def test_svd_dims_and_singular_values(sample):
    X, y = sample
    r = run_embedding("Truncated SVD", X, y, {"n_components": 4}, seed=0)
    assert r.embedding.shape == (300, 4)
    sv = r.extras["singular_values"]
    assert len(sv) == 4 and np.all(np.diff(sv) <= 1e-9)


def test_pca_and_svd_differ_on_uncentred_data(sample):
    X, y = sample
    p = run_embedding("PCA", X, y, {"n_components": 2}, seed=0)
    s = run_embedding("Truncated SVD", X, y, {"n_components": 2}, seed=0)
    assert not np.allclose(np.abs(p.embedding), np.abs(s.embedding))


def test_tsne_reproducible_with_seed(sample):
    X, y = sample
    params = {"perplexity": 20, "learning_rate": 200, "max_iter": 250}
    a = run_embedding("t-SNE", X, y, params, seed=3)
    b = run_embedding("t-SNE", X, y, params, seed=3)
    assert a.embedding.shape == (300, 2)
    assert np.allclose(a.embedding, b.embedding, atol=1e-4)
    assert a.trustworthiness is not None


def test_umap_shape_and_reproducible(sample):
    X, y = sample
    params = {"n_components": 3, "n_neighbors": 10, "min_dist": 0.1}
    a = run_embedding("UMAP", X, y, params, seed=5)
    b = run_embedding("UMAP", X, y, params, seed=5)
    assert a.embedding.shape == (300, 3)
    assert np.allclose(a.embedding, b.embedding)


def test_tsne_perplexity_validation():
    base = {"learning_rate": 200, "max_iter": 500}
    with pytest.raises(ParameterError, match="Perplexity"):
        validate_params("t-SNE", {"perplexity": 100, **base}, n_samples=100, n_features=561)
    with pytest.raises(ParameterError, match="Perplexity"):
        validate_params("t-SNE", {"perplexity": 50, **base}, n_samples=50, n_features=561)
    validate_params("t-SNE", {"perplexity": 30, **base}, n_samples=100, n_features=561)


@pytest.mark.parametrize(
    "algo,params,msg",
    [
        ("PCA", {"n_components": 0}, "between 1"),
        ("PCA", {"n_components": 1000}, "between 1"),
        ("Truncated SVD", {"n_components": 561}, "between 1"),
        ("UMAP", {"n_components": 2, "n_neighbors": 500, "min_dist": 0.1}, "n_neighbors"),
        ("UMAP", {"n_components": 2, "n_neighbors": 15, "min_dist": 1.5}, "min_dist"),
    ],
)
def test_invalid_params_rejected(algo, params, msg):
    with pytest.raises(ParameterError, match=msg):
        validate_params(algo, params, n_samples=200, n_features=561)


def test_trustworthiness_valid_and_guarded(sample):
    X, y = sample
    emb = X[:, :2]
    score, note, _ = compute_trustworthiness(X, emb, 5)
    assert 0.0 <= score <= 1.0 and note == ""
    score, note, _ = compute_trustworthiness(X, emb, 200)
    assert score is None and "k=200" in note


def test_perfect_embedding_has_trustworthiness_one(sample):
    X, _ = sample
    score, _, _ = compute_trustworthiness(X, X.copy(), 5)
    assert score == pytest.approx(1.0)


def test_sampling_reproducible_and_stratified(har):
    a = stratified_sample(har.X_train, har.y_train, 120, seed=7)
    b = stratified_sample(har.X_train, har.y_train, 120, seed=7)
    assert np.array_equal(a[2], b[2])
    assert set(np.unique(a[1])) == set(range(1, 7))
