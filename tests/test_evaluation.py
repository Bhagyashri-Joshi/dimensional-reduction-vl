import numpy as np
import pytest

from src.evaluation import (
    compute_metrics,
    fit_transform_split,
    run_comparison,
    run_method,
)
from src.dimensionality_reduction import ParameterError


def test_metrics_on_known_sample():
    y_true = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    y_pred = np.array([1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 5])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(10 / 12)
    # per-class precision: 1,2/3,1,1,2/3(class5: tp2 fp1),1 -> macro
    assert m["precision_macro"] == pytest.approx((1 + 2 / 3 + 1 + 1 + 2 / 3 + 1) / 6)
    assert m["recall_macro"] == pytest.approx((0.5 + 1 + 1 + 1 + 1 + 0.5) / 6)
    assert m["confusion"].shape == (6, 6) and m["confusion"].sum() == 12
    assert m["confusion"][0, 1] == 1  # one WALKING predicted as WALKING_UPSTAIRS


@pytest.mark.parametrize("method", ["Baseline", "PCA", "Truncated SVD"])
def test_test_data_never_influences_fit(har, method):
    """Leakage guard: replacing X_test must not change the transformed TRAIN data."""
    rng = np.random.default_rng(0)
    junk = rng.normal(5, 3, size=har.X_test.shape)
    a_tr, _ = fit_transform_split(method, har.X_train, har.X_test, 5, seed=1)
    b_tr, _ = fit_transform_split(method, har.X_train, junk, 5, seed=1)
    assert np.allclose(a_tr, b_tr)


def test_umap_test_data_never_influences_fit(har):
    rng = np.random.default_rng(0)
    junk = rng.normal(5, 3, size=har.X_test.shape)
    p = {"n_neighbors": 10, "min_dist": 0.1}
    a_tr, _ = fit_transform_split("UMAP", har.X_train, har.X_test, 2, seed=1, umap_params=p)
    b_tr, _ = fit_transform_split("UMAP", har.X_train, junk, 2, seed=1, umap_params=p)
    assert np.allclose(a_tr, b_tr)


def test_comparison_runs_all_four_methods(har):
    seen = []
    res = run_comparison(har, 10, 42, {"n_neighbors": 10, "min_dist": 0.1}, progress=seen.append)
    assert [r.method for r in res] == ["Baseline", "PCA", "Truncated SVD", "UMAP"] == seen
    base = res[0]
    assert base.is_original and base.n_dims == 561 and base.reduction_time_s is None
    for r in res:
        assert 0 <= r.accuracy <= 1 and 0 <= r.f1_macro <= 1
        assert r.n_test == har.X_test.shape[0]  # same test samples everywhere
        assert r.confusion.sum() == har.X_test.shape[0]
    assert [r.n_dims for r in res[1:]] == [10, 10, 10]
    assert all(r.reduction_time_s > 0 for r in res[1:])


def test_comparison_reproducible(har):
    a = run_method("PCA", har, 5, 1)
    b = run_method("PCA", har, 5, 1)
    assert a.accuracy == b.accuracy and np.array_equal(a.confusion, b.confusion)


def test_invalid_comparison_params(har):
    with pytest.raises(ParameterError):
        run_method("PCA", har, 100000, 1)
    with pytest.raises(ParameterError):
        run_method("UMAP", har, 2, 1, {"n_neighbors": 99999, "min_dist": 0.1})


def test_umap_subset_fit_is_train_only(har):
    from src.evaluation import fit_transform_split

    rng = np.random.default_rng(0)
    junk = rng.normal(5, 3, size=har.X_test.shape)
    p = {"n_neighbors": 10, "min_dist": 0.1, "fit_subset": 200}
    a, _ = fit_transform_split("UMAP", har.X_train, har.X_test, 2, 1, p, y_train=har.y_train)
    b, _ = fit_transform_split("UMAP", har.X_train, junk, 2, 1, p, y_train=har.y_train)
    assert a.shape[0] == har.X_train.shape[0] and np.allclose(a, b)


def test_umap_subset_validation(har):
    with pytest.raises(ParameterError, match="fit subset"):
        run_method("UMAP", har, 2, 1, {"n_neighbors": 15, "min_dist": 0.1, "fit_subset": 5})
