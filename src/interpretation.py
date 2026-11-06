"""Interpretation notes generated from the *actual* computed values (never hard-coded scores)."""
from __future__ import annotations

from .dimensionality_reduction import EmbeddingResult, components_for_variance
from .evaluation import ClassificationResult


def embedding_notes(r: EmbeddingResult) -> list[str]:
    notes: list[str] = []
    if r.algorithm == "PCA":
        cum = r.extras["cumulative_explained_variance"]
        notes.append(f"The {r.n_reduced} selected component(s) together explain **{cum[-1] * 100:.1f}%** of the total variance of the standardised features.")
        full = r.extras.get("full_cumulative")
        if full is not None:
            k90, k95 = components_for_variance(full, 0.90), components_for_variance(full, 0.95)
            notes.append(f"On this sample, **{k90}** components are needed for 90% and **{k95}** for 95% of the variance.")
    elif r.algorithm == "Truncated SVD":
        evr = r.extras["explained_variance_ratio"]
        notes.append(
            f"The selected components capture **{evr.sum() * 100:.1f}%** of the column variance (scikit-learn definition). "
            "Because the data are not centred, the first singular vector mostly reflects the mean of the features, "
            "so this differs from PCA."
        )
    elif r.algorithm == "t-SNE":
        notes.append("t-SNE keeps **local** neighbourhoods. Cluster sizes, gaps between clusters and axis values carry no reliable global meaning.")
    elif r.algorithm == "UMAP":
        notes.append("UMAP preserves local structure and often some global layout, but distances between clusters should still be read with caution.")
    if r.trustworthiness is not None:
        notes.append(
            f"Trustworthiness (k={r.trust_k}) = **{r.trustworthiness:.3f}** on a 0-1 scale: the closer to 1, the fewer false neighbours "
            "were introduced by the embedding. It measures neighbourhood preservation, not classification accuracy."
        )
    elif r.trust_note:
        notes.append(f"Trustworthiness was not computed: {r.trust_note}")
    if r.n_reduced > 2:
        notes.append(f"The plot shows only the first 2 of {r.n_reduced} dimensions; the remaining dimensions are not visible.")
    notes.append("A clean-looking 2D plot does not prove that a classifier will perform well, and an overlapping plot does not prove it will fail.")
    return notes


def comparison_notes(results: list[ClassificationResult]) -> list[str]:
    base = next((r for r in results if r.is_original), None)
    notes: list[str] = []
    if base is None:
        return notes
    for r in results:
        if r.is_original:
            continue
        diff = (r.accuracy - base.accuracy) * 100
        direction = "higher" if diff > 0 else "lower" if diff < 0 else "equal"
        notes.append(
            f"**{r.method}** ({r.n_dims} dims): accuracy {r.accuracy:.3f} vs baseline {base.accuracy:.3f} "
            f"({abs(diff):.2f} percentage points {direction}); the feature count dropped from {base.n_dims} to {r.n_dims}."
        )
    unconverged = [r.method for r in results if not r.converged]
    if unconverged:
        notes.append("Logistic Regression reported a convergence warning for: " + ", ".join(unconverged) + ". Treat those scores with care.")
    notes.append("Each score comes from a single run on the fixed HAR test set with one seed; small differences may not be meaningful.")
    return notes
