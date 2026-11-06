"""Streamlit-facing helpers: cached data access and cached computations (TRD section 10)."""
from __future__ import annotations

import streamlit as st

from .data_loader import DatasetError, HARData, load_har
from .dimensionality_reduction import EmbeddingResult, run_embedding
from .evaluation import ClassificationResult, run_method
from .preprocessing import stratified_sample


@st.cache_resource(show_spinner="Loading the HAR dataset (the first run may download it from UCI)...")
def _load_dataset() -> HARData:
    # cache_resource is safe here: the arrays are marked read-only, so sessions cannot corrupt them.
    return load_har()


def get_data() -> HARData:
    """Return the dataset or show an actionable error and stop the page."""
    try:
        return _load_dataset()
    except DatasetError as exc:
        st.error("The HAR dataset could not be loaded.")
        st.markdown(str(exc).replace("\n", "  \n"))
        st.info("See `data/README.md` in the repository for manual setup options.")
        st.stop()


@st.cache_data(show_spinner=False, max_entries=48)
def cached_embedding(algorithm: str, n_sample: int, seed: int, params_items: tuple, trust_k: int) -> EmbeddingResult:
    """Cache key = algorithm + every parameter + seed, so stale results cannot be returned."""
    data = _load_dataset()
    X, y, _ = stratified_sample(data.X_train, data.y_train, n_sample, seed)
    return run_embedding(algorithm, X, y, dict(params_items), seed=seed, trust_k=trust_k)


@st.cache_data(show_spinner=False, max_entries=48)
def cached_method(method: str, n_components: int, seed: int, umap_items: tuple) -> ClassificationResult:
    return run_method(method, _load_dataset(), n_components, seed, dict(umap_items))


def page_intro(title: str, caption: str | None = None) -> None:
    st.title(title)
    if caption:
        st.caption(caption)
