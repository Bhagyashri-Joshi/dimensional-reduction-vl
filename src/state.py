"""Session-state defaults and reset behaviour (TRD section 10)."""
from __future__ import annotations

SEED_DEFAULT = 42

WIDGET_DEFAULTS = {
    "sim_algo": "PCA",
    "sim_sample": 2000,
    "sim_seed": SEED_DEFAULT,
    "sim_trust_k": 5,
    "sim_pca_n": 2,
    "sim_svd_n": 2,
    "sim_tsne_perp": 30,
    "sim_tsne_lr": 200,
    "sim_tsne_iter": 500,
    "sim_umap_n": 2,
    "sim_umap_nn": 15,
    "sim_umap_md": 0.1,
    "cmp_dims": 20,
    "cmp_seed": SEED_DEFAULT,
    "cmp_umap_nn": 15,
    "cmp_umap_md": 0.1,
    "cmp_umap_fit": 3000,
}

RESULT_KEYS = {
    "sim_result": None,  # latest EmbeddingResult
    "sim_history": {},  # run-key -> row dict (embedding runs)
    "cmp_results": None,  # latest list[ClassificationResult]
    "cmp_history": {},  # run-key -> list[ClassificationResult]
}


def init_state(st) -> None:
    for k, v in {**WIDGET_DEFAULTS, **RESULT_KEYS}.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, dict) else v


def reset_embedding_controls(st) -> None:
    for k, v in WIDGET_DEFAULTS.items():
        if k.startswith("sim_"):
            st.session_state[k] = v


def reset_comparison_controls(st) -> None:
    for k, v in WIDGET_DEFAULTS.items():
        if k.startswith("cmp_"):
            st.session_state[k] = v


def clear_results(st) -> None:
    for k, v in RESULT_KEYS.items():
        st.session_state[k] = v.copy() if isinstance(v, dict) else v
