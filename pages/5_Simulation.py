import traceback

import numpy as np
import streamlit as st

from src.data_loader import ACTIVITY_NAMES
from src.dimensionality_reduction import (
    ALGORITHMS,
    DIM_OPTIONS,
    TSNE_MAX_ITER,
    TSNE_MAX_SAMPLES,
    TSNE_MIN_ITER,
    ParameterError,
    validate_params,
    validate_trust_k,
)
from src.evaluation import METHOD_LABELS, validate_comparison_params
from src.interpretation import comparison_notes, embedding_notes
from src.report import comparison_label, comparison_rows, embedding_row, embedding_run_key
from src.state import reset_comparison_controls, reset_embedding_controls
from src.ui import cached_embedding, cached_method, get_data
from src.visualization import (
    class_distribution_figure,
    confusion_figure,
    embedding_figure,
    explained_variance_figure,
    full_cumulative_figure,
    metric_comparison_figure,
    singular_values_figure,
)

st.title("Simulation")
data = get_data()
n_train = data.X_train.shape[0]

# ------------------------------------------------------------------ dataset information
with st.expander("Dataset information - UCI HAR (predefined train/test split)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train samples", f"{data.X_train.shape[0]:,}")
    c2.metric("Test samples", f"{data.X_test.shape[0]:,}")
    c3.metric("Features", data.n_features)
    c4.metric("Activities", len(data.classes))
    dist = data.class_distribution()
    left, right = st.columns([2, 3])
    left.dataframe(dist, hide_index=True, width="stretch")
    right.plotly_chart(class_distribution_figure(dist), width="stretch")
    st.caption(
        "Labels are used for colouring and for the classifier only - PCA, SVD, t-SNE and UMAP never see them. "
        "Dataset source and citation: see the References page."
    )

tab_embed, tab_cmp = st.tabs(["1 · Embedding & visualization", "2 · Classification comparison"])


def show_error(exc: Exception) -> None:
    if isinstance(exc, ParameterError):
        st.error(f"Invalid setting: {exc}")
    else:
        st.error("Something went wrong while running this experiment. Try smaller settings or reset to defaults.")
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(exc)))


# ================================================================== TAB 1 - embedding
with tab_embed:
    st.markdown(
        "Choose an algorithm, adjust its parameters and press **Run algorithm**. Nothing is computed until you press the button."
    )
    st.selectbox("Algorithm", ALGORITHMS, key="sim_algo")
    algo = st.session_state["sim_algo"]

    max_sample = min(TSNE_MAX_SAMPLES if algo == "t-SNE" else 3000, n_train)
    min_sample = min(200, max_sample)
    st.session_state["sim_sample"] = int(np.clip(st.session_state["sim_sample"], min_sample, max_sample))
    with st.form("embedding_form"):
        st.caption("Dataset: HAR (fixed). Visualisation uses a reproducible, class-balanced subset of the **training** split.")
        c1, c2, c3 = st.columns(3)
        n_sample = c1.slider(
            "Sample size", min_sample, max_sample, step=50, key="sim_sample", help="Number of training samples to embed. t-SNE is capped for runtime.",
        )
        seed = c2.number_input("Random seed", 0, 99999, key="sim_seed", step=1)
        trust_k = c3.slider("Trustworthiness k", 1, 30, key="sim_trust_k", help="Neighbourhood size used by the trustworthiness score.")

        params: dict = {}
        if algo == "PCA":
            params["n_components"] = st.select_slider("Reduced dimensions (components)", DIM_OPTIONS, key="sim_pca_n")
        elif algo == "Truncated SVD":
            params["n_components"] = st.select_slider("Reduced dimensions (components)", DIM_OPTIONS, key="sim_svd_n")
        elif algo == "t-SNE":
            d1, d2, d3 = st.columns(3)
            params["perplexity"] = d1.slider("Perplexity", 5, 50, key="sim_tsne_perp", help="Effective neighbourhood size; must be < number of samples.")
            params["learning_rate"] = d2.slider("Learning rate", 10, 1000, key="sim_tsne_lr", step=10)
            params["max_iter"] = d3.slider("Iterations", TSNE_MIN_ITER, TSNE_MAX_ITER, key="sim_tsne_iter", step=50, help="More iterations = slower. Minimum 250 (scikit-learn).")
            st.caption("t-SNE output is always 2-dimensional in this lab.")
        else:
            d1, d2, d3 = st.columns(3)
            params["n_components"] = d1.select_slider("Reduced dimensions", DIM_OPTIONS, key="sim_umap_n")
            params["n_neighbors"] = d2.slider("n_neighbors", 2, 100, key="sim_umap_nn", help="Local neighbourhood size.")
            params["min_dist"] = d3.slider("min_dist", 0.0, 1.0, key="sim_umap_md", step=0.05, help="How tightly points may be packed.")

        submitted = st.form_submit_button("Run algorithm", type="primary")

    st.button("Reset to defaults", on_click=reset_embedding_controls, args=(st,), key="reset_embed")

    if submitted:
        try:
            validate_params(algo, params, n_sample, data.n_features)
            validate_trust_k(int(trust_k), n_sample)
            with st.spinner(f"Running {algo} on {n_sample:,} samples... (t-SNE and UMAP can take a while on shared hosting)"):
                res = cached_embedding(algo, int(n_sample), int(seed), tuple(sorted(params.items())), int(trust_k))
            st.session_state["sim_result"] = res
            st.session_state["sim_history"][embedding_run_key(res)] = embedding_row(res)
        except Exception as exc:  # shown to the student, never crashes the app
            show_error(exc)

    res = st.session_state.get("sim_result")
    if res is not None:
        st.divider()
        st.subheader(f"Result: {res.algorithm}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original → reduced", f"{res.n_original} → {res.n_reduced}")
        m2.metric("Samples", f"{res.n_samples:,}")
        m3.metric(f"Trustworthiness (k={res.trust_k})", "N/A" if res.trustworthiness is None else f"{res.trustworthiness:.4f}")
        m4.metric("Runtime", f"{res.runtime_s:.2f} s")
        st.caption(f"Parameters: {', '.join(f'{k}={v}' for k, v in res.params.items())} · seed={res.seed}. "
                   "Identical settings are cached; the runtime shown is from the original computation.")

        prefix = {"PCA": "PC", "Truncated SVD": "SVD", "t-SNE": "t-SNE", "UMAP": "UMAP"}[res.algorithm]
        title = f"{res.algorithm} embedding ({res.n_reduced}D" + (", first 2 dims shown)" if res.n_reduced > 2 else ")")
        st.plotly_chart(embedding_figure(res.embedding, res.labels, title, prefix), width="stretch")
        if res.algorithm in ("t-SNE", "UMAP"):
            st.caption("Axes of t-SNE/UMAP have no direct physical meaning, and distances between clusters should not be over-interpreted.")

        if res.algorithm == "PCA":
            evr, cum = res.extras["explained_variance_ratio"], res.extras["cumulative_explained_variance"]
            a, b = st.columns(2)
            a.plotly_chart(explained_variance_figure(evr, cum), width="stretch")
            b.plotly_chart(full_cumulative_figure(res.extras["full_cumulative"], res.n_reduced), width="stretch")
        elif res.algorithm == "Truncated SVD":
            st.plotly_chart(singular_values_figure(res.extras["singular_values"]), width="stretch")
            with st.expander("Explained variance of Truncated SVD (scikit-learn definition)"):
                evr = res.extras["explained_variance_ratio"]
                st.dataframe({"Component": np.arange(1, len(evr) + 1), "Explained variance ratio": evr.round(5)}, hide_index=True)
                st.caption("Computed by scikit-learn as the variance of each projected component divided by the total column variance of the (uncentred) input.")
        elif res.algorithm == "t-SNE":
            st.caption(f"Final KL divergence: {res.extras['kl_divergence']:.4f} · optimiser iterations: {res.extras['n_iter']}.")

        with st.container(border=True):
            st.markdown("**Interpretation notes** (generated from this run)")
            for n in embedding_notes(res):
                st.markdown(f"- {n}")

# ================================================================== TAB 2 - classification
with tab_cmp:
    st.markdown(
        "Compares **Logistic Regression** trained on the original 561 features with the same model trained on **PCA**, "
        "**Truncated SVD** and **UMAP** representations, using the predefined train/test split. "
        "Every scaler and reducer is fitted on the **training** data only; the test data are only transformed."
    )
    st.info("t-SNE is not in this table: standard scikit-learn t-SNE has no out-of-sample `transform()`, so it is evaluated on the Embedding tab (visualisation, trustworthiness, runtime).")

    with st.form("comparison_form"):
        c1, c2 = st.columns(2)
        dims = c1.select_slider("Reduced dimensions (PCA, SVD, UMAP)", DIM_OPTIONS, key="cmp_dims")
        cseed = c2.number_input("Random seed", 0, 99999, key="cmp_seed", step=1)
        u1, u2 = st.columns(2)
        unn = u1.slider("UMAP n_neighbors", 2, 100, key="cmp_umap_nn")
        umd = u2.slider("UMAP min_dist", 0.0, 1.0, key="cmp_umap_md", step=0.05)
        fit_opt = st.selectbox(
            "UMAP fitting set (training data only)",
            [0, 2000, 3000, 5000],
            format_func=lambda v: "Full training set (slowest)" if v == 0 else f"Stratified subset of {v:,} training samples",
            key="cmp_umap_fit",
            help="UMAP is the slowest step. Fitting on a training subset is still leak-free: the full training set and the test set are only transformed.",
        )
        run_cmp = st.form_submit_button("Run comparison", type="primary")
    st.button("Reset to defaults", on_click=reset_comparison_controls, args=(st,), key="reset_cmp")

    if run_cmp:
        umap_params = {"n_neighbors": int(unn), "min_dist": float(umd), "fit_subset": int(fit_opt)}
        try:
            for m in ("PCA", "Truncated SVD", "UMAP"):
                validate_comparison_params(m, int(dims), umap_params, n_train, data.n_features)
            results = []
            with st.status("Running comparison...", expanded=True) as status:
                for m in ("Baseline", "PCA", "Truncated SVD", "UMAP"):
                    status.update(label=f"Running {METHOD_LABELS[m]} ...")
                    st.write(METHOD_LABELS[m])
                    results.append(
                        cached_method(m, 0 if m == "Baseline" else int(dims), int(cseed), tuple(sorted(umap_params.items())))
                    )
                status.update(label="Comparison complete", state="complete", expanded=False)
            st.session_state["cmp_results"] = results
            st.session_state["cmp_history"][comparison_label(results)] = results
        except Exception as exc:
            show_error(exc)

    results = st.session_state.get("cmp_results")
    if results:
        st.divider()
        rows = comparison_rows(results)
        st.subheader("Results on the HAR test set")
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            f"Train n={results[0].n_train:,} · Test n={results[0].n_test:,} (identical test samples for every model) · "
            "Logistic Regression: C=1.0, lbfgs, max_iter=1000 for every model. "
            "Reduction time = fitting the scaler/reducer on train + transforming train and test; classifier time = fit + predict."
        )
        if any(not r.converged for r in results):
            st.warning("Logistic Regression did not fully converge for at least one model (see the 'Converged' column).")
        st.plotly_chart(metric_comparison_figure(rows), width="stretch")

        choice = st.selectbox("Confusion matrix for", [r.label for r in results], key="cmp_cm_choice")
        sel = next(r for r in results if r.label == choice)
        st.plotly_chart(confusion_figure(sel.confusion, f"Confusion matrix - {sel.label}"), width="stretch")

        with st.container(border=True):
            st.markdown("**Interpretation notes** (generated from these results)")
            for n in comparison_notes(results):
                st.markdown(f"- {n}")
        with st.expander("Leakage-prevention checklist"):
            st.markdown(
                "- Predefined train/test split kept fixed.\n- `StandardScaler` fitted on train only.\n"
                "- PCA / Truncated SVD / UMAP fitted on train only; test data only transformed.\n"
                "- Classifier fitted on train representations; evaluated on test representations.\n"
                "- No parameter was tuned using test labels."
            )
