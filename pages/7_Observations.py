import streamlit as st

from src.interpretation import comparison_notes, embedding_notes
from src.report import (
    comparison_history_table,
    embedding_table,
    markdown_report,
    to_csv,
)
from src.visualization import (
    confusion_figure,
    embedding_figure,
    explained_variance_figure,
    metric_comparison_figure,
    singular_values_figure,
)

st.title("Observations & Results")
st.caption("Everything on this page comes from runs you performed in this session. Nothing is pre-filled.")

sim_hist = st.session_state["sim_history"]
cmp_hist = st.session_state["cmp_history"]
res = st.session_state.get("sim_result")
cmp_res = st.session_state.get("cmp_results")

if res is None and not cmp_res:
    st.info("No results yet. Run an algorithm and/or the classification comparison on the Simulation page first.")
    st.page_link("pages/5_Simulation.py", label="Go to the Simulation")

# ------------------------------------------------------------------ latest embedding
if res is not None:
    st.header("Latest embedding run")
    st.markdown(f"**Algorithm:** {res.algorithm}  \n**Parameters:** {', '.join(f'{k}={v}' for k, v in res.params.items())} · seed={res.seed} · samples={res.n_samples:,}")
    a, b = st.columns([3, 2])
    prefix = {"PCA": "PC", "Truncated SVD": "SVD", "t-SNE": "t-SNE", "UMAP": "UMAP"}[res.algorithm]
    a.plotly_chart(embedding_figure(res.embedding, res.labels, f"{res.algorithm} ({res.n_reduced}D)", prefix), width="stretch")
    with b:
        if res.algorithm == "PCA":
            st.plotly_chart(explained_variance_figure(res.extras["explained_variance_ratio"], res.extras["cumulative_explained_variance"]), width="stretch")
        elif res.algorithm == "Truncated SVD":
            st.plotly_chart(singular_values_figure(res.extras["singular_values"]), width="stretch")
        st.metric("Trustworthiness", "N/A" if res.trustworthiness is None else f"{res.trustworthiness:.4f}")
        st.metric("Runtime", f"{res.runtime_s:.2f} s")
    for n in embedding_notes(res):
        st.markdown(f"- {n}")

# ------------------------------------------------------------------ tables
st.header("Embedding runs")
st.caption("Each distinct setting you ran is listed once. t-SNE is kept in its own table because it is evaluated as a visualisation method only.")
t1 = embedding_table(sim_hist, tsne=False)
t2 = embedding_table(sim_hist, tsne=True)
st.subheader("PCA / Truncated SVD / UMAP")
if t1.empty:
    st.write("_No runs yet._")
else:
    st.dataframe(t1, hide_index=True, width="stretch")
    st.download_button("Download CSV", to_csv(t1), "embedding_runs_pca_svd_umap.csv", "text/csv", key="dl_t1")
st.subheader("t-SNE visualisation analysis")
if t2.empty:
    st.write("_No t-SNE runs yet._")
else:
    st.dataframe(t2, hide_index=True, width="stretch")
    st.download_button("Download CSV", to_csv(t2), "embedding_runs_tsne.csv", "text/csv", key="dl_t2")

st.header("Classification comparison runs")
cdf = comparison_history_table(cmp_hist)
if cdf.empty:
    st.write("_No comparison runs yet._")
else:
    st.dataframe(cdf, hide_index=True, width="stretch")
    st.download_button("Download CSV", to_csv(cdf), "classification_comparison.csv", "text/csv", key="dl_cmp")
    if cmp_res:
        from src.report import comparison_rows

        st.plotly_chart(metric_comparison_figure(comparison_rows(cmp_res)), width="stretch")
        pick = st.selectbox("Confusion matrix for (latest run)", [r.label for r in cmp_res], key="obs_cm")
        sel = next(r for r in cmp_res if r.label == pick)
        st.plotly_chart(confusion_figure(sel.confusion, f"Confusion matrix - {sel.label}"), width="stretch")
        for n in comparison_notes(cmp_res):
            st.markdown(f"- {n}")

# ------------------------------------------------------------------ guidance + report
st.header("How to interpret your results")
st.markdown(
    """
- **Explained variance / singular values** describe how much of the data's variance a *linear* method keeps. They do not apply to t-SNE or UMAP.
- **Trustworthiness** (0-1) describes how well *local neighbourhoods* survive. Compare values only with the same k and sample size.
- **Classification metrics** describe what a classifier achieves on the fixed test set. A better plot does not guarantee a better classifier.
- **Runtime** depends on hosting hardware and on caching; compare runs relative to each other, not as fixed promises.
- Write your own conclusions: which setting gave the best trade-off between dimensions, trustworthiness, accuracy and runtime - and how sure are you?
"""
)
st.text_area("My observations (kept only in this session)", key="my_observations", height=160)

if sim_hist or cmp_hist:
    st.download_button(
        "Download experiment report (Markdown)",
        markdown_report(sim_hist, cmp_hist) + "\n\n## My observations\n" + (st.session_state.get("my_observations") or "_none_"),
        "experiment_report.md",
        "text/markdown",
        key="dl_report",
    )
