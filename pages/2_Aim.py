import streamlit as st

from src.content import AIM, LEARNING_OUTCOMES

st.title("Aim")
st.info(AIM)
st.markdown("### What you will do")
st.markdown(
    """
1. **Apply** PCA, Truncated SVD, t-SNE and UMAP to 561-dimensional HAR data.
2. **Visualise** the reduced representations, colour-coded by activity.
3. **Analyse** information preserved (explained variance, singular values) and neighbourhood preservation (trustworthiness).
4. **Evaluate** how reduced feature sets change the performance of a Logistic Regression classifier, without data leakage.
"""
)
st.markdown("### Learning outcomes")
for item in LEARNING_OUTCOMES:
    st.markdown(f"- {item}")
st.page_link("pages/3_Theory.py", label="Continue to Theory")
