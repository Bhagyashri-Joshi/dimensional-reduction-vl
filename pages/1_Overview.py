import streamlit as st

from src.content import COURSE_OUTCOMES, EXPERIMENT_SUBTITLE, EXPERIMENT_TITLE, LEARNING_OUTCOMES, WORKFLOW

st.title(f" {EXPERIMENT_TITLE}")
st.subheader(EXPERIMENT_SUBTITLE)

st.markdown(
    """
High-dimensional data are hard to look at and expensive to model. In this virtual lab you take the
**UCI HAR** dataset - smartphone accelerometer and gyroscope recordings summarised by **561 engineered features**
for **6 everyday activities** - and squeeze it into far fewer dimensions with four classic techniques:
**PCA, Truncated SVD, t-SNE and UMAP**. You will *see* the result, *measure* how well neighbourhoods are preserved
and *test* what the reduction does to a classifier. No coding is required.
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Input features", "561")
c2.metric("Activities", "6")
c3.metric("Algorithms", "4")
c4.metric("Classifier", "Logistic Regression")

st.divider()
left, right = st.columns([3, 2])
with left:
    st.markdown("### Learning objectives")
    st.markdown("After this lab you should be able to:")
    for item in LEARNING_OUTCOMES:
        st.markdown(f"- {item}")
with right:
    st.markdown("### Course outcome mapping")
    labels = "  ".join(f"`{co}`" for co in COURSE_OUTCOMES)
    st.markdown(f"This experiment addresses: {labels}")
    for co, text in COURSE_OUTCOMES.items():
        if text:
            st.markdown(f"- **{co}** - {text}")
    st.markdown("### Experiment workflow")
    for i, (step, desc) in enumerate(WORKFLOW, 1):
        st.markdown(f"**{i}. {step}** - {desc}")

st.divider()
st.markdown("### Ready to begin?")
b1, b2, b3 = st.columns(3)
b1.page_link("pages/2_Aim.py", label="Start with the Aim")
b2.page_link("pages/4_Pre_test.py", label="Take the Pre-test")
b3.page_link("pages/5_Simulation.py", label="Jump to the Simulation")
