import streamlit as st

from src.content import CONTRIBUTORS, FACULTY_GUIDE

st.title("Contributors")
if not CONTRIBUTORS and not FACULTY_GUIDE:
    st.warning(
        "Contributor details have not been added yet. The project owner should fill in `CONTRIBUTORS` and "
        "`FACULTY_GUIDE` in `src/content.py` (names, roles and affiliations are not invented by the app)."
    )
for c in CONTRIBUTORS:
    st.markdown(f"### {c.get('name', '')}")
    for key in ("role", "affiliation"):
        if c.get(key):
            st.markdown(f"**{key.capitalize()}:** {c[key]}")
if FACULTY_GUIDE:
    st.subheader("Faculty guide")
    st.markdown(f"**{FACULTY_GUIDE.get('name', '')}**")
    for key, value in FACULTY_GUIDE.items():
        if key != "name" and value:
            st.markdown(f"{key.capitalize()}: {value}")
st.caption("Built with Streamlit, scikit-learn, umap-learn and Plotly. Dataset: UCI HAR (see References).")
