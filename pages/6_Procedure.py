import streamlit as st

from src.content import PROCEDURE

st.title("Procedure")
st.caption("Follow these steps in order.")
for i, (title, detail) in enumerate(PROCEDURE, 1):
    st.markdown(f"**{i}. {title}**")
    st.markdown(f"<div style='margin:-0.4rem 0 0.8rem 1.4rem;opacity:.85'>{detail}</div>", unsafe_allow_html=True)
st.divider()
c1, c2 = st.columns(2)
c1.page_link("pages/5_Simulation.py", label="Open the Simulation")
c2.page_link("pages/7_Observations.py", label="Observations & Results")
