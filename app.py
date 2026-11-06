"""Dimensionality Reduction Explorer - Streamlit entry point.

Run locally:  streamlit run app.py
This file only sets up the page, navigation and session state; algorithms live in ``src/``.
"""
import streamlit as st

from src.content import EXPERIMENT_TITLE
from src.state import clear_results, init_state

st.set_page_config(
    page_title=EXPERIMENT_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state(st)

pages = [
    st.Page("pages/1_Overview.py", title="Home / Overview", default=True),
    st.Page("pages/2_Aim.py", title="Aim"),
    st.Page("pages/3_Theory.py", title="Theory"),
    st.Page("pages/4_Pre_test.py", title="Pre-test"),
    st.Page("pages/5_Simulation.py", title="Simulation"),
    st.Page("pages/6_Procedure.py", title="Procedure"),
    st.Page("pages/7_Observations.py", title="Observations & Results"),
    st.Page("pages/8_Exercises.py", title="Exercises"),
    st.Page("pages/9_Post_test.py", title="Post-test"),
    st.Page("pages/10_References.py", title="References"),
    st.Page("pages/11_Contributors.py", title="Contributors"),
    st.Page("pages/12_Feedback.py", title="Feedback"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    n_runs = len(st.session_state["sim_history"]) + len(st.session_state["cmp_history"])
    st.caption(f"Runs this session: {n_runs}")
    if st.button("Clear session results", help="Removes stored simulation and comparison results (cached computations stay cached)."):
        clear_results(st)
        st.rerun()

nav.run()
