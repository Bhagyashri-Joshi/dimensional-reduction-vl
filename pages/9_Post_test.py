import streamlit as st

from src.content import POST_TEST
from src.quiz import render_quiz

st.title("Post-test")
st.caption("10 applied questions based on what you observed in the simulation. Scores are kept only for this browser session.")
pre = st.session_state.get("pre_result")
if pre:
    st.caption(f"Your pre-test score was {pre['score']} / {pre['total']}.")
render_quiz(st, "post", POST_TEST)
