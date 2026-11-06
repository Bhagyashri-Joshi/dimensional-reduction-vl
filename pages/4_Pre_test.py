import streamlit as st

from src.content import PRE_TEST
from src.quiz import render_quiz

st.title("Pre-test")
st.caption("10 multiple-choice questions - no account needed. Scores are kept only for this browser session.")
render_quiz(st, "pre", PRE_TEST)
