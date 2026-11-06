import streamlit as st

from src.content import EXERCISES

st.title("Exercises")
st.caption("Guided tasks. Do each one on the Simulation page, then write your answers (kept only in this session).")
st.page_link("pages/5_Simulation.py", label="Open the Simulation")

for i, ex in enumerate(EXERCISES):
    with st.expander(ex["title"], expanded=(i == 0)):
        st.markdown(f"**Task:** {ex['task']}")
        st.markdown("**Steps**")
        for j, step in enumerate(ex["steps"], 1):
            st.markdown(f"{j}. {step}")
        st.markdown("**Questions to answer**")
        for q in ex["questions"]:
            st.markdown(f"- {q}")
        st.text_area("Your observations and answers", key=f"ex_answer_{i}", height=120)
